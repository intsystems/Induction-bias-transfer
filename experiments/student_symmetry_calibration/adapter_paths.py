"""Transfer composition geometry to an adapter that is fine-tuned on clean target labels."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiment import CIFAR10, HERE, class_split, data_partitions, discover_basis, fit_student, fit_teacher
from generator_budget import TEST_VIEWS, embed_orbit, source_priors


def source_adapter_priors(views, budget, seed):
    priors, diagnostics, model = source_priors(views, budget, seed, path_composition=True)
    whitener = model[2]
    precisions = {}
    methods = {"path": "composition_mean_prediction", "rotated": "path_rotated_0",
               "isotropic": "isotropic", "split": "split_balanced"}
    for name, key in methods.items():
        precision = whitener.T @ np.linalg.solve(priors[key], whitener)
        precision = (precision + precision.T) / 2
        precisions[name] = precision / (np.trace(precision) / len(precision))
    return whitener, precisions, diagnostics["composition_operators"]["relative_moment_error"]["mean_prediction"]


def coordinate_adapter(adapter, whitener, offset):
    original = adapter.weight.detach().T
    weight = torch.linalg.solve(whitener, original).T.contiguous()
    bias = adapter.bias.detach() + offset @ original
    return weight, bias


def fit_target(center, bias, init_head, supports, labels, queries, truth, precision, args, *, freeze=False):
    weight = nn.Parameter(center.clone(), requires_grad=not freeze)
    intercept = nn.Parameter(bias.clone(), requires_grad=not freeze)
    head = nn.Parameter(init_head.clone())
    head_bias = nn.Parameter(center.new_zeros(len(head)))
    optimizer = torch.optim.AdamW([head, head_bias, *(() if freeze else (weight, intercept))], lr=args.lr)
    for _ in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = F.linear(F.linear(supports, weight, intercept), head, head_bias)
        loss = F.cross_entropy(logits, labels)
        if precision is not None:
            displacement = weight - center
            loss = loss + args.penalty * ((displacement @ precision) * displacement).mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        correct = F.linear(F.linear(queries, weight, intercept), head, head_bias).argmax(-1) == truth
        residual = (weight - center).square().mean().sqrt()
        return {"clean": correct[0].float().mean().item(), "all_views": correct.all(0).float().mean().item(),
                "adapter_rms_change": residual.item(),
                **{key: (correct[0] & correct[index]).float().mean().item() for key, index in TEST_VIEWS.items()}}


def run(train, test, args, seed, device):
    source, target = class_split(seed, True)
    x, probe, support, queries = data_partitions(train, test, args, seed, device, source, target)
    y = torch.arange(len(source), device=device).repeat_interleave(args.source_examples)
    teacher = fit_teacher(x, y, args, seed, classes=len(source))
    mean, basis, _ = discover_basis(teacher, probe[::2], args)
    student = fit_student(x, teacher, basis, mean, args, seed + 100)
    stem = student[:-1]
    source_views = embed_orbit(stem, probe[1::2], device, args.batch_size)
    offset = source_views.mean(axis=(0, 1))
    whitener, priors, moment_error = source_adapter_priors(source_views - offset, args.budget, seed)
    weights, bias = coordinate_adapter(student[-1],
                                       torch.as_tensor(whitener, device=device, dtype=torch.float32),
                                       torch.as_tensor(offset, device=device, dtype=torch.float32))
    precisions = {key: torch.as_tensor(value, device=device, dtype=torch.float32) for key, value in priors.items()}
    def coordinates(views):
        return torch.as_tensor((views - offset) @ whitener, device=device, dtype=weights.dtype)

    supports = {label: coordinates(embed_orbit(stem, v, device, args.batch_size)[0]) for label, v in support.items()}
    target_views = coordinates(np.concatenate([embed_orbit(stem, queries[k], device, args.batch_size)
                                                for k in target], axis=1))
    truth = torch.arange(len(target), device=device).repeat_interleave(args.test_per_class)
    rng = np.random.default_rng(seed + 422)
    rows = {str(n): [] for n in args.shots}
    for episode in range(args.episodes):
        chosen = {k: rng.choice(len(supports[k]), max(args.shots), replace=False) for k in target}
        generator = torch.Generator(device=device).manual_seed(seed * 1000 + episode)
        head = torch.randn(len(target), len(weights), device=device, generator=generator) * .02
        for n in args.shots:
            design = torch.cat([supports[k][chosen[k][:n]] for k in target])
            labels = torch.arange(len(target), device=device).repeat_interleave(n)
            modes = {"frozen": None, "initialization": None, **precisions}
            rows[str(n)].append({name: fit_target(weights, bias, head, design, labels, target_views, truth,
                                                  precision, args, freeze=name == "frozen")
                                 for name, precision in modes.items()})
    print(seed, "completed", flush=True)
    return {"seed": seed, "latent_width": len(weights), "source_moment_error": moment_error, "shots": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/tmp/ibt-cifar-data"))
    parser.add_argument("--output", type=Path, default=HERE / "results/adapter_paths/metrics.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1021, 1037)))
    parser.add_argument("--shots", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--source-examples", type=int, default=900)
    parser.add_argument("--probe-examples", type=int, default=128)
    parser.add_argument("--teacher-epochs", type=int, default=26)
    parser.add_argument("--student-epochs", type=int, default=22)
    parser.add_argument("--target-pool", type=int, default=600)
    parser.add_argument("--test-per-class", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=.003)
    parser.add_argument("--penalty", type=float, default=25.)
    parser.add_argument("--budget", type=float, default=.25)
    args = parser.parse_args()
    if args.probe_examples < 2 or not 0 < args.budget < 1 or args.penalty < 0:
        parser.error("need at least two probe examples per class, a variance budget in (0, 1), and a nonnegative penalty")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    train, test = (CIFAR10(args.data, train=flag, download=True) for flag in (True, False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "runs": []}
    start = time.monotonic()
    for seed in args.seeds:
        output["runs"].append(run(train, test, args, seed, device))
        output["elapsed_seconds"] = time.monotonic() - start
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(output, allow_nan=False) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
