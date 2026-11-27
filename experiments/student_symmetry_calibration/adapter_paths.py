"""Transfer composition geometry to an adapter that is fine-tuned on clean target labels."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiment import (CIFAR10, GTSRB, HERE, TRAFFIC_SOURCE, class_split, data_partitions, discover_basis,
                        fit_student, fit_teacher, paths_by_class, traffic_source)
from generator_budget import TEST_VIEWS, composition_moments, embed_orbit, source_priors


def source_adapter_priors(views, budget, seed, *, reference_views=None):
    priors, diagnostics, model = source_priors(views, budget, seed, path_composition=True,
                                              reference_probe=reference_views)
    whitener = model[2]
    precisions = {}
    methods = {"path": "composition_mean_prediction", "rotated": "path_rotated_0",
               "isotropic": "isotropic", "split": "split_balanced"}
    for name, key in methods.items():
        precision = whitener.T @ np.linalg.solve(priors[key], whitener)
        precision = (precision + precision.T) / 2
        precisions[name] = precision / (np.trace(precision) / len(precision))
    action = np.eye(len(whitener)) - np.sqrt(2) * np.linalg.solve(whitener, model[0] @ whitener)
    error = diagnostics["composition_operators"]["relative_moment_error"]["mean_prediction"]
    return whitener, precisions, error, action


def coordinate_adapter(adapter, whitener, offset):
    original = adapter.weight.detach().T
    weight = torch.linalg.solve(whitener, original).T.contiguous()
    bias = adapter.bias.detach() + offset @ original
    return weight, bias


def hybrid_precisions(path, split, seed):
    identity = np.eye(len(path))
    covariance_path = np.linalg.solve(path, identity)
    covariance_split = np.linalg.solve(split, identity)
    halfway = (path + split) / 2
    rotation, _ = np.linalg.qr(np.random.default_rng(seed + 4100).normal(size=path.shape))
    harmonic = np.linalg.solve((covariance_path + covariance_split) / 2, identity)
    harmonic = (harmonic + harmonic.T) / 2
    return {"hybrid_quarter": .25 * path + .75 * split, "hybrid_half": halfway,
            "hybrid_three_quarters": .75 * path + .25 * split,
            "hybrid_covariance": harmonic / (np.trace(harmonic) / len(harmonic)),
            "hybrid_rotated": rotation @ halfway @ rotation.T}


def teacher_row_precisions(views, seed):
    centered = views - views.mean(axis=(0, 1))
    moments, _, _ = composition_moments(centered, path_composition=True)
    energies, axes = np.linalg.eigh(moments["mean_prediction"])
    strengths = 1 / (.1 + np.maximum(energies, 0))
    strengths /= strengths.mean()
    rotation, _ = np.linalg.qr(np.random.default_rng(seed + 2100).normal(size=axes.shape))
    aligned = (axes * strengths) @ axes.T
    return {"teacher_rows": aligned,
            "rotated_rows": rotation @ aligned @ rotation.T,
            "reversed_rows": (axes * strengths[::-1]) @ axes.T}, strengths


def composition_action_penalties(stem_action, teacher_views, seed):
    centered = teacher_views - teacher_views.mean(axis=(0, 1))
    _, teacher_action, _ = composition_moments(centered, path_composition=True)
    rotation, _ = np.linalg.qr(np.random.default_rng(seed + 3100).normal(size=teacher_action.shape))
    outputs = {"teacher_action": teacher_action,
               "rotated_action": rotation @ teacher_action @ rotation.T,
               "identity_action": np.eye(len(teacher_action))}
    penalties = {}
    for name, action in outputs.items():
        trace_mean = np.sum(stem_action ** 2) / len(stem_action) + np.sum(action ** 2) / len(action)
        trace_mean -= 2 * np.trace(stem_action) * np.trace(action) / (len(stem_action) * len(action))
        penalties[name] = (stem_action, action, max(float(trace_mean), 1e-12))
    return penalties


def fit_target(center, bias, init_head, supports, labels, queries, truth, precision, args, *,
               freeze=False, radii=None, record_radii=False, row_precision=None, action_penalty=None):
    weight = nn.Parameter(center.clone(), requires_grad=not freeze)
    intercept = nn.Parameter(bias.clone(), requires_grad=not freeze)
    head = nn.Parameter(init_head.clone())
    head_bias = nn.Parameter(center.new_zeros(len(head)))
    optimizer = torch.optim.AdamW([head, head_bias, *(() if freeze else (weight, intercept))], lr=args.lr)
    history = []
    for step in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = F.linear(F.linear(supports, weight, intercept), head, head_bias)
        loss = F.cross_entropy(logits, labels)
        if precision is not None:
            displacement = weight - center
            output_penalty = displacement if row_precision is None else row_precision @ displacement
            penalty = ((displacement @ precision) * output_penalty).mean()
            if action_penalty is not None:
                stem_action, teacher_action, scale = action_penalty
                commutator = stem_action @ displacement.T - displacement.T @ teacher_action
                penalty = (penalty + commutator.square().mean() / scale) / 2
            loss = loss + args.penalty * penalty
        loss.backward()
        optimizer.step()
        if radii is not None or record_radii:
            with torch.no_grad():
                displacement = weight - center
                if radii is not None:
                    rms = displacement.square().mean().sqrt().clamp_min(1e-12)
                    weight.copy_(center + displacement * (radii[step] / rms))
                history.append((weight - center).square().mean().sqrt())
    with torch.no_grad():
        correct = F.linear(F.linear(queries, weight, intercept), head, head_bias).argmax(-1) == truth
        residual = (weight - center).square().mean().sqrt()
        metrics = {"clean": correct[0].float().mean().item(), "all_views": correct.all(0).float().mean().item(),
                   "adapter_rms_change": residual.item(),
                   **{key: (correct[0] & correct[index]).float().mean().item() for key, index in TEST_VIEWS.items()}}
        if radii is not None:
            metrics["max_rms_gap"] = (torch.stack(history) - torch.stack(radii)).abs().max().item()
        return metrics, history


def run(train, test, args, seed, device, traffic_paths=None):
    source, target = class_split(seed, True)
    x, probe, support, queries = data_partitions(train, test, args, seed, device, source, target)
    auxiliary_probe = probe[1::2] if args.compare_probe_domain else None
    if traffic_paths is not None:
        x, probe = traffic_source(traffic_paths, args, seed, device)
        source = TRAFFIC_SOURCE
    y = torch.arange(len(source), device=device).repeat_interleave(args.source_examples)
    teacher = fit_teacher(x, y, args, seed, classes=len(source))
    mean, basis, _ = discover_basis(teacher, probe[::2], args)
    student = fit_student(x, teacher, basis, mean, args, seed + 100)
    stem = student[:-1]
    source_views = embed_orbit(stem, probe[1::2], device, args.batch_size)
    offset = source_views.mean(axis=(0, 1))
    whitener, priors, moment_error, stem_action = source_adapter_priors(source_views - offset, args.budget, seed)
    weights, bias = coordinate_adapter(student[-1],
                                       torch.as_tensor(whitener, device=device, dtype=torch.float32),
                                       torch.as_tensor(offset, device=device, dtype=torch.float32))
    precisions = {key: torch.as_tensor(value, device=device, dtype=torch.float32) for key, value in priors.items()}
    row_priors = {}
    action_penalties = {}
    diagnostics = {}
    extra_precisions = {}
    if auxiliary_probe is not None:
        auxiliary_views = embed_orbit(stem, auxiliary_probe, device, args.batch_size)
        auxiliary_views -= auxiliary_views.mean(axis=(0, 1))
        _, auxiliary_priors, auxiliary_error, _ = source_adapter_priors(
            auxiliary_views, args.budget, seed, reference_views=source_views - offset
        )
        extra_precisions.update({
            f"auxiliary_{key}": torch.as_tensor(auxiliary_priors[key], device=device, dtype=weights.dtype)
            for key in ("path", "rotated", "split")
        })
        diagnostics["auxiliary_moment_error"] = auxiliary_error
    if args.hybrid_prior:
        extra_precisions.update({
            key: torch.as_tensor(value, device=device, dtype=weights.dtype)
            for key, value in hybrid_precisions(priors["path"], priors["split"], seed).items()
        })
    if args.teacher_row_prior or args.intertwining_prior:
        teacher_views = embed_orbit(lambda batch: (teacher.encode(batch) - mean) @ basis,
                                    probe[1::2], device, args.batch_size)
    if args.teacher_row_prior:
        row_matrices, strengths = teacher_row_precisions(teacher_views, seed)
        row_priors = {key: torch.as_tensor(value, device=device, dtype=weights.dtype) for key, value in row_matrices.items()}
        diagnostics["row_precision_spectrum"] = strengths.tolist()
    if args.intertwining_prior:
        matrices = composition_action_penalties(stem_action, teacher_views, seed)
        action_penalties = {
            key: (torch.as_tensor(a, device=device, dtype=weights.dtype),
                  torch.as_tensor(b, device=device, dtype=weights.dtype), scale)
            for key, (a, b, scale) in matrices.items()
        }
        diagnostics["action_penalty_normalizers"] = {key: scale for key, (_, _, scale) in matrices.items()}

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
            results = {}
            for name, precision in modes.items():
                record = args.match_update_rms and name == "isotropic"
                results[name], history = fit_target(weights, bias, head, design, labels, target_views, truth,
                                                    precision, args, freeze=name == "frozen", record_radii=record)
                if record:
                    reference_radii = history
            if args.match_update_rms:
                for name in ("path", "rotated", "split"):
                    results[f"{name}_matched"], _ = fit_target(
                        weights, bias, head, design, labels, target_views, truth, precisions[name], args,
                        radii=reference_radii
                    )
                for name, row_precision in row_priors.items():
                    results[name], _ = fit_target(
                        weights, bias, head, design, labels, target_views, truth, precisions["path"], args,
                        radii=reference_radii, row_precision=row_precision
                    )
                for name, action_penalty in action_penalties.items():
                    results[name], _ = fit_target(
                        weights, bias, head, design, labels, target_views, truth, precisions["path"], args,
                        radii=reference_radii, action_penalty=action_penalty
                    )
                for name, precision in extra_precisions.items():
                    results[f"{name}_matched"], _ = fit_target(
                        weights, bias, head, design, labels, target_views, truth, precision, args,
                        radii=reference_radii
                    )
            rows[str(n)].append(results)
    print(seed, "completed", flush=True)
    return {"seed": seed, "source_classes": source, "target_classes": target,
            "latent_width": len(weights), "source_moment_error": moment_error,
            "shots": rows, **diagnostics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/tmp/ibt-cifar-data"))
    parser.add_argument("--source-domain", choices=("cifar", "gtsrb"), default="cifar")
    parser.add_argument("--compare-probe-domain", action="store_true",
                        help="compare GTSRB and auxiliary CIFAR priors on the same GTSRB student")
    parser.add_argument("--hybrid-prior", action="store_true",
                        help="test mixtures of path and separate-generator precision")
    parser.add_argument("--gtsrb-data", type=Path, default=Path("/tmp/ibt-gtsrb-data"))
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
    parser.add_argument("--match-update-rms", action="store_true",
                        help="match the isotropic adapter displacement after each optimizer step")
    output_prior = parser.add_mutually_exclusive_group()
    output_prior.add_argument("--teacher-row-prior", action="store_true",
                              help="vary the output precision using the teacher's composition stability")
    output_prior.add_argument("--intertwining-prior", action="store_true",
                              help="penalize changes that disrupt the composition action across the adapter")
    args = parser.parse_args()
    if args.compare_probe_domain and args.source_domain != "gtsrb":
        parser.error("probe-domain comparison requires --source-domain gtsrb")
    if args.teacher_row_prior or args.intertwining_prior or args.compare_probe_domain or args.hybrid_prior:
        args.match_update_rms = True
    if args.probe_examples < 2 or not 0 < args.budget < 1 or args.penalty < 0:
        parser.error("need at least two probe examples per class, a variance budget in (0, 1), and a nonnegative penalty")
    if args.epochs < 1:
        parser.error("need at least one target optimization step")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    train, test = (CIFAR10(args.data, train=flag, download=True) for flag in (True, False))
    traffic_paths = (paths_by_class(GTSRB(args.gtsrb_data, split="train", download=True))
                     if args.source_domain == "gtsrb" else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "runs": []}
    start = time.monotonic()
    for seed in args.seeds:
        output["runs"].append(run(train, test, args, seed, device, traffic_paths))
        output["elapsed_seconds"] = time.monotonic() - start
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(output, allow_nan=False) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
