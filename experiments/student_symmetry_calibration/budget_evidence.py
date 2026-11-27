"""Select an orbit-variance budget from clean target support labels."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiment import CIFAR10, HERE, accuracy, class_split, coordinate_priors, data_partitions, learn_domain

BUDGETS = (1., .75, .5, .25, .1)
ROTATIONS = 4


def prior_bank(probe, seed):
    width = probe.shape[-1]
    ridge = 1e-4 * max(np.mean(probe ** 2), 1e-12)
    rng = np.random.default_rng(seed + 733)
    banks = [coordinate_priors(probe, 4., np.eye(width), ridge, BUDGETS[1:], rng)[0]
             for _ in range(ROTATIONS)]
    operators = banks[0]
    orbit = [operators["natural_iso"], *(operators[f"natural_budget_{b:g}"] for b in BUDGETS[1:])]
    rotated = [bank[f"natural_budget_rotated_{b:g}"] for bank in banks for b in BUDGETS[1:]]
    return np.stack((*orbit, *rotated, operators["ordinary_orbit"]))


def budget_selection(priors, x, labels, noise, permutations, alpha, rng):
    shuffled = np.stack([labels, *(labels[rng.permutation(len(labels))] for _ in range(permutations))])
    width, classes = x.shape[1], labels.shape[1]
    rhs = (x.T @ shuffled).transpose(1, 0, 2).reshape(width, -1)
    factors = np.linalg.cholesky(priors)
    design = x @ factors
    posterior = np.linalg.cholesky(np.eye(width) + design.transpose(0, 2, 1) @ design / noise ** 2)
    whitened = np.linalg.solve(posterior, factors.transpose(0, 2, 1) @ rhs / noise ** 2)
    samples = whitened.reshape(len(priors), width, permutations + 1, classes)
    scores = .5 * np.sum(samples ** 2, axis=(1, 3))
    count = len(BUDGETS)
    base = scores[:count]
    logdet = np.log(np.diagonal(posterior, axis1=1, axis2=2)).sum(axis=1)
    ordinary_evidence = base[:, 0] - (classes - 1) * logdet[:count]
    trace = np.sum(np.linalg.solve(posterior[:count], design[:count].transpose(0, 2, 1)) ** 2,
                   axis=1) / noise ** 2
    weights = factors @ np.linalg.solve(posterior.transpose(0, 2, 1), samples[:, :, 0, :])
    residual = (labels - x @ weights[:count]) / np.maximum(1 - trace[..., None], 1e-9)
    loo = np.mean(residual ** 2, axis=(1, 2))
    centered = base[:, 0] - base[:, 1:].mean(axis=1)
    references = scores[count:-1].reshape(ROTATIONS, count - 1, permutations + 1)
    contrast = base[1:] - references.mean(axis=0)
    contrast -= contrast.mean(axis=1, keepdims=True)
    contrast /= np.maximum(contrast.std(axis=1, keepdims=True), 1e-12)
    maximum = contrast.max(axis=0)
    pvalue = float(np.mean(maximum >= maximum[0]))
    best = int(contrast[:, 0].argmax()) + 1 if maximum[0] > 0 else 0
    chosen = {
        "evidence": int(ordinary_evidence.argmax()),
        "centered": int(centered.argmax()),
        "loo": int(loo.argmin()),
        "orientation": best,
        "orientation_guarded": best if pvalue <= alpha else 0,
    }
    return weights, chosen, pvalue


@torch.inference_mode()
def rotated_images(dataset, device, batch_size):
    result = {}
    for label, images in dataset.items():
        chunks = []
        signs = torch.ones(len(images), dtype=torch.float32, device=device)
        signs[::2] = -1
        for batch, sign in zip(images.split(batch_size), signs.split(batch_size), strict=True):
            batch = batch.to(device=device, dtype=torch.float32)
            angle = sign * np.deg2rad(22.)
            cos, sin = angle.cos(), angle.sin()
            zero = torch.zeros_like(angle)
            transforms = torch.stack((cos, -sin, zero, sin, cos, zero), dim=1).reshape(-1, 2, 3)
            grid = F.affine_grid(transforms, batch.shape, align_corners=False)
            chunks.append(F.grid_sample(batch, grid, align_corners=False, padding_mode="reflection")
                          .round().to(torch.uint8).cpu())
        result[label] = torch.cat(chunks)
    return result


def run(train, test, args, seed, device):
    source, target = class_split(seed, True)
    x, probe, support, query = data_partitions(train, test, args, seed, device, source, target)
    shifted = rotated_images(query, device, args.batch_size)
    features = learn_domain(x, probe, source, support, query, args, seed, device,
                            additional_target_tests={"rotated": shifted})
    mean = features["probe"].mean(axis=(0, 1), dtype=np.float64)
    priors = prior_bank(features["probe"] - mean, seed)
    clean, reflected = np.concatenate([features["queries"][i] for i in target], axis=1) - mean
    rotated = np.concatenate([features["additional_queries"]["rotated"][i][0] for i in target]) - mean
    queries = np.stack((clean, reflected, rotated))
    truth = np.concatenate([np.full(len(features["queries"][label][0]), i) for i, label in enumerate(target)])
    rng = np.random.default_rng(seed + 422)
    permutation_rng = np.random.default_rng(seed + 563)
    rows = {str(count): [] for count in args.shots}
    for _ in range(args.episodes):
        selected = {i: rng.choice(len(features["supports"][i][0]), max(args.shots), replace=False) for i in target}
        for count in args.shots:
            support_clean = np.concatenate([features["supports"][i][0, selected[i][:count]] for i in target]) - mean
            labels = np.repeat(np.eye(len(target)), count, axis=0) - 1 / len(target)
            weights, chosen, pvalue = budget_selection(priors, support_clean, labels, args.noise,
                                                       args.permutations, args.alpha, permutation_rng)
            selected_methods = {**{f"fixed_{b:g}": i for i, b in enumerate(BUDGETS)},
                                **chosen, "ordinary_orbit": len(priors) - 1}
            logits = queries[None] @ weights[:, None]
            scores = {}
            for method, i in selected_methods.items():
                row = accuracy(logits[i, :2], truth)
                row["rotated_accuracy"] = float(np.mean(logits[i, 2].argmax(axis=1) == truth))
                scores[method] = row
            rows[str(count)].append({"scores": scores, "chosen": {k: BUDGETS[i] for k, i in chosen.items()},
                                      "pvalue": pvalue})
    print(seed, "completed", flush=True)
    return {"seed": seed, "source_classes": source, "target_classes": target, "shots": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/tmp/ibt-cifar-data"))
    parser.add_argument("--output", type=Path, default=HERE / "results/budget_evidence/metrics.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(601, 621)))
    parser.add_argument("--shots", type=int, nargs="+", default=[5, 20, 60])
    parser.add_argument("--teacher-epochs", type=int, default=26)
    parser.add_argument("--student-epochs", type=int, default=22)
    parser.add_argument("--source-examples", type=int, default=900)
    parser.add_argument("--probe-examples", type=int, default=128)
    parser.add_argument("--target-pool", type=int, default=600)
    parser.add_argument("--test-per-class", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--permutations", type=int, default=39)
    parser.add_argument("--noise", type=float, default=.45)
    parser.add_argument("--alpha", type=float, default=.10)
    args = parser.parse_args()
    if args.permutations < 1 or not 0 < args.alpha < 1 or not np.isfinite(args.noise) or args.noise <= 0:
        parser.error("need positive permutation count and noise, and alpha in (0, 1)")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    train, test = (CIFAR10(args.data, train=split, download=True) for split in (True, False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "budgets": BUDGETS, "orientation_controls": ROTATIONS, "runs": []}
    start = time.monotonic()
    for seed in args.seeds:
        output["runs"].append(run(train, test, args, seed, device))
        output["elapsed_seconds"] = time.monotonic() - start
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
