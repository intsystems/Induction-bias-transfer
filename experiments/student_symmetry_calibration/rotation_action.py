"""Validate joint label-action contrasts on half-turned handwritten digits."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.datasets import MNIST

from experiment import HERE, balanced_indices, discover_basis, embed, fit_student, fit_teacher
from reflection_evidence.experiment import prior_kernels
from unlabeled_action import measure_task

SOURCE = (2, 3, 4, 5, 6)
DIGITS = (0, 1, 7, 8, 9)
TASKS = {"invariant": ((0, 8), (1, 7), (1, 9), (7, 9)),
         "antisymmetric": ((7,), (9,), (1, 7), (7, 9))}


def half_turn(x):
    return x.flip((-2, -1))


def images(dataset, indices):
    return F.pad(dataset.data[indices].unsqueeze(1), (2, 2, 2, 2)).expand(-1, 3, -1, -1)


def identity_groups(views, digits, rng):
    result = {}
    for group, digit in enumerate(digits):
        n = views[digit].shape[1]
        orientation = np.zeros(n, dtype=int)
        orientation[rng.permutation(n)[:n // 2]] = 1
        index = np.arange(n)
        result[group] = np.stack((views[digit][orientation, index], views[digit][1 - orientation, index]))
    return result


def orientation_groups(views, digits):
    first, second = [], []
    for digit in digits:
        features = views[digit]
        middle = features.shape[1] // 2
        first.append(features[:, :middle])
        second.append(features[::-1, middle:2 * middle])
    return {0: np.concatenate(first, axis=1), 1: np.concatenate(second, axis=1)}


def run(train, test, args, device, seed):
    generator = torch.Generator().manual_seed(seed)
    selected = balanced_indices(train.targets, SOURCE, args.source_examples + args.probe_examples, generator)
    raw = images(train, selected[:, :args.source_examples].flatten()).to(device, dtype=torch.float32) / 255
    probe = images(train, selected[:, args.source_examples:].flatten()).to(device, dtype=torch.float32) / 255
    labels = torch.arange(len(SOURCE), device=device).repeat_interleave(args.source_examples)
    teacher = fit_teacher(raw, labels, args, seed, classes=len(SOURCE), view=half_turn)
    mean, basis, energies = discover_basis(teacher, probe[::2], args)
    student = fit_student(raw, teacher, basis, mean, args, seed + 100)
    with torch.inference_mode():
        center = ((student(probe[1::2]) + student(half_turn(probe[1::2]))).mean(0) / 2).cpu().numpy()
    train_ids = balanced_indices(train.targets, DIGITS, args.target_pool, generator)
    test_ids = balanced_indices(test.targets, DIGITS, args.test_per_class, generator)
    features = [{k: a - center for k, a in embed(
        student, {digit: images(split, idx) for digit, idx in zip(DIGITS, selected_ids, strict=True)},
        args, device, view=half_turn).items()} for split, selected_ids in ((train, train_ids), (test, test_ids))]
    rng = np.random.default_rng(seed)
    orientation_rng = np.random.default_rng(seed + 210)
    null_rng = np.random.default_rng(seed + 600)
    priors = prior_kernels(energies.cpu().numpy(), rng)
    parity = np.concatenate((priors["fixed_even"], priors["fixed_odd"]))
    tasks = {}
    for family, definitions in TASKS.items():
        tasks[family] = {}
        for digits in definitions:
            if family == "invariant":
                train_views, test_views = (identity_groups(view, digits, orientation_rng) for view in features)
            else:
                train_views, test_views = (orientation_groups(view, digits) for view in features)
            sample_ids = {k: [[j] for j in range(v.shape[1])] for k, v in train_views.items()}
            tasks[family]["+".join(map(str, digits))] = measure_task(
                (0, 1), 1 if family == "invariant" else -1, train_views, test_views, sample_ids,
                parity, priors["isotropic"][0], args, rng, null_rng)
    print(seed, "completed", flush=True)
    return {"seed": seed, "tasks": tasks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE.parent / "image_shift/data")
    parser.add_argument("--output", type=Path, default=HERE / "results/rotation_action/metrics.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(411, 427)))
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4, 5])
    parser.add_argument("--source-examples", type=int, default=600)
    parser.add_argument("--probe-examples", type=int, default=100)
    parser.add_argument("--teacher-epochs", type=int, default=18)
    parser.add_argument("--student-epochs", type=int, default=16)
    parser.add_argument("--target-pool", type=int, default=480)
    parser.add_argument("--test-per-class", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=8)
    args = parser.parse_args()
    if min(args.shots) < 1 or args.target_pool // 2 <= max(args.shots) or args.test_per_class < 2:
        parser.error("need at least one test image and space for disjoint support and calibration in each class")
    args.noise, args.shrink, args.null_draws, args.unlabeled_tracks = .5, 4., 63, 12
    args.track_cv, args.pair_evidence = True, True
    torch.set_num_threads(2)
    device = torch.device(args.device)
    train, test = (MNIST(args.data, train=flag, download=True) for flag in (True, False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "source_digits": SOURCE, "tasks": TASKS, "runs": []}
    start = time.monotonic()
    for seed in args.seeds:
        output["runs"].append(run(train, test, args, device, seed))
        output["elapsed_seconds"] = time.monotonic() - start
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(output, allow_nan=False) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
