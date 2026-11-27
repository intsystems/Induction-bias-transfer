"""Infer action signs from unlabelled target orbit pairs."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiment import GTSRB, HERE, discover_basis, embed, fit_student, fit_teacher
from reflection_evidence.experiment import TASKS, preprocess, prior_kernels
from signed_budget import (judge_probabilities, predictive_bank, predictive_probabilities,
                           probabilities, support_fit)


def common_head(x, labels, kernel, noise):
    products = x @ kernel
    chol = np.linalg.cholesky(products @ x.T + noise ** 2 * np.eye(len(x)))
    features = np.linalg.solve(chol, products)
    mean = features.T @ np.linalg.solve(chol, labels)
    return mean, kernel - features.T @ features


def sign_weight(pairs, weight_moment, shrink):
    """Compare signed agreement after removing each view's mean."""
    left, right = pairs - pairs.mean(axis=1, keepdims=True)
    cross = np.sum((left @ weight_moment) * right)
    energy = np.sum((left @ weight_moment) * left) * np.sum((right @ weight_moment) * right)
    correlation = float(np.clip(cross / np.sqrt(max(energy, 1e-24)), -1, 1))
    count = len(left)
    return .5 * (1 + correlation * count / (count + shrink))


def combine_signs(first, second):
    return float(probabilities(np.log([max(first * second, 1e-12),
                                       max((1 - first) * (1 - second), 1e-12)]))[0])


def allocate_episode(pair, tracks, shots, calibration_tracks, rng):
    support, available = [], []
    for label in pair:
        groups = tracks[label]
        chosen = rng.permutation(len(groups))
        support.append([rng.choice(groups[k]) for k in chosen[:shots]])
        available.extend((label, groups[k]) for k in chosen[shots:])
    sampled = rng.permutation(len(available))[:calibration_tracks]
    calibration = [(available[k][0], rng.choice(available[k][1])) for k in sampled]
    return support, calibration


def action_estimates(x, labels, pairs, parity, isotropic, noise, shrink, rng):
    _, evidence = support_fit(parity, x, labels, noise)
    clean = float(probabilities(evidence["centered"])[0])
    mean, covariance = common_head(x, labels, isotropic, noise)
    moment = np.outer(mean, mean)
    point = sign_weight(pairs, moment, shrink)
    posterior = sign_weight(pairs, moment + covariance, shrink)
    shuffled = pairs.copy()
    shuffled[1] = shuffled[1, rng.permutation(len(shuffled[1]))]
    placebo = sign_weight(shuffled, moment, shrink)
    return {"clean": clean, "pair_only": point, "combined": combine_signs(clean, point),
            "combined_posterior": combine_signs(clean, posterior), "combined_shuffled": combine_signs(clean, placebo),
            "prior_geometry": sign_weight(pairs, isotropic, shrink), "uniform": .5}


def measure_task(pair, sign, train, test, tracks, parity, isotropic, args, rng):
    query = np.concatenate([test[label] for label in pair], axis=1)
    truth = np.concatenate([np.full(len(test[label][0]), k)
                            for label, k in zip(pair, (-1, 1), strict=True)])
    cache = predictive_bank(parity, query)
    rows = {str(n): [] for n in args.shots}
    for _ in range(args.episodes):
        selected, calibration = allocate_episode(pair, tracks, max(args.shots), args.unlabeled_tracks, rng)
        if len(calibration) < 2:
            raise ValueError(f"task {pair} needs at least two tracks outside the support")
        pairs = np.stack([train[label][:, i] for label, i in calibration], axis=1)
        for shots in args.shots:
            x = np.concatenate([train[label][0, indices[:shots]]
                                for label, indices in zip(pair, selected, strict=True)])
            labels = np.repeat([-1., 1.], shots)
            weights = action_estimates(x, labels, pairs, parity, isotropic, args.noise, args.shrink, rng)
            _, posterior = predictive_probabilities(cache, x, labels, args.noise, query.shape[:-1])
            weights["known_sign"] = float(sign == 1)
            metrics = {key: judge_probabilities(p * posterior[0] + (1 - p) * posterior[1], truth, sign)
                       for key, p in weights.items()}
            rows[str(shots)].append({"metrics": metrics, "same_weights": weights,
                                     "calibration_tracks": len(calibration)})
    return rows


def run(data, args, device, seed):
    (source, labels), (probe, _), target, target_test, tracks = data
    source = source.to(device, dtype=torch.float32) / 255
    probe = probe.to(device, dtype=torch.float32) / 255
    teacher = fit_teacher(source, labels.to(device), args, seed)
    mean, basis, energies = discover_basis(teacher, probe[::2], args)
    student = fit_student(source, teacher, basis, mean, args, seed + 100)
    with torch.inference_mode():
        reference = torch.cat([student(x) for x in probe[1::2].split(args.batch_size)])
        reflected = torch.cat([student(x.flip(-1)) for x in probe[1::2].split(args.batch_size)])
        center = ((reference + reflected).mean(dim=0) / 2).cpu().numpy().astype(np.float64)
    train = {k: v - center for k, v in embed(student, target, args, device).items()}
    test = {k: v - center for k, v in embed(student, target_test, args, device).items()}
    rng = np.random.default_rng(seed)
    kernels = prior_kernels(energies.cpu().numpy(), rng)
    parity = np.concatenate((kernels["fixed_even"], kernels["fixed_odd"]))
    tasks = {family: {"-".join(map(str, pair)): measure_task(
        pair, 1 if family == "invariant" else -1, train, test, tracks, parity, kernels["isotropic"][0], args, rng)
        for pair in pairs} for family, pairs in TASKS.items()}
    print(seed, "completed", flush=True)
    return {"seed": seed, "tasks": tasks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/tmp/ibt-gtsrb-data"))
    parser.add_argument("--output", type=Path, default=HERE / "results/unlabeled_action/metrics.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(291, 307)))
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4, 5])
    parser.add_argument("--teacher-epochs", type=int, default=20)
    parser.add_argument("--student-epochs", type=int, default=18)
    parser.add_argument("--source-examples", type=int, default=300)
    parser.add_argument("--probe-examples", type=int, default=65)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--noise", type=float, default=.5)
    parser.add_argument("--unlabeled-tracks", type=int, default=12)
    parser.add_argument("--shrink", type=float, default=4.)
    args = parser.parse_args()
    if (min(args.shots) <= 0 or not np.isfinite(args.noise) or args.noise <= 0
            or args.unlabeled_tracks < 2 or not np.isfinite(args.shrink) or args.shrink <= 0):
        parser.error("need positive shots, noise, shrink and two unlabelled tracks")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    data = preprocess(GTSRB(args.data, split="train", download=True),
                      GTSRB(args.data, split="test", download=True), args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "tasks": TASKS, "runs": []}
    started = time.monotonic()
    for seed in args.seeds:
        output["runs"].append(run(data, args, device, seed))
        output["elapsed_seconds"] = time.monotonic() - started
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(output, allow_nan=False) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
