"""Infer action signs from unlabelled target orbit pairs."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit, logit

from experiment import GTSRB, HERE, discover_basis, embed, fit_student, fit_teacher
from reflection_evidence.experiment import TASKS, preprocess, prior_kernels
from signed_budget import (judge_probabilities, predictive_bank, predictive_probabilities,
                           probabilities, support_fit)


def head_projection(x, kernel, noise):
    products = x @ kernel
    chol = np.linalg.cholesky(products @ x.T + noise ** 2 * np.eye(len(x)))
    features = np.linalg.solve(chol, products)
    projection = np.linalg.solve(chol.T, features).T
    return projection, kernel - features.T @ features


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


def direction_log_odds(pairs, directions, shrink):
    """Evaluate each linear head on the same pair geometry."""
    scores = (pairs - pairs.mean(axis=1, keepdims=True)) @ directions
    left, right = scores
    energy = np.sum(left ** 2, axis=0) * np.sum(right ** 2, axis=0)
    rho = np.clip(np.sum(left * right, axis=0) / np.sqrt(np.maximum(energy, 1e-24)), -1, 1)
    return 2 * np.arctanh(rho * len(left) / (len(left) + shrink))


def centered_action_weights(labels, pairs, shuffled, projection, mean, clean, prior, shrink, draws, rng):
    random = rng.normal(size=(len(mean), draws))
    permuted = np.stack([labels[rng.permutation(len(labels))] for _ in range(draws)], axis=1)
    extra = np.column_stack((mean, random, projection @ permuted))
    odds = direction_log_odds(pairs, extra, shrink)
    random_offset = odds[1:draws + 1].mean()
    label_offset = odds[draws + 1:].mean()
    geometry_offset = logit(prior)
    clean_odds = logit(np.clip(clean, 1e-12, 1 - 1e-12))
    placebo = direction_log_odds(shuffled, np.column_stack((mean, extra[:, draws + 1:])), shrink)
    return {
        "centered_geometry": float(expit(clean_odds + odds[0] - geometry_offset)),
        "centered_random": float(expit(clean_odds + odds[0] - random_offset)),
        "centered_labels": float(expit(clean_odds + odds[0] - label_offset)),
        "centered_labels_shuffled": float(expit(clean_odds + placebo[0] - placebo[1:].mean())),
    }, {"head_log_odds": float(odds[0]), "geometry_offset": float(geometry_offset),
        "random_offset": float(random_offset), "label_offset": float(label_offset)}


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


def action_estimates(x, labels, pairs, parity, isotropic, noise, shrink, rng, null_rng=None, draws=63):
    _, evidence = support_fit(parity, x, labels, noise)
    clean = float(probabilities(evidence["centered"])[0])
    projection, covariance = head_projection(x, isotropic, noise)
    mean = projection @ labels
    moment = np.outer(mean, mean)
    point = sign_weight(pairs, moment, shrink)
    posterior = sign_weight(pairs, moment + covariance, shrink)
    shuffled = pairs.copy()
    shuffled[1] = shuffled[1, rng.permutation(len(shuffled[1]))]
    placebo = sign_weight(shuffled, moment, shrink)
    prior = sign_weight(pairs, isotropic, shrink)
    weights = {"clean": clean, "pair_only": point, "combined": combine_signs(clean, point),
               "combined_posterior": combine_signs(clean, posterior), "combined_shuffled": combine_signs(clean, placebo),
               "prior_geometry": prior, "uniform": .5}
    diagnostics = {}
    if null_rng is not None:
        centered, diagnostics = centered_action_weights(
            labels, pairs, shuffled, projection, mean, clean, prior, shrink, draws, null_rng)
        weights.update(centered)
    return weights, diagnostics


def track_cross_validation(x, labels, parity, noise):
    """Score held-out balanced track pairs and center the score under fold-wise label swaps."""
    folds = list(zip(np.flatnonzero(labels < 0), np.flatnonzero(labels > 0), strict=True))
    null_basis = np.zeros((len(labels), len(folds)))
    for i, indices in enumerate(folds):
        null_basis[list(indices), i] = labels[list(indices)]
    covariance = x @ parity @ x.T + noise ** 2 * np.eye(len(labels))
    precision = np.linalg.solve(covariance, np.eye(len(labels)))
    projected = precision @ np.column_stack((labels, null_basis))
    raw = np.zeros(len(parity))
    centered = np.zeros_like(raw)
    for indices in folds:
        held = list(indices)
        block = precision[:, held, :][:, :, held]
        values = projected[:, held, :]
        quadratic = (values * np.linalg.solve(block, values)).sum(axis=1)
        raw += .5 * (np.linalg.slogdet(block)[1] - quadratic[:, 0])
        centered += .5 * (quadratic[:, 1:].sum(axis=1) - quadratic[:, 0])
    return raw, centered


def cross_validated_weights(x, labels, parity, noise, null_diagnostics):
    raw, centered = track_cross_validation(x, labels, parity, noise)
    balance = null_diagnostics["head_log_odds"] - null_diagnostics["label_offset"]
    return {
        "track_cv": float(probabilities(raw)[0]),
        "track_cv_centered": float(probabilities(centered)[0]),
        "track_cv_pairs": float(expit(centered[0] - centered[1] + balance)),
    }


def block_precision(observations):
    centered = observations - observations.mean(axis=0)
    covariance = centered.T @ centered / len(centered)
    dimension = len(covariance)
    scale = max(np.trace(covariance) / dimension, 1e-12)
    inverse = np.linalg.solve(covariance + scale * np.eye(dimension), np.eye(dimension))
    return inverse / np.trace(inverse)


def calibrated_block_priors(pairs, source_pairs, odd, parity):
    """Keep block shares before matching predicted variance across actions."""
    pool = pairs.reshape(-1, len(odd))
    reference = np.mean(np.sum((pool @ parity) * pool, axis=-1), axis=-1)
    target_scale = max(float(reference.mean()), 1e-12)
    shapes = {kind: parity.copy() for kind in ("scale_only", "source_blocks", "target_marginals", "target_pairs")}
    for k, allowed in enumerate((~odd, odd)):
        sign = (1, -1)[k]
        region = np.ix_(allowed, allowed)
        source = (source_pairs[0] + sign * source_pairs[1])[:, allowed] / 2
        target = (pairs[0] + sign * pairs[1])[:, allowed] / 2
        mass = np.trace(parity[k][region])
        for kind, observations in (("source_blocks", source), ("target_marginals", pool[:, allowed]),
                                   ("target_pairs", target)):
            shapes[kind][k][region] = mass * block_precision(observations)
    factors = {}
    for kind, priors in shapes.items():
        scale = np.mean(np.sum((pool @ priors) * pool, axis=-1), axis=-1)
        ratio = target_scale / np.maximum(scale, 1e-12)
        shapes[kind] = priors * ratio[:, None, None]
        factors[kind] = ratio.tolist()
    return shapes, factors


def measure_task(pair, sign, train, test, tracks, parity, isotropic, args, rng, null_rng=None, block_source=None):
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
        augmented_priors, factors = {}, {}
        if block_source is not None:
            augmented_priors, factors = calibrated_block_priors(pairs, *block_source, parity)
        for shots in args.shots:
            x = np.concatenate([train[label][0, indices[:shots]]
                                for label, indices in zip(pair, selected, strict=True)])
            labels = np.repeat([-1., 1.], shots)
            weights, diagnostics = action_estimates(
                x, labels, pairs, parity, isotropic, args.noise, args.shrink, rng, null_rng, args.null_draws)
            if args.track_cv:
                weights.update(cross_validated_weights(x, labels, parity, args.noise, diagnostics))
            for kind, prior in augmented_priors.items():
                score, _ = track_cross_validation(x, labels, prior, args.noise)
                weights[kind] = float(probabilities(score)[0])
            _, posterior = predictive_probabilities(cache, x, labels, args.noise, query.shape[:-1])
            weights["known_sign"] = float(sign == 1)
            metrics = {key: judge_probabilities(p * posterior[0] + (1 - p) * posterior[1], truth, sign)
                       for key, p in weights.items()}
            rows[str(shots)].append({"metrics": metrics, "same_weights": weights,
                                     "calibration_tracks": len(calibration), "null_diagnostics": diagnostics,
                                     "block_scale_factors": factors})
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
    null_rng = np.random.default_rng(seed + 600) if args.centered_null else None
    kernels = prior_kernels(energies.cpu().numpy(), rng)
    parity = np.concatenate((kernels["fixed_even"], kernels["fixed_odd"]))
    block_source = None
    if args.target_blocks:
        source_pairs = np.stack((reference.cpu().numpy(), reflected.cpu().numpy())) - center
        block_source = (source_pairs, energies.cpu().numpy() > .5)
    tasks = {family: {"-".join(map(str, pair)): measure_task(
        pair, 1 if family == "invariant" else -1, train, test, tracks, parity, kernels["isotropic"][0],
        args, rng, null_rng, block_source)
        for pair in pairs} for family, pairs in TASKS.items()}
    print(seed, "completed", flush=True)
    return {"seed": seed, "tasks": tasks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/tmp/ibt-gtsrb-data"))
    parser.add_argument("--output", type=Path)
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
    parser.add_argument("--centered-null", action="store_true")
    parser.add_argument("--track-cv", action="store_true")
    parser.add_argument("--target-blocks", action="store_true")
    parser.add_argument("--null-draws", type=int, default=63)
    args = parser.parse_args()
    if args.target_blocks:
        args.track_cv = True
    if args.track_cv:
        args.centered_null = True
    if args.output is None:
        name = "centered_action" if args.centered_null else "unlabeled_action"
        if args.track_cv:
            name = "cross_validated_action"
        if args.target_blocks:
            name = "target_block_action"
        args.output = HERE / "results" / name / "metrics.json"
    if (min(args.shots) <= 0 or not np.isfinite(args.noise) or args.noise <= 0
            or args.unlabeled_tracks < 2 or not np.isfinite(args.shrink) or args.shrink <= 0 or args.null_draws < 1):
        parser.error("need positive shots, noise, shrink and null draws, and at least two unlabelled tracks")
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
