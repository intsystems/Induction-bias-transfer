"""Calibrate a student orbit prior from a mixture of paired and mismatched views."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiment import (CIFAR10, HERE, budget_weights, class_split, data_partitions,
                        evaluate, learn_domain)


def mismatched_indices(size, fraction, rng):
    """Permute a nested subset with no fixed points, preserving both marginals."""
    count = round(size * fraction)
    order = rng.permutation(size)
    selected = order[:count] if count >= 2 else order[:0]
    mapping = np.arange(size)
    mapping[selected] = np.roll(selected, 1)
    return mapping


def covariance(residual, weights):
    return residual.T @ (residual * weights[:, None]) / weights.sum()


def pair_scores(probe, odd):
    even = probe[..., ~odd]
    total = even.reshape(-1, even.shape[-1])
    total = total - total.mean(axis=0)
    gram = total.T @ total / len(total)
    ridge = 0.05 * max(np.trace(gram) / len(gram), 1e-12)
    factor = np.linalg.cholesky(gram + ridge * np.eye(len(gram)))
    coordinates = np.linalg.solve(factor, even.reshape(-1, even.shape[-1]).T).T.reshape(even.shape)

    def scores(left, right):
        distance = coordinates[0, left] - coordinates[1, right]
        return np.log(np.mean(distance ** 2, axis=1) + 1e-10)

    return scores


def estimate_confidence(positive, negative, observed):
    def log_density(sample):
        variance = max(float(sample.var()), 0.05)
        return -0.5 * ((observed - sample.mean()) ** 2 / variance + np.log(variance))

    matched, shuffled = log_density(positive), log_density(negative)
    clean_fraction = 0.5
    for _ in range(100):
        good = np.log(clean_fraction) + matched
        bad = np.log1p(-clean_fraction) + shuffled
        confidence = np.exp(good - np.logaddexp(good, bad))
        estimate = float(np.clip(confidence.mean(), 0.005, 0.995))
        if abs(estimate - clean_fraction) < 1e-7:
            break
        clean_fraction = estimate
    return np.maximum(confidence, np.finfo(float).tiny ** .5)


def budget_prior(orbit, whitener, budget, rotation=None):
    energies, axes = np.linalg.eigh(orbit)
    weights, _, achieved = budget_weights(np.maximum(energies, 0), budget)
    axes = whitener @ (axes if rotation is None else rotation @ axes)
    return (axes * weights) @ axes.T, achieved


def run(train, test, args, seed, device):
    source, target = class_split(seed, True)
    x, probe, support, query = data_partitions(train, test, args, seed, device, source, target)
    features = learn_domain(x, probe, source, support, query, args, seed, device)
    probe = features["probe"].astype(np.float64)
    total = probe.reshape(-1, probe.shape[-1])
    centered = total - total.mean(axis=0)
    width = centered.shape[-1]
    identity = np.eye(width)
    ridge = 1e-4 * max(np.mean(centered ** 2), 1e-12)
    whitener = np.linalg.solve(np.linalg.cholesky(centered.T @ centered / len(total) + ridge * identity).T,
                               identity)
    rng = np.random.default_rng(seed + 7700)
    order = rng.permutation(probe.shape[1])
    anchors, candidates = order[:args.anchors], order[args.anchors:]
    scores = pair_scores(probe, features["odd"])
    positive, negative = scores(anchors, anchors), scores(anchors, np.roll(anchors, 1))

    def residual(left, right):
        return (probe[0, left] - probe[1, right]) @ whitener / 2

    anchor_residual = residual(anchors, anchors)
    clean = covariance(residual(candidates, candidates), np.ones(len(candidates)))
    anchor = covariance(anchor_residual, np.ones(len(anchors)))
    rotation, _ = np.linalg.qr(rng.normal(size=(width, width)))
    centered_support = {k: v - total.mean(axis=0) for k, v in features["supports"].items()}
    centered_query = {k: v - total.mean(axis=0) for k, v in features["queries"].items()}
    measurements = {}
    for fraction in args.fractions:
        mapping = mismatched_indices(len(candidates), fraction, np.random.default_rng(seed + 7200))
        assigned = candidates[mapping]
        observed = residual(candidates, assigned)
        matched = mapping == np.arange(len(mapping))
        confidence = estimate_confidence(positive, negative, scores(candidates, assigned))
        matrices = {
            "trusted_only": anchor,
            "clean_full": clean,
            "naive": covariance(observed, np.ones(len(observed))),
            "confidence": covariance(observed, confidence),
            "oracle": covariance(observed, matched.astype(float)),
        }
        priors, achieved = {}, {}
        for key, matrix in matrices.items():
            priors[key], achieved[key] = budget_prior(matrix, whitener, args.budget)
        priors["confidence_rotated"] = budget_prior(matrices["confidence"], whitener, args.budget, rotation)[0]
        measurements[str(fraction)] = {
            "scores": evaluate(priors, centered_support, centered_query, target, args,
                               np.random.default_rng(seed + 422)),
            "actual_clean_fraction": float(matched.mean()),
            "estimated_clean_fraction": float(confidence.mean()),
            "effective_pairs": float(confidence.sum() ** 2 / (confidence @ confidence)),
            "achieved_budgets": achieved,
            "relative_orbit_error": {key: float(np.linalg.norm(value - clean) / np.linalg.norm(clean))
                                     for key, value in matrices.items()},
        }
    print(seed, "completed", flush=True)
    return {"seed": seed, "source_classes": source, "target_classes": target, "measurements": measurements}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/tmp/ibt-cifar-data"))
    parser.add_argument("--output", type=Path, default=HERE / "results/pair_noise/metrics.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(89, 101)))
    parser.add_argument("--fractions", type=float, nargs="+", default=[0., .25, .5, .75])
    parser.add_argument("--anchors", type=int, default=32)
    parser.add_argument("--budget", type=float, default=.25)
    parser.add_argument("--shots", type=int, nargs="+", default=[5, 20, 60])
    parser.add_argument("--teacher-epochs", type=int, default=26)
    parser.add_argument("--student-epochs", type=int, default=22)
    parser.add_argument("--source-examples", type=int, default=900)
    parser.add_argument("--probe-examples", type=int, default=128)
    parser.add_argument("--target-pool", type=int, default=600)
    parser.add_argument("--test-per-class", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--noise", type=float, default=.45)
    args = parser.parse_args()
    if args.anchors < 2 or 4 * args.probe_examples - args.anchors < 4:
        parser.error("need at least two trusted pairs and four candidate pairs")
    if not 0 < args.budget < 1 or any(not 0 <= f < 1 for f in args.fractions):
        parser.error("invalid variance budget or mismatch fraction")
    if any(round((4 * args.probe_examples - args.anchors) * f) >=
           4 * args.probe_examples - args.anchors for f in args.fractions):
        parser.error("at least one clean candidate must remain for the oracle")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    train, test = (CIFAR10(args.data, train=split, download=True) for split in (True, False))
    data = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "runs": [], "torch_version": torch.__version__}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    for seed in args.seeds:
        data["runs"].append(run(train, test, args, seed, device))
        data["elapsed_seconds"] = time.monotonic() - start
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
