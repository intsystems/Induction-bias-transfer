"""Estimate a student orbit prior when some source views lack a counterpart."""

import argparse
import json
import time
from copy import copy
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from experiment import CIFAR10, HERE, class_split, data_partitions, evaluate, learn_domain
from pair_noise import align_to_anchors, budget_prior, covariance, matching_coordinates, pair_scores


def cost_matrix(scores, left, right):
    blocks = np.array_split(left, (len(left) + 63) // 64)
    return np.concatenate([scores(block[:, None], right[None, :]) for block in blocks])


def rejection_cost(coordinates, anchors, candidates, ridge):
    """Set the threshold from held-out true anchors and their nearest impostors."""
    positives, negatives = [], []
    for fold in np.array_split(anchors, 4):
        fitted = np.setdiff1d(anchors, fold)
        scores = pair_scores(align_to_anchors(coordinates, fitted, fitted, ridge))
        positives.extend(scores(fold, fold))
        negatives.extend(cost_matrix(scores, fold, candidates).min(axis=1))
    positive, negative = np.asarray(positives), np.asarray(negatives)
    boundaries = np.unique(np.concatenate((positive, negative)))
    thresholds = np.r_[boundaries[0], np.nextafter(boundaries, np.inf)]
    balanced = ((positive[:, None] < thresholds).mean(axis=0)
                + (negative[:, None] >= thresholds).mean(axis=0)) / 2
    best = balanced.argmax()
    return float(thresholds[best]), float(balanced[best])


def match(cost, threshold):
    rows, forced = linear_sum_assignment(cost)
    nearest = cost.argmin(axis=1)
    padded = np.column_stack((cost, np.full((len(cost), len(cost)), threshold)))
    _, optional = linear_sum_assignment(padded)
    assignments = {"posthoc": forced, "nearest": nearest, "optional": optional}
    result = {"forced": (rows, forced)}
    for name, columns in assignments.items():
        accepted = (columns < cost.shape[1]) & (padded[rows, columns] < threshold)
        result[name] = rows[accepted], columns[accepted]
    return result


def candidate_pools(probe_size, anchor_count, class_size, rng):
    """Hold the candidate-set class counts fixed even when views are replaced."""
    anchors, left, reserve = [], [], []
    for start in range(0, probe_size, class_size):
        order = start + rng.permutation(class_size)
        anchors.append(order[:anchor_count])
        size = (class_size - anchor_count) // 2
        left.append(order[anchor_count:anchor_count + size])
        reserve.append(order[anchor_count + size:anchor_count + 2 * size])
    return np.concatenate(anchors), np.stack(left), np.stack(reserve)


def run(train, test, args, seed, device):
    source, target = class_split(seed, True)
    partition_args = copy(args)
    if not args.shared_basis_probe:
        partition_args.probe_examples *= 2
    x, probe, support, query = data_partitions(train, test, partition_args, seed, device, source, target)
    basis_probe = None
    if not args.shared_basis_probe:
        by_class = probe.reshape(len(source), partition_args.probe_examples, *probe.shape[1:])
        probe = by_class[:, :args.probe_examples].flatten(0, 1)
        basis_probe = by_class[:, args.probe_examples:].flatten(0, 1)
    features = learn_domain(x, probe, source, support, query, args, seed, device, basis_probe=basis_probe)
    probe = features["probe"].astype(np.float64)
    rng = np.random.default_rng(seed + 7700)
    anchors, left, reserve = candidate_pools(probe.shape[1], args.anchors // len(source),
                                             args.probe_examples, rng)
    anchor_rows = np.arange(len(anchors))
    candidate_rows = len(anchors) + np.arange(left.size)
    measurements = {}
    for overlap in args.overlaps:
        common = round(overlap * left.shape[1])
        right = np.concatenate((left[:, :common], reserve[:, common:]), axis=1).ravel()
        # Hold the right-view permutation fixed across overlap levels.
        right = right[np.random.default_rng(seed + 7460).permutation(left.size)]
        observed = np.stack((probe[0, np.r_[anchors, left.ravel()]], probe[1, np.r_[anchors, right]]))
        signed = observed.copy()
        signed[1][:, features["odd"]] *= -1
        coordinates = matching_coordinates(signed)
        threshold, balanced = rejection_cost(coordinates, anchor_rows, candidate_rows, args.alignment_ridge)
        scores = pair_scores(align_to_anchors(coordinates, anchor_rows, anchor_rows, args.alignment_ridge))
        variables = observed.reshape(-1, observed.shape[-1])
        mean = variables.mean(axis=0)
        centered = variables - mean
        identity = np.eye(len(mean))
        ridge = 1e-4 * max(np.mean(centered ** 2), 1e-12)
        factor = np.linalg.cholesky(centered.T @ centered / len(centered) + ridge * identity)
        whitener = np.linalg.solve(factor.T, identity)

        def orbit(rows, columns):
            local = (probe[0, rows] - probe[1, columns]) @ whitener / 2
            return covariance(local, np.ones(len(local)))

        common_ids = left[:, :common].ravel()
        true_pairs = np.r_[anchors, common_ids]
        matrices = {"trusted_only": orbit(anchors, anchors), "oracle": orbit(true_pairs, true_pairs)}
        common_energy = np.trace(orbit(common_ids, common_ids))
        correspondence = {}
        for name, (rows, columns) in match(cost_matrix(scores, candidate_rows, candidate_rows), threshold).items():
            first, second = left.ravel()[rows], right[columns]
            recovered = first[first == second]
            correct = len(recovered)
            matrices[name] = orbit(np.r_[anchors, first], np.r_[anchors, second])
            correspondence[name] = {
                "accepted": int(len(first)),
                "precision": float(correct / len(first)) if len(first) else None,
                "recall": float(correct / (common * len(source))),
                "coverage": float(len(first) / left.size),
                "recovered_true_orbit_energy_ratio": float(np.trace(orbit(recovered, recovered)) / common_energy)
                if correct and common_energy > 0 else None,
            }
        reference = matrices["oracle"]
        supports = {k: v - mean for k, v in features["supports"].items()}
        queries = {k: v - mean for k, v in features["queries"].items()}
        priors = {k: budget_prior(v, whitener, args.budget)[0] for k, v in matrices.items()}
        measurements[str(overlap)] = {
            "scores": evaluate(priors, supports, queries, target, args, np.random.default_rng(seed + 422)),
            "actual_overlap": float(common / left.shape[1]),
            "threshold": threshold,
            "anchor_balanced_accuracy": balanced,
            "correspondence": correspondence,
            "relative_orbit_error": {k: float(np.linalg.norm(v - reference)
                                             / max(np.linalg.norm(reference), 1e-12))
                                     for k, v in matrices.items()},
        }
    print(seed, "completed", flush=True)
    return {"seed": seed, "source_classes": source, "target_classes": target, "measurements": measurements}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/tmp/ibt-cifar-data"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(301, 313)))
    parser.add_argument("--overlaps", type=float, nargs="+", default=[.25, .50, .75, 1.])
    parser.add_argument("--anchors", type=int, default=32)
    parser.add_argument("--alignment-ridge", type=float, default=.25)
    parser.add_argument("--budget", type=float, default=.25)
    parser.add_argument("--shots", type=int, nargs="+", default=[5, 20, 60])
    parser.add_argument("--teacher-epochs", type=int, default=26)
    parser.add_argument("--student-epochs", type=int, default=22)
    parser.add_argument("--source-examples", type=int, default=900)
    parser.add_argument("--probe-examples", type=int, default=128)
    parser.add_argument("--shared-basis-probe", action="store_true")
    parser.add_argument("--target-pool", type=int, default=600)
    parser.add_argument("--test-per-class", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--noise", type=float, default=.45)
    args = parser.parse_args()
    if args.output is None:
        result_dir = "partial_pairs" if args.shared_basis_probe else "partial_pairs_independent"
        args.output = HERE / "results" / result_dir / "metrics.json"
    pool_size = (args.probe_examples - args.anchors // 4) // 2
    if args.anchors < 4 or args.anchors % 4 or pool_size < 2:
        parser.error("need a positive anchor count divisible by four and two candidates per class per pool")
    if not all(0 < f <= 1 and round(f * pool_size) > 0 for f in args.overlaps):
        parser.error("each overlap must supply at least one common candidate per class")
    if not 0 < args.budget < 1 or not np.isfinite(args.alignment_ridge) or args.alignment_ridge <= 0:
        parser.error("invalid variance budget or alignment ridge")
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
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
