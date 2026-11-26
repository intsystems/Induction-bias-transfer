"""Infer an unknown reflection permutation of classes from clean few-shot labels."""

import argparse
import json
import math
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import GTSRB

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from reflection_evidence.experiment import (TASKS, discover_basis, embed, fit_student, fit_teacher,
                                             preprocess, prior_kernels)

SWAPS = dict(TASKS["antisymmetric"])
SWAPS.update({second: first for first, second in TASKS["antisymmetric"]})
DIRECTIONAL = tuple(sorted(SWAPS))
STATIONARY = tuple(sorted(label for pair in TASKS["invariant"] for label in pair))
SCENARIOS = {
    "mixed": tuple(sorted((*DIRECTIONAL, 11, 18, 22, 26))),
    "only_fixed": STATIONARY,
    "only_swapped": DIRECTIONAL,
}


def matching_posterior(edge):
    """Return exact permutation marginals and a coherent MAP involution."""
    size = len(edge)
    full = (1 << size) - 1

    @lru_cache(maxsize=None)
    def recurse(mask):
        if not mask:
            return 0.0, 0.0, -1
        first_bit = mask & -mask
        first = first_bit.bit_length() - 1
        rest = mask ^ first_bit
        total, best_score, _ = recurse(rest)
        best_index = first
        partners = rest
        while partners:
            bit = partners & -partners
            partner = bit.bit_length() - 1
            log_z, map_score, _ = recurse(rest ^ bit)
            weight = edge[first, partner]
            total = np.logaddexp(total, log_z + weight)
            if map_score + weight > best_score:
                best_score, best_index = map_score + weight, partner
            partners ^= bit
        return total, best_score, best_index

    log_z = recurse(full)[0]
    posterior = np.zeros_like(edge)
    for i in range(size):
        rest = full ^ (1 << i)
        posterior[i, i] = np.exp(recurse(rest)[0] - log_z)
        for j in range(i):
            joint = np.exp(edge[i, j] + recurse(rest ^ (1 << j))[0] - log_z)
            posterior[i, j] = posterior[j, i] = joint
    mapping = np.arange(size)
    mask = full
    while mask:
        bit = mask & -mask
        first = bit.bit_length() - 1
        partner = recurse(mask)[2]
        mapping[first] = partner
        mapping[partner] = first
        mask &= ~((1 << first) | (1 << partner))
    return posterior, mapping


def gaussian_head(kernel, support, labels, query, noise, solve_precision=False):
    cross = support @ kernel
    covariance = cross @ support.T + noise ** 2 * np.eye(len(support))
    if solve_precision:
        precision = np.linalg.inv(covariance)
        alpha = precision @ labels
    else:
        precision = None
        alpha = np.linalg.solve(covariance, labels)
    return query @ cross.T @ alpha, alpha, np.linalg.slogdet(covariance)[1], precision


def quad_contrasts(quadratic):
    diagonal = np.diag(quadratic)
    return diagonal[:, None] + diagonal[None, :] - quadratic - quadratic.T


def edge_evidence(even_head, odd_head, labels):
    _, alpha_e, determinant_e, _ = even_head
    _, alpha_o, determinant_o, _ = odd_head
    contrasts = quad_contrasts(labels.T @ (alpha_o - alpha_e))
    return -(determinant_o - determinant_e) / 2 - contrasts / 4


def calibrated_evidence(even_head, odd_head, labels, shots):
    _, alpha_e, logdet_e, precision_e = even_head
    _, alpha_o, logdet_o, precision_o = odd_head
    size = len(labels)
    half_log_variance_ratio = (logdet_o - logdet_e) / (2 * size)
    # Match determinants without changing conditional predictions for a parity.
    volume_difference = (np.exp(half_log_variance_ratio) * alpha_o
                         - np.exp(-half_log_variance_ratio) * alpha_e)
    volume = -quad_contrasts(labels.T @ volume_difference) / 4

    # Under uniformly permuted support labels the contrast has zero mean and
    # squared norm 2*shots. Its covariance is a scaled centering projector.
    difference = precision_o - precision_e
    centered_trace = np.trace(difference) - difference.sum() / size
    null_contrast = 2 * shots * centered_trace / (size - 1)
    observed = quad_contrasts(labels.T @ (alpha_o - alpha_e))
    null_centered = -(observed - null_contrast) / 4
    return {"volume_balanced": volume, "null_centered": null_centered}


def prototype_evidence(support, odd, shots, size):
    prototypes = support.reshape(size, shots, -1).mean(axis=1)
    mirrored = prototypes * np.where(odd, -1, 1)
    distance = ((prototypes[:, None] - mirrored[None, :]) ** 2).mean(axis=-1)
    self_distance = np.diag(distance)
    change = (self_distance[:, None] + self_distance[None, :]) / 2 - distance
    scale = np.median(np.abs(change[np.triu_indices(size, 1)]))
    return change / max(scale, 1e-8)


def crossfit_edges(support, transformed, shots, count):
    clean = support.reshape(count, shots, -1)
    reflected = transformed.reshape(count, shots, -1)
    width = clean.shape[-1]
    score = np.zeros((count, count))
    for index in range(shots):
        retained = clean[:, np.arange(shots) != index]
        center = retained.mean(axis=1)
        within = ((retained - center[:, None]) ** 2).mean(axis=(0, 1))
        pooled = retained.reshape(-1, width).var(axis=0)
        variance = 0.6 * within + 0.4 * pooled + 0.05
        residual = reflected[:, index, None, :] - center[None, :, :]
        score -= ((residual ** 2) / variance).sum(axis=2) / (2 * np.sqrt(width))
    improvements = score - np.diag(score)[:, None]
    return improvements + improvements.T


def combine(even, odd, matrix):
    return (even + odd + (even - odd) @ matrix) / 2


def score_predictions(prediction, labels, mapping, num_classes):
    clean, mirror = prediction.argmax(axis=1).reshape(2, -1)
    scores = {}
    for name, correct in (
        ("clean_accuracy", clean == labels), ("mirror_accuracy", mirror == mapping[labels]),
        ("parity_agreement", mirror == mapping[clean]),
    ):
        by_class = np.array([correct[labels == i].mean() for i in range(num_classes)])
        scores[name] = float(by_class.mean())
        scores[name + "_per_class"] = by_class.tolist()
    return scores


def episode(support, support_mirror, labels, query, truth, mapping, kernels, odd, count, args):
    iso, even, odd_kernel, random_even, random_odd = kernels
    size = len(mapping)
    identity = np.eye(size)
    iso_prediction = gaussian_head(iso, support, labels, query, args.noise)[0]
    even_head = gaussian_head(even, support, labels, query, args.noise, args.calibrate_evidence)
    odd_head = gaussian_head(odd_kernel, support, labels, query, args.noise, args.calibrate_evidence)
    random_e = gaussian_head(random_even, support, labels, query, args.noise)
    random_o = gaussian_head(random_odd, support, labels, query, args.noise)
    edge_cost = math.log(args.match_strength / (size - 1))
    evidence = edge_evidence(even_head, odd_head, labels) + edge_cost
    real_posterior, map_action = matching_posterior(evidence)
    random_posterior, random_map = matching_posterior(edge_evidence(random_e, random_o, labels) + edge_cost)
    proto, proto_map = matching_posterior(prototype_evidence(support, odd, count, size) + edge_cost)
    local = evidence.copy()
    np.fill_diagonal(local, 0)
    local = np.exp(local - local.max(axis=1, keepdims=True))
    local /= local.sum(axis=1, keepdims=True)

    cases = {
        "isotropic": (iso_prediction, None, None),
        "all_fixed": (even_head[0], identity, np.arange(size)),
        "rowwise": (combine(even_head[0], odd_head[0], local), local, local.argmax(axis=1)),
        "prototype_match": (combine(even_head[0], odd_head[0], proto), proto, proto_map),
        "random_basis": (combine(random_e[0], random_o[0], random_posterior),
                         random_posterior, random_map),
        "posterior": (combine(even_head[0], odd_head[0], real_posterior), real_posterior, map_action),
        "oracle": (combine(even_head[0], odd_head[0], identity[mapping]), identity[mapping], mapping),
    }
    if args.calibrate_evidence:
        alternatives = calibrated_evidence(even_head, odd_head, labels, count)
        if count > 1:
            virtual = support * np.where(odd, -1, 1)
            alternatives["crossfit_virtual"] = crossfit_edges(support, virtual, count, size)
            alternatives["crossfit_observed"] = crossfit_edges(support, support_mirror, count, size)
        for name, scores_by_edge in alternatives.items():
            probabilities, decision = matching_posterior(scores_by_edge + edge_cost)
            cases[name] = (combine(even_head[0], odd_head[0], probabilities), probabilities, decision)
    scores = {}
    swapped = mapping != np.arange(size)
    for name, (prediction, probabilities, decision) in cases.items():
        row = score_predictions(prediction, truth, mapping, size)
        mirror = np.asarray(row["mirror_accuracy_per_class"])
        if swapped.any():
            row["swapped_mirror_accuracy"] = float(mirror[swapped].mean())
        if (~swapped).any():
            row["fixed_mirror_accuracy"] = float(mirror[~swapped].mean())
        if decision is not None:
            row["class_mapping_accuracy"] = float((decision == mapping).mean())
            row["true_mapping_probability"] = float(probabilities[np.arange(size), mapping].mean())
            row["exact_permutation"] = float(np.array_equal(decision, mapping))
            row["number_moved_classes"] = float((decision != np.arange(size)).sum())
        if name == "posterior":
            row["raw_log_volume_bonus"] = float((even_head[2] - odd_head[2]) / 2)
        scores[name] = row
    return scores


def average(rows):
    return {
        method: {key: np.asarray([row[method][key] for row in rows]).mean(axis=0).tolist()
                 for key in values}
        for method, values in rows[0].items()
    }


def run_scenario(classes, train, test, tracks, kernels, odd, args, rng):
    size = len(classes)
    mapping = np.array([classes.index(SWAPS.get(label, label)) for label in classes])
    query = np.concatenate([np.concatenate([test[label][view] for label in classes])
                            for view in range(2)])
    truth = np.concatenate([np.full(len(test[label][0]), i) for i, label in enumerate(classes)])
    output = {str(n): [] for n in args.shots}
    for _ in range(args.episodes):
        selected = []
        for label in classes:
            groups = tracks[label]
            indices = rng.choice(len(groups), max(args.shots), replace=False)
            selected.append([rng.choice(groups[i]) for i in indices])
        for n in args.shots:
            support = np.concatenate([train[label][0, ids[:n]] for label, ids in zip(classes, selected)])
            support_mirror = np.concatenate([train[label][1, ids[:n]]
                                             for label, ids in zip(classes, selected)])
            labels = np.repeat(np.eye(size), n, axis=0) - 1 / size
            output[str(n)].append(episode(support, support_mirror, labels, query, truth,
                                          mapping, kernels, odd, n, args))
    return {budget: average(rows) for budget, rows in output.items()}


def run(data, args, device, seed):
    rng = np.random.default_rng(seed)
    (source, labels), (probe, probe_labels), train, test, tracks = data
    source = source.to(device, dtype=torch.float32) / 255
    probe = probe.to(device, dtype=torch.float32) / 255
    teacher = fit_teacher(source, labels.to(device), args, seed)
    mean, basis, energies = discover_basis(teacher, probe, args)
    student = fit_student(source, teacher, basis, mean, args, seed + 100)
    with torch.no_grad():
        accuracy = torch.cat([teacher(batch).argmax(dim=1).cpu()
                              for batch in probe.split(args.batch_size)]).eq(probe_labels).float().mean().item()
    train = embed(student, train, args, device)
    test = embed(student, test, args, device)
    energies = energies.cpu().numpy()
    bank = prior_kernels(energies, rng)
    kernels = (bank["isotropic"][0], bank["fixed_even"][0], bank["fixed_odd"][0],
               bank["random_adaptive"][0], bank["random_adaptive"][-1])
    result = {"seed": seed, "teacher_probe_accuracy": accuracy, "latent_width": len(energies),
              "scenarios": {name: run_scenario(classes, train, test, tracks, kernels, energies > 0.5, args, rng)
                            for name, classes in SCENARIOS.items()}}
    focal = result["scenarios"]["mixed"][str(max(args.shots))]["posterior"]
    print(seed, "teacher", round(accuracy, 4), "mirror", round(focal["mirror_accuracy"], 4),
          "mapping", round(focal["class_mapping_accuracy"], 4), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/tmp/ibt-gtsrb-data"))
    parser.add_argument("--output", type=Path, default=HERE / "results/metrics.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4, 5])
    parser.add_argument("--teacher-epochs", type=int, default=40)
    parser.add_argument("--student-epochs", type=int, default=30)
    parser.add_argument("--source-examples", type=int, default=600)
    parser.add_argument("--probe-examples", type=int, default=65)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--noise", type=float, default=0.35)
    parser.add_argument("--match-strength", type=float, default=1.0)
    parser.add_argument("--calibrate-evidence", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    data = preprocess(GTSRB(args.data, split="train", download=True),
                      GTSRB(args.data, split="test", download=True), args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
               "torch_version": torch.__version__, "scenarios": SCENARIOS, "runs": []}
    start = time.monotonic()
    for seed in args.seeds:
        metrics["runs"].append(run(data, args, device, seed))
        metrics["elapsed_seconds"] = time.monotonic() - start
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(metrics, indent=2) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
