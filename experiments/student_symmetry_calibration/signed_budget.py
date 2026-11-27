"""Infer a label action and orbit budget from untransformed support."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiment import GTSRB, HERE, budget_weights, discover_basis, embed, fit_student, fit_teacher
from partial_pairs import project_teacher_action
from reflection_evidence.experiment import TASKS, preprocess, prior_kernels

BUDGETS = (.75, .5, .25, .1)
SIGNS = (1, -1)
FIXED_BUDGET = BUDGETS.index(.25)


def signed_banks(probe, odd):
    mean = probe.mean(axis=(0, 1))
    h = probe - mean
    stacked = h.reshape(-1, h.shape[-1])
    width = len(mean)
    identity = np.eye(width)
    ridge = 1e-4 * max(np.mean(stacked ** 2), 1e-12)
    factor = np.linalg.cholesky(stacked.T @ stacked / len(stacked) + ridge * identity)
    whitener = np.linalg.solve(factor.T, identity)
    native = {"paired": [], "proxy": []}
    achieved = {key: [] for key in native}
    for sign in SIGNS:
        error = (h[0] - sign * h[1]) @ whitener / 2
        disallowed = odd if sign == 1 else ~odd
        surrogate = project_teacher_action(h, identity[:, disallowed], whitener, True).reshape(-1, width)
        for key, residual in (("paired", error), ("proxy", surrogate)):
            energies, vectors = np.linalg.eigh(residual.T @ residual / len(residual))
            axes = whitener @ vectors
            solutions = [budget_weights(np.maximum(energies, 0), budget) for budget in BUDGETS]
            native[key].append([(axes * diagonal) @ axes.T for diagonal, _, _ in solutions])
            achieved[key].append([ratio for _, _, ratio in solutions])
    return mean, {key: np.asarray(values) for key, values in native.items()}, achieved


def support_fit(kernels, x, labels, noise):
    """Calculate log evidence and its exact balanced-permutation centering."""
    products = x @ kernels
    count = len(labels)
    covariance = products @ x.T + noise ** 2 * np.eye(count)
    inverse = np.linalg.solve(covariance, np.eye(count))
    alpha = inverse @ labels
    quad = alpha @ labels
    baseline = (np.trace(inverse, axis1=-2, axis2=-1) - inverse.sum(axis=(-2, -1)) / count)
    baseline *= count / (count - 1)
    scores = {"marginal": -.5 * (np.linalg.slogdet(covariance)[1] + quad),
              "centered": -.5 * (quad - baseline)}
    return products.swapaxes(-2, -1) @ alpha[..., None], scores


def probabilities(scores):
    weights = np.exp(scores - scores.max())
    return weights / weights.sum()


def mixture(coefficients, scores):
    weights = probabilities(scores)
    return np.sum(coefficients * weights[..., None], axis=(0, 1)), weights


def judge(test, truth, true_sign, coefficients):
    clean, reflected = np.where(test @ coefficients >= 0, 1, -1)
    good_clean, good_reflected = clean == truth, reflected == true_sign * truth
    return {key: float((result[truth < 0].mean() + result[truth > 0].mean()) / 2)
            for key, result in (("clean", good_clean), ("reflected", good_reflected),
                                ("both", good_clean & good_reflected),
                                ("parity", reflected == true_sign * clean))}


def episode(x, test, labels, truth, sign, banks, parity, noise):
    estimates, diagnostics = {}, {}
    for name, kernels in banks.items():
        fitted, scores = support_fit(kernels, x, labels, noise)
        fitted = fitted[..., 0]
        for kind, logits in scores.items():
            coefficients, weights = mixture(fitted, logits)
            key = f"{name}_{kind}"
            estimates[key] = coefficients
            diagnostics[key] = {"same_probability": float(weights[0].sum()),
                                "mean_budget": float(weights.sum(axis=0) @ BUDGETS)}
            if name == "paired":
                estimates[f"paired_{kind}_map"] = fitted[np.unravel_index(logits.argmax(), logits.shape)]
        if name == "paired":
            centered = scores["centered"]
            estimates["paired_fixed"] = probabilities(centered[:, FIXED_BUDGET]) @ fitted[:, FIXED_BUDGET]
            estimates["paired_always_same"] = fitted[0, FIXED_BUDGET]
            k = SIGNS.index(sign)
            estimates["paired_known_sign"] = probabilities(centered[k]) @ fitted[k]
    fitted, scores = support_fit(parity, x, labels, noise)
    weights = probabilities(scores["centered"])
    estimates["teacher_parity"] = weights @ fitted[..., 0]
    diagnostics["teacher_parity"] = {"same_probability": float(weights[0])}
    return {"metrics": {key: judge(test, truth, sign, weights) for key, weights in estimates.items()},
            "diagnostics": diagnostics}


def measure_task(pair, sign, train, test, tracks, banks, parity, args, rng):
    query = np.concatenate([test[label] for label in pair], axis=1)
    truth = np.concatenate([np.full(len(test[label][0]), k) for label, k in zip(pair, (-1, 1), strict=True)])
    rows = {str(n): [] for n in args.shots}
    for _ in range(args.episodes):
        selected = []
        for label in pair:
            groups = tracks[label]
            indices = rng.choice(len(groups), max(args.shots), replace=False)
            selected.append([rng.choice(groups[k]) for k in indices])
        for shots in args.shots:
            x = np.concatenate([train[label][0, ids[:shots]] for label, ids in zip(pair, selected, strict=True)])
            labels = np.repeat([-1., 1.], shots)
            rows[str(shots)].append(episode(x, query, labels, truth, sign, banks, parity, args.noise))
    return rows


def run(data, args, device, seed):
    (source, labels), (probe, _), target, target_test, tracks = data
    source = source.to(device, dtype=torch.float32) / 255
    probe = probe.to(device, dtype=torch.float32) / 255
    teacher = fit_teacher(source, labels.to(device), args, seed)
    center, basis, energies = discover_basis(teacher, probe[::2], args)
    student = fit_student(source, teacher, basis, center, args, seed + 100)
    with torch.inference_mode():
        h = np.stack([torch.cat([student(view) for view in samples.split(args.batch_size)]).cpu().numpy()
                      for samples in (probe[1::2], probe[1::2].flip(-1))]).astype(np.float64)
    mean, banks, achieved = signed_banks(h, energies.cpu().numpy() > .5)
    train = {k: a - mean for k, a in embed(student, target, args, device).items()}
    test = {k: a - mean for k, a in embed(student, target_test, args, device).items()}
    rng = np.random.default_rng(seed)
    kernels = prior_kernels(energies.cpu().numpy(), rng)
    parity = np.concatenate((kernels["fixed_even"], kernels["fixed_odd"]))
    tasks = {family: {"-".join(map(str, pair)): measure_task(
        pair, 1 if family == "invariant" else -1, train, test, tracks, banks, parity, args, rng)
        for pair in pairs} for family, pairs in TASKS.items()}
    print(seed, "completed", flush=True)
    return {"seed": seed, "achieved_budgets": achieved, "tasks": tasks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/tmp/ibt-gtsrb-data"))
    parser.add_argument("--output", type=Path, default=HERE / "results/signed_budget/metrics.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(151, 167)))
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4, 5])
    parser.add_argument("--teacher-epochs", type=int, default=20)
    parser.add_argument("--student-epochs", type=int, default=18)
    parser.add_argument("--source-examples", type=int, default=300)
    parser.add_argument("--probe-examples", type=int, default=65)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--noise", type=float, default=.5)
    args = parser.parse_args()
    if not np.isfinite(args.noise) or args.noise <= 0 or min(args.shots) <= 0:
        parser.error("shots and noise must be positive")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    data = preprocess(GTSRB(args.data, split="train", download=True),
                      GTSRB(args.data, split="test", download=True), args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "budgets": BUDGETS, "sign_order": SIGNS, "tasks": TASKS, "runs": []}
    start = time.monotonic()
    for seed in args.seeds:
        output["runs"].append(run(data, args, device, seed))
        output["elapsed_seconds"] = time.monotonic() - start
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
