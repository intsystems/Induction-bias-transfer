"""Transfer joint generator constraints to held-out image classes."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiment import (CIFAR10, HERE, budget_weights, class_split, data_partitions, discover_basis,
                        fit_student, fit_teacher, predict)

ROTATION = (2, 7, 4, 1, 6, 3, 0, 5)
REFLECTION = (1, 0, 3, 2, 5, 4, 7, 6)
COMPOSITION = tuple(REFLECTION[i] for i in ROTATION)
TEST_VIEWS = {"reflection": 1, "rotation": 2, "composition": 3, "half_turn": 4}


@torch.inference_mode()
def embed_orbit(student, images, device, batch_size):
    views = [[] for _ in range(8)]
    for batch in images.split(batch_size):
        batch = batch.to(device=device, dtype=torch.float32)
        if images.dtype == torch.uint8:
            batch = batch / 255
        for k in range(4):
            rotated = torch.rot90(batch, k, (-2, -1))
            views[2 * k].append(student(rotated).cpu().numpy())
            views[2 * k + 1].append(student(rotated.flip(-1)).cpu().numpy())
    return np.stack([np.concatenate(view) for view in views]).astype(np.float64)


def generator_moments(orbit, complete, actions=(ROTATION, REFLECTION)):
    outputs = []
    start = np.arange(8) if complete else np.array([0])
    for action in actions:
        error = ((orbit[start] - orbit[np.asarray(action)[start]]) / 2).reshape(-1, orbit.shape[-1])
        moment = error.T @ error / len(error)
        outputs.append(moment / max(np.trace(moment) / len(moment), 1e-12))
    return np.stack(outputs)


def budget_covariance(moment, budget):
    energies, axes = np.linalg.eigh(moment)
    weights, _, _ = budget_weights(np.maximum(energies, 0), budget)
    return (axes * weights) @ axes.T


def generator_balanced(moments, requested):
    """Solve the two trace constraints by searching the mixing weight of their dual."""
    def solve(budget):
        def candidate(weight):
            covariance = budget_covariance(weight * moments[0] + (1 - weight) * moments[1], budget)
            return covariance, np.einsum("gij,ij->g", moments, covariance)

        left, right = 0., 1.
        cov, energy = candidate(left)
        if energy[0] <= energy[1]:
            return cov, energy, left
        cov, energy = candidate(right)
        if energy[1] <= energy[0]:
            return cov, energy, right
        for _ in range(28):
            mid = (left + right) / 2
            cov, energy = candidate(mid)
            if energy[0] > energy[1]:
                left = mid
            else:
                right = mid
        return cov, energy, mid

    covariance, achieved, weight = solve(requested)
    # If the two budgets have no common interior, relax only on source data.
    budget = requested
    if achieved.max() > requested + 1e-5:
        low, high = requested, 1.
        for _ in range(22):
            mid = (low + high) / 2
            _, energy, _ = solve(mid)
            if energy.max() <= mid + 1e-5:
                high = mid
            else:
                low = mid
        budget = high
        covariance, achieved, weight = solve(budget)
    return covariance, achieved.tolist(), float(weight), float(budget)


def source_priors(probe, budget, seed, generator_controls=False):
    width = probe.shape[-1]
    flat = probe.reshape(-1, width)
    ridge = 1e-4 * max(float(np.mean(flat ** 2)), 1e-12)
    metric = flat.T @ flat / len(flat) + ridge * np.eye(width)
    whitener = np.linalg.solve(np.linalg.cholesky(metric).T, np.eye(width))
    orbit = probe @ whitener
    natural = {"isotropic": np.eye(width) / width}
    diagnostics = {}
    rng = np.random.default_rng(seed + 407)
    for key, complete in (("canonical", False), ("orbit", True)):
        moments = generator_moments(orbit, complete)
        joint, achieved, mixture, feasible = generator_balanced(moments, budget)
        average = budget_covariance(moments.mean(axis=0), budget)
        natural[f"{key}_average"] = average
        natural[f"{key}_balanced"] = joint
        diagnostics[key] = {"balanced": achieved, "rotation_weight": mixture, "budget": feasible,
                            "average": np.einsum("gij,ij->g", moments, average).tolist()}
        if complete:
            for i in range(4):
                rotation, _ = np.linalg.qr(rng.normal(size=(width, width)))
                natural[f"orbit_rotated_{i}"] = rotation @ joint @ rotation.T
            if generator_controls:
                composition, = generator_moments(orbit, complete, (COMPOSITION,))
                action_moments = np.concatenate([moments, composition[None]], axis=0)
                for key, moment in zip(("rotation", "reflection", "composition"), action_moments, strict=True):
                    control = budget_covariance(moment, budget)
                    natural[f"orbit_{key}_only"] = control
                    diagnostics["orbit"][f"{key}_only"] = np.einsum("gij,ij->g", action_moments, control).tolist()
                diagnostics["orbit"]["balanced_composition"] = float(np.sum(joint * composition))
                diagnostics["orbit"]["average_composition"] = float(np.sum(average * composition))
    return {k: whitener @ v @ whitener.T for k, v in natural.items()}, diagnostics


def run(train, test, args, seed, device):
    source, target = class_split(seed, True)
    x, probe, support, test_views = data_partitions(train, test, args, seed, device, source, target)
    training_labels = torch.arange(len(source), device=device).repeat_interleave(args.source_examples)
    teacher = fit_teacher(x, training_labels, args, seed, classes=len(source))
    mean, basis, _ = discover_basis(teacher, probe[::2], args)
    student = fit_student(x, teacher, basis, mean, args, seed + 100)
    source_views = embed_orbit(student, probe[1::2], device, args.batch_size)
    center = source_views.mean(axis=(0, 1))
    priors, diagnostics = source_priors(source_views - center, args.budget, seed, args.generator_controls)
    supports = {k: embed_orbit(student, v, device, args.batch_size)[0] - center for k, v in support.items()}
    queries = np.concatenate([embed_orbit(student, test_views[k], device, args.batch_size)
                              for k in target], axis=1) - center
    truth = np.repeat(np.arange(len(target)), args.test_per_class)
    rows = {str(n): [] for n in args.shots}
    rng = np.random.default_rng(seed + 422)
    for _ in range(args.episodes):
        chosen = {k: rng.choice(len(supports[k]), max(args.shots), replace=False) for k in target}
        for n in args.shots:
            inputs = np.concatenate([supports[k][chosen[k][:n]] for k in target])
            labels = np.repeat(np.eye(len(target)), n, axis=0) - 1 / len(target)
            methods = {}
            for key, prior in priors.items():
                correct = predict(prior, inputs, labels, queries, args.noise).argmax(axis=-1) == truth
                methods[key] = {"clean": float(correct[0].mean()), "all_views": float(correct.all(axis=0).mean()),
                                **{view: float((correct[0] & correct[index]).mean())
                                   for view, index in TEST_VIEWS.items()}}
            rows[str(n)].append(methods)
    print(seed, "completed", flush=True)
    return {"seed": seed, "diagnostics": diagnostics, "shots": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/tmp/ibt-cifar-data"))
    parser.add_argument("--output", type=Path, default=HERE / "results/generator_budget/metrics.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(701, 717)))
    parser.add_argument("--shots", type=int, nargs="+", default=[5, 20, 60])
    parser.add_argument("--source-examples", type=int, default=900)
    parser.add_argument("--probe-examples", type=int, default=128)
    parser.add_argument("--teacher-epochs", type=int, default=26)
    parser.add_argument("--student-epochs", type=int, default=22)
    parser.add_argument("--target-pool", type=int, default=600)
    parser.add_argument("--test-per-class", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--budget", type=float, default=.25)
    parser.add_argument("--generator-controls", action="store_true",
                        help="include full-orbit priors estimated from each action alone")
    parser.add_argument("--noise", type=float, default=.45)
    args = parser.parse_args()
    if not 0 < args.budget < 1 or args.noise <= 0:
        parser.error("need a variance budget in (0, 1) and positive noise")
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
