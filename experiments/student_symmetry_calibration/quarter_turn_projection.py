"""Compare a learned quarter-turn projection in distillation and inference."""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
from scipy.linalg import schur
import torch

from experiment import (CIFAR10, HERE, class_split, data_partitions, discover_basis,
                        evaluate, fit_student, fit_teacher, predict)

RESULTS = HERE / "results" / "quarter_turn_projection"
VIEWS = tuple(partial(torch.rot90, k=rotation, dims=(-2, -1)) for rotation in range(1, 4))
QUARTER_TURNS = np.array([[[1, 0], [0, 1]], [[0, -1], [1, 0]],
                          [[-1, 0], [0, -1]], [[0, 1], [-1, 0]]])
VIEW_BUDGETS = {"one": (0,), "opposite": (0, 2), "adjacent": (0, 1), "four": (0, 1, 2, 3)}


def rotation_from_pairs(orbit):
    left = orbit.reshape(-1, orbit.shape[-1])
    right = np.roll(orbit, -1, axis=0).reshape(left.shape)
    u, _, vt = np.linalg.svd(left.T @ right, full_matrices=False)
    return u @ vt


def round_quarter_turn(action, fraction=0.):
    blocks, basis = schur(action, output="real")
    selected = np.zeros_like(blocks)
    offset = 0
    while offset < len(blocks):
        if offset + 1 < len(blocks) and abs(blocks[offset + 1, offset]) > 1e-10:
            subspace = slice(offset, offset + 2)
            angle = np.arctan2(blocks[offset + 1, offset], blocks[offset, offset])
            quarter = int(np.rint(angle / (np.pi / 2)))
            if fraction:
                angle = (1 - fraction) * quarter * (np.pi / 2) + fraction * angle
                cosine, sine = np.cos(angle), np.sin(angle)
                selected[subspace, subspace] = ((cosine, -sine), (sine, cosine))
            else:
                selected[subspace, subspace] = QUARTER_TURNS[quarter % 4]
            offset += 2
        else:
            selected[offset, offset] = 1 if blocks[offset, offset] >= 0 else -1
            offset += 1
    return basis @ selected @ basis.T


def rotation_powers(action):
    return np.stack([np.linalg.matrix_power(action, turns) for turns in range(4)])


@torch.inference_mode()
def teacher_orbit(teacher, images, basis, mean, batch_size):
    parts = []
    for batch in images.split(batch_size):
        views = torch.cat((batch, *(view(batch) for view in VIEWS)))
        parts.append(((teacher.encode(views) - mean) @ basis).reshape(4, len(batch), -1).cpu().numpy())
    return np.concatenate(parts, axis=1).astype(np.float64)


@torch.inference_mode()
def invariant_embeddings(student, dataset, fixed, args, device, *, return_orbit=False):
    result = {}
    for label, images in dataset.items():
        parts = []
        for batch in images.split(args.batch_size):
            batch = batch.to(device, dtype=torch.float32) / 255
            views = torch.cat((batch, *(view(batch) for view in VIEWS)))
            orbit = student(views).reshape(4, len(batch), -1)
            embedding = orbit if return_orbit else orbit.mean(dim=0)
            parts.append((embedding @ fixed).cpu().numpy())
        result[label] = np.concatenate(parts, axis=1) if return_orbit else np.stack([np.concatenate(parts)] * 2)
    return result


def pool_rotation_views(orbit, offsets):
    if len(offsets) == len(orbit):
        return np.broadcast_to(orbit.mean(axis=0), orbit.shape)
    return np.mean([np.roll(orbit, -offset, axis=0) for offset in offsets], axis=0)


def evaluate_view_budgets(prior, support, queries, target_classes, args, seed):
    query_orbit = np.concatenate([queries[label] for label in target_classes], axis=1).astype(np.float64)
    truth = np.repeat(np.arange(len(target_classes)), [queries[label].shape[1] for label in target_classes])
    fixed_queries = {name: pool_rotation_views(query_orbit, offsets) for name, offsets in VIEW_BUDGETS.items()}
    fixed_support = {name: {label: pool_rotation_views(orbit, offsets)[0] for label, orbit in support.items()}
                     for name, offsets in VIEW_BUDGETS.items()}
    rng = np.random.default_rng(seed)
    observations = {str(count): {name: [] for name in VIEW_BUDGETS} for count in args.shots}
    for _ in range(args.episodes):
        chosen = [rng.choice(support[label].shape[1], max(args.shots), replace=False) for label in target_classes]
        for count in args.shots:
            labels = np.repeat(np.eye(len(target_classes)), count, axis=0) - 1 / len(target_classes)
            for name, pooled_support in fixed_support.items():
                train = np.concatenate([pooled_support[label][indices[:count]]
                                        for label, indices in zip(target_classes, chosen, strict=True)]).astype(np.float64)
                correct = predict(prior, train, labels, fixed_queries[name], args.noise).argmax(axis=-1) == truth
                observations[str(count)][name].append((correct.mean(), correct.all(axis=0).mean()))
    return {count: {name: dict(zip(("mean_accuracy", "all_rotations_accuracy"),
                                  np.mean(rows, axis=0).tolist(), strict=True))
                    for name, rows in measurements.items()} for count, measurements in observations.items()}


def run(train, test, args, seed, device):
    source_classes, target_classes = class_split(seed, vary_classes=True)
    source, probe, target, query = data_partitions(train, test, args, seed, device,
                                                   source_classes, target_classes)
    labels = torch.arange(len(source_classes), device=device).repeat_interleave(args.source_examples)
    teacher = fit_teacher(source, labels, args, seed, len(source_classes))
    mean, basis, _ = discover_basis(teacher, probe, args)
    probe_orbit = teacher_orbit(teacher, probe, basis, mean, args.batch_size)
    fitted = rotation_from_pairs(probe_orbit[:, ::2])
    action = round_quarter_turn(fitted)
    powers = rotation_powers(action)
    candidates = {"learned": action}
    if args.approximate_action:
        candidates.update(midpoint=round_quarter_turn(fitted, .5), unrounded=fitted)
    elif not args.view_budget:
        generator = np.random.default_rng(seed + 250)
        rotation, _ = np.linalg.qr(generator.standard_normal(action.shape))
        candidates["rotated"] = rotation @ action @ rotation.T
    projector = powers.mean(axis=0)
    eig, directions = np.linalg.eigh((projector + projector.T) / 2)
    fixed = directions[:, eig > .5]
    if not fixed.shape[1]:
        raise RuntimeError("the learned rotation has no invariant coordinates")
    prior = np.eye(fixed.shape[1]) / fixed.shape[1]
    fixed_tensor = source.new_tensor(fixed)

    shots = {str(count): {} for count in args.shots}
    variants = [("inference_only", None)]
    variants.extend((f"{name}_training", rotation_powers(generator)) for name, generator in candidates.items())
    for name, actions in variants:
        student = fit_student(source, teacher, basis, mean, args, seed + 100, training_views=VIEWS,
                              orbit_actions=None if actions is None else source.new_tensor(actions))
        support, queries = (invariant_embeddings(student, part, fixed_tensor, args, device,
                                                 return_orbit=args.view_budget)
                            for part in (target, query))
        if args.view_budget:
            scores = evaluate_view_budgets(prior, support, queries, target_classes, args, seed + 500)
            for count, row in scores.items():
                shots[count][name] = row
        else:
            scores = evaluate({name: prior}, support, queries, target_classes, args,
                              np.random.default_rng(seed + 500))
            for count, row in scores.items():
                shots[count].update(row)
    holdout = probe_orbit[:, 1::2]
    energy = max(float(np.square(holdout).mean()), 1e-12)
    diagnostics = {name: float(np.square(np.roll(holdout, -1, axis=0) - holdout @ generator).mean() / energy)
                   for name, generator in candidates.items()}
    identity = np.eye(len(action))
    order_errors = {name: float(np.max(np.abs(np.linalg.matrix_power(generator, 4) - identity)))
                    for name, generator in candidates.items()}
    result = {"seed": seed, "source_classes": source_classes, "target_classes": target_classes,
              "latent_width": int(len(action)), "invariant_width": int(fixed.shape[1]),
              "action_error": diagnostics, "order_four_error": order_errors["learned"], "shots": shots}
    if args.approximate_action:
        result["approximate_order_four_errors"] = {name: order_errors[name] for name in ("midpoint", "unrounded")}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/tmp/ibt-cifar-data"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--shots", type=int, nargs="+", default=[5, 20, 60])
    parser.add_argument("--teacher-epochs", type=int, default=26)
    parser.add_argument("--student-epochs", type=int, default=22)
    parser.add_argument("--source-examples", type=int, default=1400)
    parser.add_argument("--probe-examples", type=int, default=320)
    parser.add_argument("--target-pool", type=int, default=600)
    parser.add_argument("--test-per-class", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--noise", type=float, default=.45)
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument("--approximate-action", action="store_true")
    comparison.add_argument("--view-budget", action="store_true")
    args = parser.parse_args()
    if args.output is None:
        folder = RESULTS.with_name("quarter_turn_approximate") if args.approximate_action else RESULTS
        if args.view_budget:
            folder = RESULTS.with_name("quarter_turn_view_budget")
        args.output = folder / "metrics.json"
    if args.seeds is None:
        first_seed = 1281 if args.view_budget else (1265 if args.approximate_action else 1249)
        args.seeds = list(range(first_seed, first_seed + 16))
    if not 0 < min(args.shots) <= max(args.shots) <= args.target_pool <= 5000:
        parser.error("shots must fit a nonempty target pool of at most 5000 examples per class")
    if args.probe_examples < 2 or not 0 < args.source_examples <= 5000 - args.probe_examples:
        parser.error("need distinct source and probe samples with at least two probe examples per class")
    if not 0 < args.test_per_class <= 1000 or min(args.batch_size, args.teacher_epochs, args.student_epochs,
                                                   args.episodes) < 1 or not 0 < args.noise < float("inf"):
        parser.error("invalid test count, batch size, training steps, target episodes or head noise")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    train, test = (CIFAR10(args.data, train=is_train, download=True) for is_train in (True, False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = {"config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "torch_version": torch.__version__, "runs": []}
    start = time.monotonic()
    for seed in args.seeds:
        data["runs"].append(run(train, test, args, seed, device))
        data["elapsed_seconds"] = time.monotonic() - start
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)
        print(seed, "completed", flush=True)


if __name__ == "__main__":
    main()
