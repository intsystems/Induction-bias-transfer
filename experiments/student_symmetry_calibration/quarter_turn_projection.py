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
                        evaluate, fit_student, fit_teacher)

RESULTS = HERE / "results" / "quarter_turn_projection"
VIEWS = tuple(partial(torch.rot90, k=rotation, dims=(-2, -1)) for rotation in range(1, 4))
QUARTER_TURNS = np.array([[[1, 0], [0, 1]], [[0, -1], [1, 0]],
                          [[-1, 0], [0, -1]], [[0, 1], [-1, 0]]])


def quarter_turn_from_pairs(orbit):
    left = orbit.reshape(-1, orbit.shape[-1])
    right = np.roll(orbit, -1, axis=0).reshape(left.shape)
    u, _, vt = np.linalg.svd(left.T @ right, full_matrices=False)
    blocks, basis = schur(u @ vt, output="real")
    selected = np.zeros_like(blocks)
    offset = 0
    while offset < len(blocks):
        if offset + 1 < len(blocks) and abs(blocks[offset + 1, offset]) > 1e-10:
            subspace = slice(offset, offset + 2)
            angle = np.arctan2(blocks[offset + 1, offset], blocks[offset, offset])
            quarter = int(np.rint(angle / (np.pi / 2))) % 4
            selected[subspace, subspace] = QUARTER_TURNS[quarter]
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
def invariant_embeddings(student, dataset, fixed, args, device):
    result = {}
    for label, images in dataset.items():
        parts = []
        for batch in images.split(args.batch_size):
            batch = batch.to(device, dtype=torch.float32) / 255
            views = torch.cat((batch, *(view(batch) for view in VIEWS)))
            mean = student(views).reshape(4, len(batch), -1).mean(dim=0)
            parts.append((mean @ fixed).cpu().numpy())
        result[label] = np.stack([np.concatenate(parts)] * 2)
    return result


def run(train, test, args, seed, device):
    source_classes, target_classes = class_split(seed, vary_classes=True)
    source, probe, target, query = data_partitions(train, test, args, seed, device,
                                                   source_classes, target_classes)
    labels = torch.arange(len(source_classes), device=device).repeat_interleave(args.source_examples)
    teacher = fit_teacher(source, labels, args, seed, len(source_classes))
    mean, basis, _ = discover_basis(teacher, probe, args)
    probe_orbit = teacher_orbit(teacher, probe, basis, mean, args.batch_size)
    action = quarter_turn_from_pairs(probe_orbit[:, ::2])
    powers = rotation_powers(action)
    generator = np.random.default_rng(seed + 250)
    rotation, _ = np.linalg.qr(generator.standard_normal(action.shape))
    rotated_powers = rotation_powers(rotation @ action @ rotation.T)
    projector = powers.mean(axis=0)
    eig, directions = np.linalg.eigh((projector + projector.T) / 2)
    fixed = directions[:, eig > .5]
    if not fixed.shape[1]:
        raise RuntimeError("the learned rotation has no invariant coordinates")
    prior = np.eye(fixed.shape[1]) / fixed.shape[1]
    fixed_tensor = source.new_tensor(fixed)

    shots = {str(count): {} for count in args.shots}
    for name, actions in (("inference_only", None), ("learned_training", powers),
                          ("rotated_training", rotated_powers)):
        student = fit_student(source, teacher, basis, mean, args, seed + 100, training_views=VIEWS,
                              orbit_actions=None if actions is None else source.new_tensor(actions))
        support, queries = (invariant_embeddings(student, part, fixed_tensor, args, device)
                            for part in (target, query))
        scores = evaluate({name: prior}, support, queries, target_classes, args,
                          np.random.default_rng(seed + 500))
        for count, row in scores.items():
            shots[count].update(row)
    holdout = probe_orbit[:, 1::2]
    energy = max(float(np.square(holdout).mean()), 1e-12)
    diagnostics = {name: float(np.square(np.roll(holdout, -1, axis=0) - holdout @ generator).mean() / energy)
                   for name, generator in (("learned", action), ("rotated", rotated_powers[1]))}
    return {"seed": seed, "source_classes": source_classes, "target_classes": target_classes,
            "latent_width": int(len(action)), "invariant_width": int(fixed.shape[1]),
            "action_error": diagnostics, "order_four_error": float(np.max(np.abs(
                np.linalg.matrix_power(action, 4) - np.eye(len(action))))), "shots": shots}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/tmp/ibt-cifar-data"))
    parser.add_argument("--output", type=Path, default=RESULTS / "metrics.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1249, 1265)))
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
    args = parser.parse_args()
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
