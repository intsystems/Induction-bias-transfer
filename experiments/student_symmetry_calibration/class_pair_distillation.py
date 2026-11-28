"""Test when to transfer geometry estimated from natural CIFAR-10 class pairs."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiment import (CIFAR10, HERE, class_split, data_partitions, discover_basis,
                        embed, evaluate, fit_student, fit_teacher)

RESULTS = HERE / "results" / "class_pair_distillation"


@torch.no_grad()
def class_pair_metrics(teacher, probe, basis, odd, args, seed):
    latent = torch.cat([teacher.encode(batch) @ basis for batch in probe.split(args.batch_size)])
    groups = latent.cpu().numpy().reshape(-1, args.probe_examples, latent.shape[-1])
    rng = np.random.default_rng(seed)
    pairs = [group[rng.permutation(len(group))] for group in groups]
    delta = np.concatenate([group - np.roll(group, 1, axis=0) for group in pairs]).astype(np.float64)
    width = delta.shape[1]
    natural, rotated = np.zeros((2, width, width))
    for mask in (~odd, odd):
        block = delta[:, mask]
        energies, axes = np.linalg.eigh(block.T @ block / len(block))
        energies = np.maximum(energies, 0)
        weights = 1 / (.25 + energies / max(float(energies.mean()), 1e-12))
        weights /= weights.sum()
        shuffled, _ = np.linalg.qr(rng.standard_normal(axes.shape))
        index = np.ix_(mask, mask)
        natural[index] = (axes * weights) @ axes.T
        rotated[index] = (shuffled * weights) @ shuffled.T
    return {key: probe.new_tensor(matrix / np.diag(matrix).mean())
            for key, matrix in (("natural", natural), ("rotated", rotated))}


def run(train, test, args, seed, device):
    source_classes, target_classes = class_split(seed, vary_classes=True)
    source, probe, target, query = data_partitions(train, test, args, seed, device,
                                                   source_classes, target_classes)
    labels = torch.arange(len(source_classes), device=device).repeat_interleave(args.source_examples)
    teacher = fit_teacher(source, labels, args, seed, len(source_classes))
    mean, basis, energies = discover_basis(teacher, probe, args)
    odd = energies.cpu().numpy() > .5
    metrics = class_pair_metrics(teacher, probe, basis, odd, args, seed + 250)
    half = args.student_epochs // 2
    schedules = {"early": range(half), "late": range(args.student_epochs - half, args.student_epochs)}
    variants = [("uniform", None, None)]
    variants.extend((f"{phase}_{key}", metric, epochs)
                    for phase, epochs in schedules.items() for key, metric in metrics.items())

    shots = {str(count): {} for count in args.shots}
    equivariance = {}
    head = np.eye(basis.shape[1]) / basis.shape[1]
    for name, metric, epochs in variants:
        student = fit_student(source, teacher, basis, mean, args, seed + 100, metric=metric, metric_epochs=epochs)
        support, queries = (embed(student, part, args, device) for part in (target, query))
        priors = {name: head}
        if args.reflection_readouts:
            priors["two_view"] = head
        scores = evaluate(priors, support, queries, target_classes, args, np.random.default_rng(seed + 500))
        for count, row in scores.items():
            if args.reflection_readouts:
                row[f"{name}_pooled"] = row.pop("two_view")
            shots[count].update(row)
        if args.reflection_readouts:
            even = ~odd
            even_support, even_query = ({label: views[..., even] for label, views in part.items()}
                                        for part in (support, queries))
            even_head = np.eye(even.sum()) / even.sum()
            scores = evaluate({f"{name}_even": even_head}, even_support, even_query,
                              target_classes, args, np.random.default_rng(seed + 500))
            for count, row in scores.items():
                shots[count].update(row)
            clean, reflected = np.concatenate(list(queries.values()), axis=1).astype(np.float64)
            equivariance[name] = {}
            for label, mask, sign in (("even", even, 1), ("odd", odd, -1)):
                first, second = clean[:, mask], reflected[:, mask]
                energy = np.square(first).sum() + np.square(second).sum()
                equivariance[name][label] = float(np.square(second - sign * first).sum() / max(energy, 1e-12))
    result = {"seed": seed, "source_classes": source_classes, "target_classes": target_classes,
              "latent_width": basis.shape[1], "teacher_odd_count": int(odd.sum()), "shots": shots}
    if args.reflection_readouts:
        result["reflection_error"] = equivariance
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/tmp/ibt-cifar-data"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1201, 1217)))
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
    parser.add_argument("--reflection-readouts", action="store_true")
    args = parser.parse_args()
    if args.output is None:
        folder = HERE / "results" / "class_pair_reflection_readouts" if args.reflection_readouts else RESULTS
        args.output = folder / "metrics.json"
    if args.student_epochs < 2 or args.probe_examples < 2:
        parser.error("need at least two student epochs and two probe examples per source class")
    if not 0 < min(args.shots) <= max(args.shots) <= args.target_pool <= 5000:
        parser.error("shots must fit a nonempty target pool of at most 5000 examples per class")
    if not 0 < args.source_examples <= 5000 - args.probe_examples or not 0 < args.test_per_class <= 1000:
        parser.error("source, probe and test counts must fit their CIFAR-10 splits")
    if min(args.batch_size, args.teacher_epochs, args.episodes) < 1 or not 0 < args.noise < float("inf"):
        parser.error("batch size, teacher epochs, episodes and noise must be positive and finite")
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
