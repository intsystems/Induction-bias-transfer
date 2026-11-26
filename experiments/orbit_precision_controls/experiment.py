"""Factorially isolate orbit geometry from the semantics of the teacher."""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

from torch.nn import functional as F
from torchvision.datasets import MNIST

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from orbit_pullback_prior.experiment import OFFSETS, distill, feasible_basis
from orbit_spectral_prior.experiment import (FAR, NEAR, Teacher, balanced_indices, batches,
                                             evaluate, features, prepare, shift, train_teacher)


def normalize_covariance(rows):
    covariance = rows.T @ rows / len(rows)
    trace_mean = covariance.trace() / covariance.shape[0]
    identity = torch.eye(covariance.shape[0], device=covariance.device)
    return 0.95 * covariance / trace_mean.clamp_min(1e-8) + 0.05 * identity


def covariance_choices(student, x, args, seed):
    encoded = features(student.stem, x, OFFSETS, args.batch_size)
    mean = encoded.mean(dim=0)
    orbit = normalize_covariance((encoded - mean).flatten(0, 1))
    among_samples = normalize_covariance(mean - mean.mean(dim=0))
    rng = torch.Generator().manual_seed(seed)
    size = orbit.shape[0]
    axes, _ = torch.linalg.qr(torch.randn(size, size, generator=rng, dtype=torch.float64))
    axes = axes.to(orbit)
    return {"none": None, "isotropic": torch.eye(size, device=x.device),
            "rotated": axes @ orbit @ axes.T, "between": among_samples, "orbit": orbit}


@torch.inference_mode()
def near_accuracy(model, x, y):
    correct = sum((model(shift(x, offset)).argmax(dim=-1) == y).sum().item() for offset in NEAR)
    return correct / (len(NEAR) * len(y))


def adapt(model, kernel, train, test, support_aug, args, seed):
    model.stem.requires_grad_(False)
    x, y = train
    weights = model.adapter.weight
    center = weights.detach().clone()
    optimizer = torch.optim.AdamW([*model.adapter.parameters(), *model.head.parameters()], lr=0.003)
    rng = torch.Generator().manual_seed(seed)
    aug_rng = torch.Generator().manual_seed(seed + 1000)
    augmentation = torch.stack([shift(x, offset) for offset in OFFSETS])
    history = []
    for epoch in range(args.epochs + 1):
        if epoch % 10 == 0 or epoch == args.epochs:
            history.append({"epoch": epoch, **evaluate(model, *test, args.batch_size),
                            "near_accuracy": near_accuracy(model, *test)})
        if epoch == args.epochs:
            break
        x_epoch = x
        if support_aug:
            views = torch.randint(len(OFFSETS), (len(x),), generator=aug_rng)
            x_epoch = augmentation[views, torch.arange(len(x))]
        for ix in batches(len(x), args.target_batch_size, rng):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x_epoch[ix]), y[ix])
            if kernel is not None:
                deviation = weights - center
                loss = loss + args.prior_weight * ((deviation @ kernel) * deviation).mean()
            loss.backward()
            optimizer.step()
    return history


def canonical_targets(teacher, source, probe, args):
    z_probe = features(teacher.encoder, probe, OFFSETS, args.batch_size)
    mean, basis, spectrum = feasible_basis(z_probe)
    z = (features(teacher.encoder, source, OFFSETS, args.batch_size) - mean) @ basis
    return z, spectrum


def run(args, seed, train_data, test_data, device):
    rng = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(10, generator=rng).tolist()
    source_classes, target_classes = sorted(permutation[:5]), sorted(permutation[5:])
    source = balanced_indices(train_data.targets, source_classes, args.source_per_class, rng)
    targets = balanced_indices(train_data.targets, target_classes, max(args.shots), rng)
    tests = balanced_indices(test_data.targets, target_classes, args.test_per_class, rng)
    probe_count = args.source_per_class // 5
    x, y = prepare(train_data.data, source[:, :-probe_count], device)
    probe, _ = prepare(train_data.data, source[:, -probe_count:], device)
    test = prepare(test_data.data, tests, device)
    teacher = train_teacher(x, y, args, seed)
    torch.manual_seed(seed)
    controls = {"trained": teacher, "untrained": Teacher().to(device).eval()}
    latent = {key: canonical_targets(model, x, probe, args) for key, model in controls.items()}
    width = min(args.max_width, *(z.shape[-1] for z, _ in latent.values()))
    result = {"seed": seed, "source_classes": source_classes, "target_classes": target_classes,
              "width": width, "encoders": {}}
    for key, (z, spectrum) in latent.items():
        coordinates = torch.linspace(0, z.shape[-1] - 1, width, device=device).round().long()
        z = z[:, :, coordinates]
        stable = max(1, width // 3)
        z[:, :, :stable] = z[:, :, :stable].mean(dim=0)
        student = distill(x, z, OFFSETS, args, seed + 1000)
        kernels = covariance_choices(student, probe, args, seed + 3000)
        kernels["target_augmentation"] = None
        result["encoders"][key] = {"spectrum": spectrum.tolist(), "shots": {}}
        for shot in args.shots:
            support = prepare(train_data.data, targets[:, :shot], device)
            output = result["encoders"][key]["shots"][str(shot)] = {}
            for name, kernel in kernels.items():
                output[name] = adapt(copy.deepcopy(student), kernel, support, test,
                                     name == "target_augmentation", args, seed + 2000)
                print(seed, key, shot, name, output[name][-1], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data", type=Path, default=HERE.parent / "image_shift" / "data")
    parser.add_argument("--output", type=Path, default=HERE / "results/metrics.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 13)))
    parser.add_argument("--shots", type=int, nargs="+", default=[10, 40])
    parser.add_argument("--source-per-class", type=int, default=1800)
    parser.add_argument("--test-per-class", type=int, default=400)
    parser.add_argument("--max-width", type=int, default=24)
    parser.add_argument("--teacher-epochs", type=int, default=6)
    parser.add_argument("--init-epochs", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-batch-size", type=int, default=32)
    parser.add_argument("--prior-weight", type=float, default=100)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    train = MNIST(args.data, train=True, download=True)
    test = MNIST(args.data, train=False, download=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
              "near_offsets": NEAR, "far_offsets": FAR, "torch_version": torch.__version__, "runs": []}
    start = time.monotonic()
    for seed in args.seeds:
        result["runs"].append(run(args, seed, train, test, device))
        result["elapsed_seconds"] = time.monotonic() - start
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
