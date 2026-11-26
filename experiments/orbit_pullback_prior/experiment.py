"""Use held-out orbit secants to define a parameter-space prior on a dense adapter."""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from orbit_spectral_prior.experiment import (FAR, NEAR, balanced_indices, batches, evaluate,
                                             features, prepare, shift, train_teacher)


METHODS = ("scratch", "clean_init", "augmentation_init", "orbit_init", "isotropic",
           "axis_control", "reversed_control", "pullback")
OFFSETS = [(0, 0), *NEAR]


class Student(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.stem = nn.Sequential(nn.Flatten(), nn.Linear(36 * 36, 128), nn.GELU())
        self.adapter = nn.Linear(128, width, bias=False)
        self.head = nn.Linear(width, 5)

    def forward(self, x):
        return self.head(self.adapter(self.stem(x)))


def feasible_basis(z):
    z = z.double()
    centers = z.mean(dim=0)
    mean = centers.mean(dim=0)
    centered = centers - mean
    b = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = torch.linalg.eigh(b)
    # Remove poorly identified coordinates before ranking orbit sensitivity.
    keep = eigenvalues > 0.001 * eigenvalues[-1]
    if not keep.any():
        raise ValueError("Teacher has zero between-example variation")
    whitening = eigenvectors[:, keep] * eigenvalues[keep].rsqrt()
    residual = ((z - centers) @ whitening).flatten(0, 1)
    spectrum, directions = torch.linalg.eigh(residual.T @ residual / len(residual))
    return mean.float(), (whitening @ directions).float(), spectrum.float()


def distill(x, targets, offsets, args, seed):
    torch.manual_seed(seed)
    student = Student(targets.shape[-1]).to(x.device)
    optimizer = torch.optim.AdamW([*student.stem.parameters(), *student.adapter.parameters()],
                                  lr=0.002)
    rng = torch.Generator().manual_seed(seed)
    for _ in range(args.init_epochs):
        for offset, target in zip(offsets, targets, strict=True):
            for ix in batches(len(x), args.batch_size, rng):
                optimizer.zero_grad(set_to_none=True)
                prediction = student.adapter(student.stem(shift(x[ix], offset)))
                F.mse_loss(prediction, target[ix]).backward()
                optimizer.step()
    return student


def pullback_precision(student, probe, args, seed):
    h = features(student.stem, probe, OFFSETS, args.batch_size)
    residual = (h - h.mean(dim=0)).flatten(0, 1)
    covariance = residual.T @ residual / len(residual)
    dimension = covariance.shape[0]
    covariance = covariance / (covariance.trace() / dimension).clamp_min(1e-8)
    # A ridge gives every parameter a finite prior variance.
    covariance = 0.95 * covariance + 0.05 * torch.eye(dimension, device=probe.device)
    rng = torch.Generator().manual_seed(seed)
    axes = torch.randn(dimension, dimension, generator=rng, dtype=torch.float64)
    rotation, _ = torch.linalg.qr(axes)
    rotation = rotation.to(covariance)
    return covariance, rotation @ covariance @ rotation.T


def target_fit(model, prior, train, test, args, seed, freeze):
    if freeze:
        model.stem.requires_grad_(False)
    x, y = train
    weight = model.adapter.weight
    center = weight.detach().clone()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.003)
    rng = torch.Generator().manual_seed(seed)
    history = [{"epoch": 0, **evaluate(model, *test, args.batch_size)}]
    for epoch in range(1, args.epochs + 1):
        for ix in batches(len(x), args.target_batch_size, rng):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[ix]), y[ix])
            if prior is not None:
                rows, kernel = prior
                delta = weight - center
                loss = loss + args.prior_weight * (rows[:, None] * (delta @ kernel) * delta).mean()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0 or epoch == args.epochs:
            history.append({"epoch": epoch, **evaluate(model, *test, args.batch_size)})
    return history


def run(args, seed, train_data, test_data, device):
    rng = torch.Generator().manual_seed(seed)
    source_classes = list(range(5)) if seed % 2 else list(range(5, 10))
    target_classes = [label for label in range(10) if label not in source_classes]
    source = balanced_indices(train_data.targets, source_classes, args.source_per_class, rng)
    chosen = balanced_indices(train_data.targets, target_classes, max(args.shots), rng)
    test_ids = balanced_indices(test_data.targets, target_classes, args.test_per_class, rng)
    test = prepare(test_data.data, test_ids, device)
    probe_count = max(1, args.source_per_class // 5)
    x, y = prepare(train_data.data, source[:, :-probe_count], device)
    probe, probe_y = prepare(train_data.data, source[:, -probe_count:], device)
    teacher = train_teacher(x, y, args, seed)
    mean, basis, spectrum = feasible_basis(features(teacher.encoder, probe, OFFSETS, args.batch_size))
    z = (features(teacher.encoder, x, OFFSETS, args.batch_size) - mean) @ basis
    stable = max(1, len(spectrum) // 3)
    canonical = z.clone()
    canonical[:, :, :stable] = z[:, :, :stable].mean(dim=0)
    clean = z[0:1].expand_as(z)
    # Repeat the clean view to match the five training passes per epoch.
    clean_init = distill(x, clean, [(0, 0)] * len(OFFSETS), args, seed + 1000)
    augmentation_init = distill(x, z, OFFSETS, args, seed + 1000)
    orbit_init = distill(x, canonical, OFFSETS, args, seed + 1000)
    covariance, rotated = pullback_precision(orbit_init, probe, args, seed + 3000)
    rows = z.new_full((len(spectrum),), 0.25)
    rows[:stable] += 0.75 * len(rows) / stable
    flat = torch.ones_like(rows)
    identity = torch.eye(covariance.shape[0], device=device)
    torch.manual_seed(seed + 1000)
    scratch = Student(len(spectrum)).to(device)
    variants = {
        "scratch": (scratch, None, False), "clean_init": (clean_init, None, True),
        "augmentation_init": (augmentation_init, None, True), "orbit_init": (orbit_init, None, True),
        "isotropic": (orbit_init, (flat, identity), True),
        "axis_control": (orbit_init, (rows, rotated), True),
        "reversed_control": (orbit_init, (rows.flip(0), covariance), True),
        "pullback": (orbit_init, (rows, covariance), True),
    }
    with torch.no_grad():
        probe_accuracy = (teacher(probe).argmax(dim=1) == probe_y).float().mean().item()
        stem_eigenvalues = torch.linalg.eigvalsh(covariance).tolist()
    result = {"seed": seed, "source_classes": source_classes, "target_classes": target_classes,
              "teacher_probe_accuracy": probe_accuracy, "orbit_spectrum": spectrum.tolist(),
              "stable_count": stable, "stem_precision_spectrum": stem_eigenvalues, "shots": {}}
    for budget in args.shots:
        train = prepare(train_data.data, chosen[:, :budget], device)
        result["shots"][str(budget)] = {}
        for method, (initial, prior, freeze) in variants.items():
            trajectory = target_fit(copy.deepcopy(initial), prior, train, test, args, seed + 2000, freeze)
            result["shots"][str(budget)][method] = trajectory
            print(seed, budget, method, trajectory[-1], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data", type=Path, default=HERE.parent / "image_shift" / "data")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "metrics.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--shots", type=int, nargs="+", default=[10, 40])
    parser.add_argument("--source-per-class", type=int, default=2400)
    parser.add_argument("--test-per-class", type=int, default=400)
    parser.add_argument("--teacher-epochs", type=int, default=6)
    parser.add_argument("--init-epochs", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-batch-size", type=int, default=32)
    parser.add_argument("--prior-weight", type=float, default=100.0)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    train_data = MNIST(args.data, train=True, download=True)
    test_data = MNIST(args.data, train=False, download=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = {"config": {key: str(value) if isinstance(value, Path) else value
                       for key, value in vars(args).items()},
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "torch_version": torch.__version__,
            "near_offsets": NEAR, "far_offsets": FAR, "runs": []}
    start = time.monotonic()
    for seed in args.seeds:
        data["runs"].append(run(args, seed, train_data, test_data, device))
        data["elapsed_seconds"] = time.monotonic() - start
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
