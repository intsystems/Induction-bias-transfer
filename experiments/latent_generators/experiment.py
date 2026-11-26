"""Learn transferable instance-wise augmentation from a teacher's stable hidden subspace."""

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from orbit_pullback_prior.experiment import feasible_basis
from orbit_spectral_prior.experiment import (FAR, Student, balanced_indices, batches, evaluate,
                                             prepare, train_teacher)

# Entries are (rotation in degrees, horizontal shear, x shift, y shift).
TRANSFORMS = (
    (-23, 0, 0, 0), (23, 0, 0, 0), (0, -0.28, 0, 0), (0, 0.28, 0, 0),
    (0, 0, -3, 0), (0, 0, 3, 0), (0, 0, 0, -3), (0, 0, 0, 3),
)
TEST_AFFINE = ((-38, 0, 0, 0), (38, 0, 0, 0), (0, -0.43, 0, 0), (0, 0.43, 0, 0))
METHODS = ("clean", "uniform", "logit", "spectral", "spectral_marginal", "spectral_class_shuffle")


def transform(x, parameters):
    angle, shear, dx, dy = parameters
    angle = math.radians(angle)
    cosine, sine = math.cos(angle), math.sin(angle)
    matrix = x.new_tensor([[cosine, -sine + shear, -2 * dx / x.shape[-1]],
                           [sine, cosine, -2 * dy / x.shape[-2]]])
    grid = F.affine_grid(matrix.expand(len(x), -1, -1), x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 12, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(12, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(3), nn.Flatten(), nn.Linear(144, len(TRANSFORMS)),
        )

    def forward(self, x):
        return self.network(x)


@torch.no_grad()
def source_costs(teacher, x, batch_size):
    clean = torch.cat([teacher.encoder(batch) for batch in x.split(batch_size)])
    views = torch.stack([
        torch.cat([teacher.encoder(transform(batch, params)) for batch in x.split(batch_size)])
        for params in TRANSFORMS
    ])
    _, basis, spectrum = feasible_basis(torch.cat([clean[None], views]))
    stable_dim = max(1, len(spectrum) // 3)
    difference = (views - clean) @ basis[:, :stable_dim]
    spectral = difference.square().mean(dim=-1).T
    logits = teacher.head(clean).log_softmax(dim=-1)
    shifted = teacher.head(views).log_softmax(dim=-1)
    logit = (logits.exp() * (logits - shifted)).sum(dim=-1).T
    return {
        "spectral": spectral / spectral.mean().clamp_min(1e-8),
        "logit": logit.clamp_min(0) / logit.mean().clamp_min(1e-8),
    }, spectrum.tolist()


def fit_generator(x, cost, args, seed):
    torch.manual_seed(seed)
    generator = Generator().to(x.device)
    optimizer = torch.optim.AdamW(generator.parameters(), lr=0.003)
    rng = torch.Generator().manual_seed(seed)
    for _ in range(args.generator_epochs):
        for ix in batches(len(x), args.batch_size, rng):
            optimizer.zero_grad(set_to_none=True)
            log_prob = generator(x[ix]).log_softmax(dim=-1)
            loss = (log_prob.exp() * (cost[ix] + args.temperature * log_prob)).sum(dim=-1).mean()
            loss.backward()
            optimizer.step()
    return generator.eval()


@torch.inference_mode()
def mixed_evaluation(student, test, batch_size):
    x, y = test
    metrics = evaluate(student, x, y, batch_size)
    correct = 0
    for params in TEST_AFFINE:
        for batch, labels in zip(x.split(batch_size), y.split(batch_size)):
            correct += (student(transform(batch, params)).argmax(dim=-1) == labels).sum().item()
    metrics["affine_accuracy"] = correct / (len(y) * len(TEST_AFFINE))
    return metrics


def fit_target(student, x, y, augmentations, probs, test, args, seed):
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.002)
    order_rng = torch.Generator().manual_seed(seed)
    aug_rng = torch.Generator(device=x.device).manual_seed(seed + 1000)
    history = [{"epoch": 0, **mixed_evaluation(student, test, args.batch_size)}]
    n = len(x)
    for epoch in range(1, args.epochs + 1):
        if probs is not None:
            choices = torch.multinomial(probs, 1, generator=aug_rng).squeeze(-1)
            generated = augmentations[choices, torch.arange(n, device=x.device)]
            views = torch.cat([x, generated])
            labels = y.repeat(2)
        else:
            views = x.repeat(2, 1, 1, 1)
            labels = y.repeat(2)
        for ix in batches(len(labels), args.target_batch_size, order_rng):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(student(views[ix]), labels[ix])
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0 or epoch == args.epochs:
            history.append({"epoch": epoch, **mixed_evaluation(student, test, args.batch_size)})
    return history


def policy_summary(probs):
    if probs is None:
        return {}
    return {
        "marginal": probs.mean(dim=0).tolist(),
        "conditional_entropy": (-probs * probs.clamp_min(1e-8).log()).sum(dim=1).mean().item(),
    }


def run(args, seed, train_data, test_data, device):
    rng = torch.Generator().manual_seed(seed)
    source_classes = list(range(5))
    target_classes = list(range(5, 10))
    source = balanced_indices(train_data.targets, source_classes, args.source_per_class, rng)
    target = balanced_indices(train_data.targets, target_classes, max(args.shots), rng)
    test_ids = balanced_indices(test_data.targets, target_classes, args.test_per_class, rng)
    test = prepare(test_data.data, test_ids, device)
    probe_count = args.source_per_class // 3
    x, y = prepare(train_data.data, source[:, :-probe_count], device)
    probe, probe_y = prepare(train_data.data, source[:, -probe_count:], device)
    teacher = train_teacher(x, y, args, seed)
    costs, spectrum = source_costs(teacher, probe, args.batch_size)
    generators = {name: fit_generator(probe, cost, args, seed + 1000)
                  for name, cost in costs.items()}
    with torch.no_grad():
        probe_acc = (teacher(probe).argmax(dim=1) == probe_y).float().mean().item()
    output = {"seed": seed, "teacher_probe_accuracy": probe_acc, "orbit_spectrum": spectrum, "shots": {}}
    torch.manual_seed(seed + 3000)
    initial = Student().to(device)
    for count in args.shots:
        x, y = prepare(train_data.data, target[:, :count], device)
        with torch.no_grad():
            spectral = generators["spectral"](x).softmax(dim=-1)
            logit = generators["logit"](x).softmax(dim=-1)
            shuffled = torch.cat([spectral[i * count + torch.randperm(count, generator=rng)]
                                  for i in range(len(target_classes))])
            augmentations = torch.stack([transform(x, p) for p in TRANSFORMS])
        policies = (None, torch.full_like(spectral, 1 / len(TRANSFORMS)), logit,
                    spectral, spectral.mean(dim=0).expand_as(spectral), shuffled)
        output["shots"][str(count)] = {}
        for method, policy in zip(METHODS, policies, strict=True):
            trajectory = fit_target(copy.deepcopy(initial), x, y, augmentations, policy, test,
                                    args, seed + 4000)
            output["shots"][str(count)][method] = {"curve": trajectory, **policy_summary(policy)}
            print(seed, count, method, trajectory[-1], flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data", type=Path, default=HERE.parent / "image_shift" / "data")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "metrics.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--shots", type=int, nargs="+", default=[10, 40])
    parser.add_argument("--source-per-class", type=int, default=2400)
    parser.add_argument("--test-per-class", type=int, default=400)
    parser.add_argument("--teacher-epochs", type=int, default=6)
    parser.add_argument("--generator-epochs", type=int, default=18)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-batch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.45)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    train = MNIST(args.data, train=True, download=True)
    test = MNIST(args.data, train=False, download=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics = {"config": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
               "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
               "torch_version": torch.__version__, "transforms": TRANSFORMS,
               "test_affine": TEST_AFFINE, "test_shifts": FAR, "runs": []}
    started = time.monotonic()
    for seed in args.seeds:
        metrics["runs"].append(run(args, seed, train, test, device))
        metrics["elapsed_seconds"] = time.monotonic() - started
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(metrics, indent=2) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
