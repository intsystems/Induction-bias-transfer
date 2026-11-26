"""Transfer a frozen teacher's orbit geometry to a dense student's weights."""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST


HERE = Path(__file__).resolve().parent
NEAR = [(0, 2), (0, -2), (2, 0), (-2, 0)]
FAR = [(4, 4), (4, -4), (-4, 4), (-4, -4)]
METHODS = ("scratch", "init", "isotropic", "orbit_prior", "reversed_prior")


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(),
        )
        self.head = nn.Linear(64, 5)

    def forward(self, x):
        return self.head(self.encoder(x))


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(36 * 36, 128), nn.GELU(),
                                     nn.Linear(128, 64, bias=False))
        self.head = nn.Linear(64, 5)

    def forward(self, x):
        return self.head(self.encoder(x))


def balanced_indices(labels, classes, count, generator):
    selected = []
    for label in classes:
        indices = (labels == label).nonzero().flatten()
        if count > len(indices):
            raise ValueError(f"Class {label} has only {len(indices)} items")
        selected.append(indices[torch.randperm(len(indices), generator=generator)[:count]])
    return torch.stack(selected)


def prepare(pixels, indices, device):
    x = pixels[indices.reshape(-1)].to(device, dtype=torch.float32).unsqueeze(1) / 255
    labels = torch.arange(len(indices), device=device).repeat_interleave(indices.shape[1])
    return F.pad(x, (4, 4, 4, 4)), labels


def shift(x, offset):
    # The zero halo is at least as wide as every displacement used below.
    return torch.roll(x, offset, dims=(-2, -1))


def batches(n, size, generator):
    return torch.randperm(n, generator=generator).split(size)


def features(encoder, x, offsets, batch_size):
    with torch.inference_mode():
        encoded = [torch.cat([encoder(shift(batch, offset)) for batch in x.split(batch_size)])
                   for offset in offsets]
    return torch.stack(encoded)


def orbit_basis(z):
    # z indexes orbit views, examples, channels. Estimate within-orbit scatter
    # and between-example scatter of the orbit centers independently.
    z = z.double()
    centers = z.mean(dim=0)
    mean = centers.mean(dim=0)
    between = centers - mean
    within = (z - centers).reshape(-1, z.shape[-1])
    b = between.T @ between / len(between)
    w = within.T @ within / len(within)
    eigenvalues, eigenvectors = torch.linalg.eigh(b)
    regularizer = (0.01 * b.trace() / b.shape[0]).clamp_min(1e-10)
    whitener = eigenvectors * (eigenvalues.clamp_min(0) + regularizer).rsqrt()
    spectrum, directions = torch.linalg.eigh(whitener.T @ w @ whitener)
    return mean.float(), (whitener @ directions).float(), spectrum.clamp_min(0).float()


def train_teacher(x, y, args, seed):
    torch.manual_seed(seed)
    teacher = Teacher().to(x.device)
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=0.003)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(args.teacher_epochs):
        for ix in batches(len(x), args.batch_size, generator):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(teacher(x[ix]), y[ix])
            loss.backward()
            optimizer.step()
    return teacher.eval()


def initialize(x, target, args, seed):
    torch.manual_seed(seed)
    student = Student().to(x.device)
    # Preserve paired initial heads and identical pretraining budget across priors.
    optimizer = torch.optim.AdamW(student.encoder.parameters(), lr=0.002)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(args.init_epochs):
        for ix in batches(len(x), args.batch_size, generator):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(student.encoder(x[ix]), target[ix])
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        error = F.mse_loss(student.encoder(x), target).item()
    return student, error


@torch.inference_mode()
def evaluate(model, x, y, batch_size):
    clean_correct = shifted_correct = all_correct = disagree = 0
    for batch, labels in zip(x.split(batch_size), y.split(batch_size)):
        clean = model(batch).argmax(dim=1)
        shifted = torch.stack([model(shift(batch, offset)).argmax(dim=1) for offset in FAR])
        clean_correct += (clean == labels).sum().item()
        correct = shifted == labels
        shifted_correct += correct.sum().item()
        all_correct += correct.all(dim=0).sum().item()
        disagree += (shifted != clean).sum().item()
    return {
        "clean_accuracy": clean_correct / len(y),
        "shifted_accuracy": shifted_correct / (len(y) * len(FAR)),
        "all_views_accuracy": all_correct / len(y),
        "disagreement": disagree / (len(y) * len(FAR)),
    }


def fit_student(model, precision, train, test, args, seed):
    x, y = train
    projection = model.encoder[-1].weight
    center = projection.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    generator = torch.Generator().manual_seed(seed)
    # The empirical precision has mean 1 for all regularized arms.
    history = [{"epoch": 0, **evaluate(model, *test, args.batch_size)}]
    for epoch in range(1, args.epochs + 1):
        for ix in batches(len(x), args.batch_size, generator):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[ix]), y[ix])
            if precision is not None:
                deviation = (projection - center).square().mean(dim=1)
                loss = loss + args.prior_weight * (precision * deviation).mean()
            loss.backward()
            optimizer.step()
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            history.append({"epoch": epoch, **evaluate(model, *test, args.batch_size)})
    return history


def run_seed(args, seed, train_data, test_data, device):
    rng = torch.Generator().manual_seed(seed)
    source_classes = list(range(5))
    target_classes = list(range(5, 10))
    source = balanced_indices(train_data.targets, source_classes, args.source_per_class, rng)
    candidates = balanced_indices(train_data.targets, target_classes, max(args.shots), rng)
    test_indices = balanced_indices(test_data.targets, target_classes, args.test_per_class, rng)
    test = prepare(test_data.data, test_indices, device)
    if source.shape[1] < 2:
        raise ValueError("At least two source examples per class are needed")
    probe_count = max(1, source.shape[1] // 5)
    teacher_train = prepare(train_data.data, source[:, :-probe_count], device)
    probe_x, probe_y = prepare(train_data.data, source[:, -probe_count:], device)
    teacher = train_teacher(*teacher_train, args, seed)
    z = features(teacher.encoder, probe_x, [(0, 0), *NEAR], args.batch_size)
    mean, basis, spectrum = orbit_basis(z)
    # Rank (rather than inverse raw eigenvalues) limits the precision ratio to 4:1.
    precision = torch.linspace(1.6, 0.4, len(spectrum), device=device)
    x_source, _ = teacher_train
    target = (features(teacher.encoder, x_source, [(0, 0)], args.batch_size)[0] - mean) @ basis
    initialized, feature_mse = initialize(x_source, target, args, seed + 1000)

    with torch.no_grad():
        teacher_probe_accuracy = (teacher.head(z[0]).argmax(dim=1) == probe_y).float().mean().item()
    run = {
        "seed": seed, "teacher_probe_accuracy": teacher_probe_accuracy,
        "feature_mse": feature_mse, "orbit_spectrum": spectrum.tolist(), "shots": {},
    }
    for shots in args.shots:
        train = prepare(train_data.data, candidates[:, :shots], device)
        torch.manual_seed(seed + 1000)
        scratch = Student().to(device)
        precisions = {"scratch": None, "init": None, "isotropic": torch.ones_like(precision),
                      "orbit_prior": precision, "reversed_prior": precision.flip(0)}
        arm_results = {}
        for method in METHODS:
            model = copy.deepcopy(scratch if method == "scratch" else initialized)
            arm_results[method] = fit_student(model, precisions[method], train, test, args, seed + 2000)
            print(seed, shots, method, arm_results[method][-1], flush=True)
        run["shots"][str(shots)] = arm_results
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data", type=Path, default=HERE.parent / "image_shift" / "data")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "metrics.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--shots", type=int, nargs="+", default=[10, 40])
    parser.add_argument("--source-per-class", type=int, default=2400)
    parser.add_argument("--test-per-class", type=int, default=400)
    parser.add_argument("--teacher-epochs", type=int, default=6)
    parser.add_argument("--init-epochs", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--prior-weight", type=float, default=10.0)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    train = MNIST(args.data, train=True, download=True)
    test = MNIST(args.data, train=False, download=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "near_offsets": NEAR, "far_offsets": FAR, "runs": [],
    }
    started = time.monotonic()
    for seed in args.seeds:
        metrics["runs"].append(run_seed(args, seed, train, test, device))
        metrics["elapsed_seconds"] = time.monotonic() - started
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(metrics, indent=2) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
