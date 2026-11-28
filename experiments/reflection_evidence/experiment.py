"""Adapt a teacher-derived reflection prior to held-out traffic-sign classes."""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import GTSRB
from torchvision.transforms import Compose, PILToTensor, Resize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from orbit_spectral_prior.experiment import batches

TASKS = {
    "invariant": ((11, 22), (18, 26), (12, 13), (15, 17)),
    "antisymmetric": ((19, 20), (33, 34), (36, 37), (38, 39)),
}
TARGET_IDS = {label for tasks in TASKS.values() for pair in tasks for label in pair}
SOURCE_IDS = [label for label in range(43) if label not in TARGET_IDS]
MIXTURE = (0.02, 0.2, 0.5, 0.8, 0.98)
TRANSFORM = Compose([Resize((32, 32)), PILToTensor()])


def reflect(x):
    return x.flip(-1)


class Teacher(nn.Module):
    def __init__(self, classes=len(SOURCE_IDS), *, view=reflect):
        super().__init__()
        self.view = view
        self.base = nn.Sequential(
            nn.Conv2d(3, 24, 5, padding=2), nn.GELU(), nn.AvgPool2d(2),
            nn.Conv2d(24, 32, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(),
        )
        self.head = nn.Linear(256, classes)

    def encode(self, x):
        clean, mirrored = self.base(torch.cat([x, self.view(x)])).chunk(2)
        return torch.cat(((clean + mirrored) / 2, (clean - mirrored) / 2), dim=1)

    def forward(self, x):
        return self.head(self.encode(x))


def paths_by_class(dataset):
    groups = defaultdict(list)
    for path, label in dataset._samples:
        groups[label].append(Path(path))
    return groups


def tracks(paths):
    result = defaultdict(list)
    for path in paths:
        result[path.stem.split("_")[0]].append(path)
    return list(result.values())


def images(paths):
    samples = []
    for path in paths:
        with Image.open(path) as image:
            samples.append(TRANSFORM(image.convert("RGB")))
    return torch.stack(samples)


def preprocess(train, test, args, *, target_ids=None, include_source_paths=False, include_probe_paths=False):
    rng = np.random.default_rng(2025)
    train_paths = paths_by_class(train)
    test_paths = paths_by_class(test)
    source, labels, probe, probe_labels = [], [], [], []
    source_ids = SOURCE_IDS if target_ids is None else [label for label in sorted(train_paths) if label not in target_ids]
    target_ids = TARGET_IDS if target_ids is None else target_ids
    for index, label in enumerate(source_ids):
        groups = tracks(train_paths[label])
        split = max(1, len(groups) // 4)
        permutation = rng.permutation(len(groups))
        for subset, paths, ids, limit in (
            (permutation[split:], source, labels, args.source_examples),
            (permutation[:split], probe, probe_labels, args.probe_examples),
        ):
            available = [path for group in subset for path in groups[group]]
            selected = rng.choice(len(available), min(len(available), limit), replace=False)
            paths.extend(available[i] for i in selected)
            ids.extend([index] * len(selected))
    target_train, target_test, target_tracks = {}, {}, {}
    for label in sorted(target_ids):
        paths = train_paths[label]
        target_train[label] = images(paths)
        target_test[label] = images(test_paths[label])
        indices = {path: index for index, path in enumerate(paths)}
        target_tracks[label] = [[indices[path] for path in group] for group in tracks(paths)]
    data = (
        (images(source), torch.tensor(labels)), (images(probe), torch.tensor(probe_labels)),
        target_train, target_test, target_tracks,
    )
    if include_source_paths:
        data = (*data, source)
    if include_probe_paths:
        data = (*data, probe)
    return data


def fit_teacher(x, y, args, seed, classes=len(SOURCE_IDS), *, view=reflect):
    torch.manual_seed(seed)
    teacher = Teacher(classes, view=view).to(x.device)
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=0.003)
    rng = torch.Generator().manual_seed(seed)
    for _ in range(args.teacher_epochs):
        for ix in batches(len(x), args.batch_size, rng):
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(teacher(x[ix]), y[ix]).backward()
            optimizer.step()
    return teacher.eval()


@torch.no_grad()
def discover_basis(teacher, probe, args):
    clean = torch.cat([teacher.encode(x) for x in probe.split(args.batch_size)]).double()
    mirror = torch.cat([teacher.encode(teacher.view(x)) for x in probe.split(args.batch_size)]).double()
    mean = (clean + mirror).mean(dim=0) / 2
    all_centered = torch.cat([clean - mean, mirror - mean])
    total = all_centered.T @ all_centered / len(all_centered)
    values, vectors = torch.linalg.eigh(total)
    keep = values > 0.0005 * values.max()
    stencil = vectors[:, keep] * values[keep].rsqrt()
    differences = ((clean - mirror) / 2) @ stencil
    energies, directions = torch.linalg.eigh(differences.T @ differences / len(clean))
    odd = energies > 0.5
    if odd.all() or (~odd).all():
        raise RuntimeError("Both reflection components are required")
    return mean.float(), (stencil @ directions).float(), energies.clamp(0, 1).float()


def fit_student(x, teacher, basis, mean, args, seed, *, training_views=None):
    torch.manual_seed(seed)
    student = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 192), nn.GELU(),
                            nn.Linear(192, basis.shape[1])).to(x.device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.002)
    rng = torch.Generator().manual_seed(seed)
    views = (teacher.view,) if training_views is None else training_views
    with torch.no_grad():
        target = torch.cat([(teacher.encode(batch) - mean) @ basis for batch in x.split(args.batch_size)])
        targets_by_view = [torch.cat([(teacher.encode(view(batch)) - mean) @ basis
                                     for batch in x.split(args.batch_size)]) for view in views]
    for _ in range(args.student_epochs):
        for ix in batches(len(x), args.batch_size, rng):
            optimizer.zero_grad(set_to_none=True)
            batch = x[ix]
            prediction = student(torch.cat([batch, *(view(batch) for view in views)]))
            F.mse_loss(prediction, torch.cat([target[ix], *(values[ix] for values in targets_by_view)])).backward()
            optimizer.step()
    return student.eval()


def prior_kernels(energies, rng):
    width = len(energies)
    odd = np.diag(energies > 0.5).astype(float)
    even = np.eye(width) - odd
    odd_count = odd.trace()
    even_count = width - odd_count
    mixture = np.stack([(1 - q) * even / even_count + q * odd / odd_count for q in MIXTURE])
    rotation, _ = np.linalg.qr(rng.normal(size=(width, width)))
    return {
        "isotropic": np.eye(width)[None] / width,
        "fixed_even": mixture[0:1],
        "fixed_odd": mixture[-1:],
        "adaptive": mixture,
        "random_adaptive": rotation @ mixture @ rotation.T,
    }


def predict(kernels, inputs, values, test, noise, center_evidence=False):
    # Work in observation space. With few labels this only solves tiny systems.
    products = inputs @ kernels
    covariances = products @ inputs.T + noise ** 2 * np.eye(len(inputs))
    alpha = np.linalg.solve(covariances, np.broadcast_to(values[:, None], (*covariances.shape[:-1], 1)))
    alpha = alpha[..., 0]
    if center_evidence:
        precision = np.linalg.inv(covariances)
        count = len(values)
        centered_trace = np.trace(precision, axis1=-2, axis2=-1) - precision.sum(axis=(-2, -1)) / count
        expected = np.square(values).sum() * centered_trace / (count - 1)
        # The centered score removes the mean preference under balanced null labels.
        log_evidence = -(alpha @ values - expected) / 2
    else:
        log_evidence = -0.5 * (np.linalg.slogdet(covariances)[1] + alpha @ values)
    probs = np.exp(log_evidence - log_evidence.max())
    probs /= probs.sum()
    coefficients = (products.transpose(0, 2, 1) @ alpha[..., None])[..., 0]
    return test @ (probs @ coefficients), probs


@torch.inference_mode()
def embed(student, dataset, args, device, *, view=reflect):
    embeddings = {}
    for label, images_by_class in dataset.items():
        clean, mirrored = [], []
        for batch in images_by_class.split(args.batch_size):
            batch = batch.to(device, dtype=torch.float32) / 255
            clean.append(student(batch).cpu())
            mirrored.append(student(view(batch)).cpu())
        embeddings[label] = np.stack([torch.cat(clean).numpy(), torch.cat(mirrored).numpy()])
    return embeddings


def evaluate_task(pair, sign, train, test, target_tracks, kernels, args, rng):
    selected = []
    for label in pair:
        available = target_tracks[label]
        chosen = rng.choice(len(available), max(args.shots), replace=False)
        selected.append([rng.choice(available[i]) for i in chosen])
    clean = np.concatenate([test[label][0] for label in pair])
    mirrored = np.concatenate([test[label][1] for label in pair])
    y = np.concatenate([np.full(len(test[label][0]), label_id) for label, label_id in zip(pair, (-1, 1))])
    outputs = {}
    for n in args.shots:
        f = np.concatenate([train[label][0, ids[:n]] for label, ids in zip(pair, selected)])
        train_y = np.repeat([-1, 1], n)
        outputs[str(n)] = {}
        for name, covariances in kernels.items():
            raw, weights = predict(covariances, f, train_y, np.concatenate((clean, mirrored)), args.noise)
            clean_pred, mirror_pred = np.where(raw >= 0, 1, -1).reshape(2, -1)
            scores = {}
            for metric, outcome in (
                ("clean_accuracy", clean_pred == y), ("mirror_accuracy", mirror_pred == sign * y),
                ("parity_agreement", mirror_pred == sign * clean_pred),
            ):
                scores[metric] = float((outcome[y == -1].mean() + outcome[y == 1].mean()) / 2)
            if len(weights) == len(MIXTURE):
                scores["mean_odd_share"] = float(weights @ MIXTURE)
            outputs[str(n)][name] = scores
    return outputs


def run(data, args, device, seed):
    rng = np.random.default_rng(seed)
    (source, labels), (probe, probe_labels), train, test, target_tracks = data
    source = source.to(device, dtype=torch.float32) / 255
    probe = probe.to(device, dtype=torch.float32) / 255
    teacher = fit_teacher(source, labels.to(device), args, seed)
    mean, basis, energies = discover_basis(teacher, probe, args)
    student = fit_student(source, teacher, basis, mean, args, seed + 100)
    with torch.no_grad():
        accuracy = torch.cat([teacher(batch).argmax(dim=1).cpu()
                              for batch in probe.split(args.batch_size)]).eq(probe_labels).float().mean().item()
        error = 0
        for batch in probe.split(args.batch_size):
            desired = (teacher.encode(batch) - mean) @ basis
            error += F.mse_loss(student(batch), desired, reduction="sum").item()
        projection_mse = error / (len(probe) * basis.shape[1])
    train = embed(student, train, args, device)
    test = embed(student, test, args, device)
    energies = energies.cpu().numpy()
    kernels = prior_kernels(energies, rng)
    scores = {family: {} for family in TASKS}
    for family, pairs in TASKS.items():
        for pair in pairs:
            sign = 1 if family == "invariant" else -1
            episodes = [evaluate_task(pair, sign, train, test, target_tracks, kernels, args, rng)
                        for _ in range(args.episodes)]
            means = {}
            for budget, methods in episodes[0].items():
                means[budget] = {
                    name: {key: float(np.mean([episode[budget][name][key] for episode in episodes]))
                           for key in measurements}
                    for name, measurements in methods.items()
                }
            scores[family]["-".join(map(str, pair))] = means
    print(seed, "probe accuracy", round(accuracy, 4), "embedding MSE", round(projection_mse, 4),
          "basis width", len(energies), flush=True)
    return {"seed": seed, "teacher_probe_accuracy": accuracy, "projection_mse": projection_mse,
            "spectral_energy": energies.tolist(), "tasks": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/tmp/ibt-gtsrb-data"))
    parser.add_argument("--output", type=Path, default=HERE / "results/metrics.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4, 5])
    parser.add_argument("--teacher-epochs", type=int, default=9)
    parser.add_argument("--student-epochs", type=int, default=12)
    parser.add_argument("--source-examples", type=int, default=240)
    parser.add_argument("--probe-examples", type=int, default=65)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--noise", type=float, default=0.5)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    data = preprocess(GTSRB(args.data, split="train", download=True),
                      GTSRB(args.data, split="test", download=True), args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics = {"config": {str(k): str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
               "torch_version": torch.__version__, "mixture": MIXTURE, "source_classes": SOURCE_IDS,
               "target_tasks": TASKS, "runs": []}
    started = time.monotonic()
    for seed in args.seeds:
        metrics["runs"].append(run(data, args, device, seed))
        metrics["elapsed_seconds"] = time.monotonic() - started
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(metrics, indent=2) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
