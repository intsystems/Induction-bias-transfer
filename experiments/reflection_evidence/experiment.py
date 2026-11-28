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
    def __init__(self, classes=len(SOURCE_IDS), *, view=reflect, equivariant=True):
        super().__init__()
        self.view = view
        self.equivariant = equivariant
        self.base = nn.Sequential(
            nn.Conv2d(3, 24, 5, padding=2), nn.GELU(), nn.AvgPool2d(2),
            nn.Conv2d(24, 32, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(),
        )
        self.head = nn.Linear(256 if equivariant else 128, classes)

    def encode(self, x):
        if not self.equivariant:
            return self.base(x)
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


def fit_teacher(x, y, args, seed, classes=len(SOURCE_IDS), *, view=reflect, equivariant=True):
    torch.manual_seed(seed)
    teacher = Teacher(classes, view=view, equivariant=equivariant).to(x.device)
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


def rotate_class_prototypes(prototypes, counts, seed):
    values = prototypes.double()
    weights = (counts.double() / counts.sum()).sqrt()
    contrast_count = len(weights) - 1
    rng = torch.Generator(device=values.device).manual_seed(seed)
    trial = torch.randn((len(weights), contrast_count), generator=rng, device=values.device, dtype=values.dtype)
    frame, _ = torch.linalg.qr(torch.column_stack((weights, trial)))
    contrasts = frame[:, 1:]
    trial = torch.randn((contrast_count, contrast_count), generator=rng, device=values.device, dtype=values.dtype)
    rotated, _ = torch.linalg.qr(trial)
    action = torch.outer(weights, weights) + contrasts @ rotated @ contrasts.T
    mean = (values * weights.square()[None, :, None]).sum(dim=1, keepdim=True)
    scale = weights[None, :, None]
    return (mean + (action @ ((values - mean) * scale)) / scale).to(prototypes.dtype)


def class_target_scale(targets, prototypes):
    mean = targets.mean(dim=1, keepdim=True)
    total = (targets - mean).square().mean()
    between = (prototypes - mean).square().mean()
    return (total.clamp_min(1e-12) / between.clamp_min(1e-12)).sqrt()


def class_target_means(views, labels, *, restore_variance, rotation_seed=None, within_class_weight=0.):
    targets = torch.stack(views)
    classes, membership, counts = torch.unique(labels, return_inverse=True, return_counts=True)
    means = torch.stack([targets[:, labels == label].mean(dim=1) for label in classes], dim=1)
    if rotation_seed is not None:
        means = rotate_class_prototypes(means, counts, rotation_seed)
    prototypes = means[:, membership]
    if within_class_weight:
        prototypes = torch.lerp(prototypes, targets, within_class_weight)
    if restore_variance:
        mean = targets.mean(dim=1, keepdim=True)
        prototypes = mean + (prototypes - mean) * class_target_scale(targets, prototypes)
    return prototypes.unbind(0)


def fit_student(x, teacher, basis, mean, args, seed, *, training_views=None, metric=None, metric_epochs=None,
                hidden_width=192, frozen_feature_epochs=None, reflection_action=None, reflection_penalty=0.,
                reflection_epochs=None, orbit_actions=None, orbit_order=None, invariant_variance_weight=0.,
                target_inputs=None, variance_projector=None, reflection_penalty_mask=None, reflection_strengths=None,
                match_reflection_residual=False, model_factory=None, teacher_reflection_from_action=False,
                target_transform=None):
    torch.manual_seed(seed)
    if model_factory is None:
        student = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, hidden_width), nn.GELU(),
                                nn.Linear(hidden_width, basis.shape[1]))
    else:
        student = model_factory()
    student.to(x.device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.002)
    rng = torch.Generator().manual_seed(seed)
    views = (teacher.view,) if training_views is None else training_views
    if reflection_action is not None and training_views is not None:
        raise ValueError("reflection projection requires the teacher's default training view")
    if reflection_penalty and reflection_action is None:
        raise ValueError("a reflection penalty requires a projection action")
    if reflection_strengths is not None and reflection_action is None:
        raise ValueError("reflection strengths require a projection action")
    if match_reflection_residual and (not reflection_penalty or reflection_strengths is not None):
        raise ValueError("matching reflection residuals requires a penalty with full projection")
    if teacher_reflection_from_action and reflection_action is None:
        raise ValueError("synthesizing reflected targets requires a reflection action")
    penalty_metric = metric
    if reflection_penalty_mask is not None and metric is not None:
        block = metric[reflection_penalty_mask][:, reflection_penalty_mask]
        penalty_metric = block / block.diag().mean()
    if orbit_actions is not None:
        if orbit_actions.ndim == 3:
            orbit_actions = orbit_actions.unsqueeze(0)
        if (reflection_action is not None or orbit_actions.ndim != 4
                or orbit_actions.shape[0] * orbit_actions.shape[1] != len(views) + 1):
            raise ValueError("orbit projection requires one action per view and no separate reflection action")
        if orbit_order is not None:
            orbit_order = torch.as_tensor(orbit_order, device=x.device, dtype=torch.long)
            if orbit_order.ndim == 1:
                orbit_order = orbit_order.unsqueeze(0)
            if orbit_order.ndim != 2 or not len(orbit_order) or orbit_order.shape[1] != len(views) + 1:
                raise ValueError("orbit order must contain complete permutations of the view indices")
            inverse_order = orbit_order.argsort(dim=1)
            sorted_indices = torch.gather(orbit_order, 1, inverse_order)
            if not torch.all(sorted_indices == torch.arange(len(views) + 1, device=x.device)):
                raise ValueError("orbit order must contain complete permutations of the view indices")
    elif orbit_order is not None:
        raise ValueError("orbit order requires orbit actions")
    if invariant_variance_weight:
        if variance_projector is None and (orbit_actions is None or orbit_actions.shape[0] != 1):
            raise ValueError("an invariant variance penalty requires one full orbit")
        invariant_projection = orbit_actions[0].mean(dim=0) if variance_projector is None else variance_projector
    supervision = x if target_inputs is None else target_inputs
    if supervision.shape != x.shape:
        raise ValueError("teacher target inputs must match the student's training input shape")
    with torch.no_grad():
        target = torch.cat([(teacher.encode(batch) - mean) @ basis for batch in supervision.split(args.batch_size)])
        if teacher_reflection_from_action:
            targets_by_view = [target @ reflection_action]
        else:
            targets_by_view = [torch.cat([(teacher.encode(view(batch)) - mean) @ basis
                                         for batch in supervision.split(args.batch_size)]) for view in views]
        if target_transform is not None:
            target, *targets_by_view = target_transform((target, *targets_by_view))
    for epoch in range(args.student_epochs):
        if frozen_feature_epochs is not None:
            student[1].requires_grad_(epoch not in frozen_feature_epochs)
        weighted = metric is not None and (metric_epochs is None or epoch in metric_epochs)
        project = reflection_action is not None and (reflection_epochs is None or epoch in reflection_epochs)
        if orbit_order is not None:
            order_index = epoch % len(orbit_order)
            view_order, restore_order = orbit_order[order_index], inverse_order[order_index]
        for ix in batches(len(x), args.batch_size, rng):
            optimizer.zero_grad(set_to_none=True)
            batch = x[ix]
            raw_prediction = student(torch.cat([batch, *(view(batch) for view in views)]))
            prediction = raw_prediction
            if project:
                pair = prediction.reshape(2, len(batch), -1)
                symmetrized = (pair + pair.flip(0) @ reflection_action) / 2
                if reflection_strengths is not None:
                    symmetrized = pair + reflection_strengths * (symmetrized - pair)
                prediction = symmetrized.flatten(0, 1)
            if orbit_actions is not None:
                orbit = prediction.reshape(len(views) + 1, len(batch), -1)
                if orbit_order is not None:
                    orbit = orbit[view_order]
                orbit = orbit.reshape(*orbit_actions.shape[:2], len(batch), -1)
                centered = (orbit @ orbit_actions.transpose(-2, -1)).mean(dim=1, keepdim=True)
                projected = (centered @ orbit_actions).flatten(0, 1)
                if orbit_order is not None:
                    projected = projected[restore_order]
                prediction = projected.flatten(0, 1)
            expected = torch.cat([target[ix], *(values[ix] for values in targets_by_view)])
            if not weighted:
                loss = F.mse_loss(prediction, expected)
            else:
                residual = prediction - expected
                loss = ((residual @ metric) * residual).mean()
            if project and reflection_penalty:
                complement = raw_prediction - prediction
                if match_reflection_residual:
                    expected_pair = expected.reshape(2, len(batch), -1)
                    expected_complement = (expected_pair - expected_pair.flip(0) @ reflection_action) / 2
                    complement = complement - expected_complement.flatten(0, 1)
                if reflection_penalty_mask is not None:
                    complement = complement[:, reflection_penalty_mask]
                energy = ((complement @ penalty_metric) * complement).mean() if weighted else complement.square().mean()
                loss = loss + reflection_penalty * energy
            if invariant_variance_weight:
                if variance_projector is None:
                    deviation = (raw_prediction - prediction) @ invariant_projection
                else:
                    orbit = raw_prediction.reshape(len(views) + 1, len(batch), -1)
                    deviation = (orbit - orbit.mean(dim=0)) @ invariant_projection
                loss = loss + invariant_variance_weight * deviation.square().mean()
            loss.backward()
            optimizer.step()
    if frozen_feature_epochs is not None:
        student[1].requires_grad_(True)
    return student.eval()


class FoldedReflectionStudent(nn.Module):
    def __init__(self, student, odd):
        super().__init__()
        first, last = student[1], student[3]
        with torch.no_grad():
            weights = first.weight.reshape(-1, 3, 32, 32)
            left, right = weights[..., :16], weights[..., 16:].flip(-1)
            self.register_buffer("input_even", (left + right).flatten(1))
            self.register_buffer("input_odd", (left - right).flatten(1))
            self.register_buffer("input_bias", first.bias.detach().clone())
            self.register_buffer("output_even", last.weight[~odd])
            self.register_buffer("output_even_bias", last.bias[~odd])
            self.register_buffer("output_odd", last.weight[odd])
            indices = torch.cat((torch.where(~odd)[0], torch.where(odd)[0]))
            self.register_buffer("output_order", indices.argsort())

    def forward(self, images):
        left, right = images[..., :16], images[..., 16:].flip(-1)
        even = F.linear(((left + right) / 2).flatten(1), self.input_even, self.input_bias)
        odd = F.linear(((left - right) / 2).flatten(1), self.input_odd)
        first, second = F.gelu(even + odd), F.gelu(even - odd)
        invariant = F.linear((first + second) / 2, self.output_even, self.output_even_bias)
        changing = F.linear((first - second) / 2, self.output_odd)
        return torch.cat((invariant, changing), dim=-1)[:, self.output_order]


class ParityStudent(nn.Module):
    def __init__(self, odd, width=192, *, gated=False):
        super().__init__()
        even_count, odd_count = int((~odd).sum()), int(odd.sum())
        self.even_input = nn.Linear(3 * 32 * 16, width)
        self.odd_input = nn.Linear(3 * 32 * 16, width, bias=False)
        self.even_output = nn.Linear(width, even_count)
        self.odd_output = nn.Linear(width, odd_count, bias=False)
        order = torch.cat((torch.where(~odd)[0], torch.where(odd)[0])).argsort()
        self.register_buffer("output_order", order)
        self.gated = gated

    def forward(self, images):
        left, right = images[..., :16], images[..., 16:].flip(-1)
        even = self.even_input(((left + right) / 2).flatten(1))
        odd = self.odd_input(((left - right) / 2).flatten(1))
        if self.gated:
            signed = odd.tanh()
            even_features = F.gelu(even) * (1 + signed.square())
            odd_features = signed * even.sigmoid()
        else:
            first, second = F.gelu(even + odd), F.gelu(even - odd)
            even_features, odd_features = (first + second) / 2, (first - second) / 2
        output = torch.cat((self.even_output(even_features), self.odd_output(odd_features)), dim=-1)
        return output[:, self.output_order]


class BatchedFoldedReflectionStudent(nn.Module):
    def __init__(self, folded):
        super().__init__()
        with torch.no_grad():
            self.register_buffer("input_weights", torch.stack((folded.input_even.T, folded.input_odd.T)))
            self.register_buffer("input_bias", torch.stack((folded.input_bias, torch.zeros_like(folded.input_bias))))
            self.even_count = len(folded.output_even)
            self.odd_count = len(folded.output_odd)
            width = max(self.even_count, self.odd_count)
            self.register_buffer("output_weights", torch.stack((
                F.pad(folded.output_even.T, (0, width - self.even_count)),
                F.pad(folded.output_odd.T, (0, width - self.odd_count)),
            )))
            bias = torch.stack((F.pad(folded.output_even_bias, (0, width - self.even_count)),
                                folded.output_even_bias.new_zeros((width,))))
            self.register_buffer("output_bias", bias)
            self.register_buffer("output_order", folded.output_order.detach().clone())

    def forward(self, images):
        left, right = images[..., :16], images[..., 16:].flip(-1)
        pixels = torch.stack((left + right, left - right)).flatten(2) / 2
        even, odd = torch.bmm(pixels, self.input_weights) + self.input_bias[:, None]
        first, second = F.gelu(torch.stack((even + odd, even - odd)))
        hidden = torch.stack((first + second, first - second)) / 2
        output = torch.bmm(hidden, self.output_weights) + self.output_bias[:, None]
        return torch.cat((output[0, :, :self.even_count], output[1, :, :self.odd_count]), dim=-1)[:, self.output_order]


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
