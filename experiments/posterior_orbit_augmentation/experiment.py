"""Generate plausible reflection orbits and infer the labels of generated views."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import GTSRB

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from orbit_spectral_prior.experiment import batches
from reflection_evidence.experiment import (TASKS, discover_basis, embed, fit_student, fit_teacher,
                                             predict, preprocess, prior_kernels)


class PoseGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(3, 12, 5, stride=2, padding=2), nn.GELU(),
                                     nn.Conv2d(12, 16, 3, stride=2, padding=1), nn.GELU(),
                                     nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(64, 6))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x, noise):
        mean, log_std = self.network(x).chunk(2, dim=1)
        log_std = log_std.clamp(-2, 1)
        sampled = mean + log_std.exp() * noise
        kl = (mean.square() + (2 * log_std).exp() - 1 - 2 * log_std).mean() / 2
        return sampled, kl


def warp(images, pose):
    angle, dx, dy = pose.tanh().unbind(dim=1)
    angle = angle * math.radians(12)
    dx, dy = dx * 6 / images.shape[-1], dy * 6 / images.shape[-2]
    c, s = angle.cos(), angle.sin()
    matrix = torch.stack((c, -s, dx, s, c, dy), dim=1).reshape(-1, 2, 3)
    grid = F.affine_grid(matrix, images.shape, align_corners=False)
    return F.grid_sample(images, grid, align_corners=False, padding_mode="border")


def train_generator(x, teacher, basis, args, seed):
    torch.manual_seed(seed)
    teacher.requires_grad_(False)
    generator = PoseGenerator().to(x.device)
    optimizer = torch.optim.AdamW(generator.parameters(), lr=0.003)
    rng = torch.Generator().manual_seed(seed)
    sampling = torch.Generator(device=x.device).manual_seed(seed + 100)
    for _ in range(args.generator_epochs):
        for ids in batches(len(x), args.batch_size, rng):
            mirrored = x[ids].flip(-1)
            with torch.no_grad():
                expected = teacher.encode(mirrored) @ basis
            noise = torch.randn(len(ids), 3, generator=sampling, device=x.device)
            pose, kl = generator(mirrored, noise)
            optimizer.zero_grad(set_to_none=True)
            difference = teacher.encode(warp(mirrored, pose)) @ basis - expected
            loss = difference.square().mean() + args.kl_weight * kl
            loss.backward()
            optimizer.step()
    return generator.eval()


@torch.inference_mode()
def train_views(student, generator, train, args, device, seed):
    result = {}
    rng = torch.Generator(device=device).manual_seed(seed)
    for label, samples in train.items():
        chunks = {name: [] for name in ("clean", "exact", "generated", "shuffled", "base", "pose", "base_pose")}
        for batch in samples.split(args.batch_size):
            original = batch.to(device, dtype=torch.float32) / 255
            mirrored = original.flip(-1)
            repeated = mirrored.repeat(args.views, 1, 1, 1)
            noise = torch.randn(len(repeated), 3, device=device, generator=rng)
            poses, _ = generator(repeated, noise)
            chunks["clean"].append(student(original).cpu().numpy())
            chunks["exact"].append(student(mirrored).cpu().numpy()[None])
            permutation = torch.randperm(len(poses), device=device, generator=rng)
            for key, params in (("generated", poses), ("shuffled", poses[permutation]), ("base", noise)):
                array = student(warp(repeated, params)).reshape(args.views, len(batch), -1)
                chunks[key].append(array.cpu().numpy())
            chunks["pose"].append(poses.tanh().abs().mean(dim=0).cpu().numpy())
            chunks["base_pose"].append(noise.tanh().abs().mean(dim=0).cpu().numpy())
        result[label] = {"clean": np.concatenate(chunks["clean"]),
                         "exact": np.concatenate(chunks["exact"], axis=1),
                         "generated": np.concatenate(chunks["generated"], axis=1),
                         "shuffled": np.concatenate(chunks["shuffled"], axis=1),
                         "base": np.concatenate(chunks["base"], axis=1),
                         "pose": np.mean(chunks["pose"], axis=0).tolist(),
                         "base_pose": np.mean(chunks["base_pose"], axis=0).tolist()}
    return result


def mirror_predict(kernel, clean, reflected, labels, sign, test, noise, weight=1.0):
    views = len(reflected)
    scale = np.sqrt(weight / views)
    inputs = np.concatenate((clean, reflected.reshape(-1, clean.shape[-1]) * scale))
    values = np.concatenate((labels, np.tile(labels * (sign * scale), views)))
    return predict(kernel, inputs, values, test, noise)[0]


def mode_predictions(parity_models, parity, iso, clean, labels, proposals, query, args):
    prior, centered_weights = predict(parity_models, clean, labels, query, args.noise,
                                      center_evidence=True)
    raw_prior, _ = predict(parity_models, clean, labels, query, args.noise)
    predictions = {"binary_mode_prior": raw_prior, "centered_mode_prior": prior}
    for name in ("generated", "exact"):
        proposals_by_mode = [
            mirror_predict(parity_models[i:i + 1], clean, proposals[name], labels,
                           sign, query, args.noise)
            for i, sign in enumerate((1, -1))
        ]
        predictions[f"centered_mode_{name}"] = centered_weights @ proposals_by_mode
        if name == "generated":
            predictions["raw_mode_generated"] = parity @ proposals_by_mode
    theta = float(parity[0] - parity[1])
    predictions["confidence_generated"] = mirror_predict(
        iso, clean, proposals["generated"], labels, theta, query, args.noise, weight=theta ** 2
    )
    return predictions, float(centered_weights[0])


def task_result(pair, sign, train, test, tracks, kernels, args, rng):
    originals = np.concatenate([test[label][0] for label in pair])
    mirrors = np.concatenate([test[label][1] for label in pair])
    test_features = np.concatenate((originals, mirrors))
    test_labels = np.concatenate([np.full(len(test[label][0]), value)
                                  for label, value in zip(pair, (-1, 1))])
    iso, adaptive = kernels["isotropic"], kernels["adaptive"]
    parity_models = np.concatenate((kernels["fixed_even"], kernels["fixed_odd"]))
    trajectory = {str(n): {"metrics": [], "posterior_same": [], "centered_same": []}
                  for n in args.shots}
    for _ in range(args.episodes):
        selected = []
        for label in pair:
            groups = tracks[label]
            indices = rng.choice(len(groups), max(args.shots), replace=False)
            selected.append(np.array([rng.choice(groups[i]) for i in indices]))
        for n in args.shots:
            row = trajectory[str(n)]
            labels = np.repeat([-1, 1], n)
            clean = np.concatenate([train[label]["clean"][ids[:n]]
                                    for label, ids in zip(pair, selected)])
            proposals = {key: np.concatenate([train[label][key][:, ids[:n]]
                                              for label, ids in zip(pair, selected)], axis=1)
                         for key in ("generated", "shuffled", "exact", "base")}
            _, parity = predict(parity_models, clean, labels, clean[:1], args.noise)
            theta = float(parity[0] - parity[1])
            row["posterior_same"].append(float(parity[0]))
            predictions = {"no_augmentation": predict(iso, clean, labels, test_features, args.noise)[0],
                           "adaptive_prior": predict(adaptive, clean, labels, test_features, args.noise)[0]}
            for method, proposal, label_sign in (
                ("generated_same", "generated", 1),
                ("generated_opposite", "generated", -1),
                ("generated_posterior", "generated", theta),
                ("generated_oracle", "generated", sign),
                ("random_pose_posterior", "base", theta),
                ("shuffled_pose_posterior", "shuffled", theta),
                ("exact_mirror_posterior", "exact", theta),
            ):
                predictions[method] = mirror_predict(iso, clean, proposals[proposal], labels,
                                                     label_sign, test_features, args.noise)
            if args.mode_conditioned:
                extra, centered_probability = mode_predictions(
                    parity_models, parity, iso, clean, labels, proposals, test_features, args
                )
                predictions.update(extra)
                row["centered_same"].append(centered_probability)
            metrics = {}
            for method, scores in predictions.items():
                clean_pred, mirror_pred = np.where(scores >= 0, 1, -1).reshape(2, -1)
                metrics[method] = {
                    name: float((correct[test_labels == -1].mean() +
                                 correct[test_labels == 1].mean()) / 2)
                    for name, correct in (
                        ("clean_accuracy", clean_pred == test_labels),
                        ("mirror_accuracy", mirror_pred == sign * test_labels),
                        ("parity_agreement", mirror_pred == sign * clean_pred),
                    )
                }
            row["metrics"].append(metrics)
    for row in trajectory.values():
        episodes = row["metrics"]
        row["metrics"] = {
            method: {metric: float(np.mean([ep[method][metric] for ep in episodes]))
                     for metric in metrics}
            for method, metrics in episodes[0].items()
        }
        row["posterior_same"] = float(np.mean(row["posterior_same"]))
        if args.mode_conditioned:
            row["centered_same"] = float(np.mean(row["centered_same"]))
        else:
            del row["centered_same"]
    return trajectory


def run(data, args, device, seed):
    rng = np.random.default_rng(seed)
    (source, labels), (probe, probe_labels), train, test, tracks = data
    source = source.to(device, dtype=torch.float32) / 255
    probe = probe.to(device, dtype=torch.float32) / 255
    teacher = fit_teacher(source, labels.to(device), args, seed)
    mean, basis, energies = discover_basis(teacher, probe, args)
    student = fit_student(source, teacher, basis, mean, args, seed + 100)
    generator = train_generator(source, teacher, basis, args, seed + 200)
    train = train_views(student, generator, train, args, device, seed + 300)
    test = embed(student, test, args, device)
    kernels = prior_kernels(energies.cpu().numpy(), rng)
    with torch.no_grad():
        probe_predictions = torch.cat([teacher(batch).argmax(dim=1).cpu()
                                       for batch in probe.split(args.batch_size)])
        accuracy = probe_predictions.eq(probe_labels).float().mean().item()
    scores = {
        family: {"-".join(map(str, pair)): task_result(
            pair, 1 if family == "invariant" else -1, train, test, tracks, kernels, args, rng
        ) for pair in pairs}
        for family, pairs in TASKS.items()
    }
    print(seed, "teacher probe", round(accuracy, 4), "basis width", len(energies), flush=True)
    return {"seed": seed, "teacher_probe_accuracy": accuracy, "spectral_energy": energies.tolist(),
            "generator_mean_abs_pose": np.mean([x["pose"] for x in train.values()], axis=0).tolist(),
            "base_mean_abs_pose": np.mean([x["base_pose"] for x in train.values()], axis=0).tolist(),
            "tasks": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/tmp/ibt-gtsrb-data"))
    parser.add_argument("--output", type=Path, default=HERE / "results/metrics.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4, 5])
    parser.add_argument("--teacher-epochs", type=int, default=30)
    parser.add_argument("--student-epochs", type=int, default=24)
    parser.add_argument("--generator-epochs", type=int, default=12)
    parser.add_argument("--source-examples", type=int, default=400)
    parser.add_argument("--probe-examples", type=int, default=65)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--kl-weight", type=float, default=0.5)
    parser.add_argument("--mode-conditioned", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    data = preprocess(GTSRB(args.data, split="train", download=True),
                      GTSRB(args.data, split="test", download=True), args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
               "torch_version": torch.__version__, "target_tasks": TASKS, "runs": []}
    started = time.monotonic()
    for seed in args.seeds:
        results["runs"].append(run(data, args, device, seed))
        results["elapsed_seconds"] = time.monotonic() - started
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(results, indent=2) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
