"""Transfer a conditional distribution of inverse poses to held-out sign classes."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from experiment import (GTSRB, HERE, TASKS, PoseGenerator, batches, discover_basis, embed,
                        fit_student, fit_teacher, predict, preprocess, prior_kernels)

ANGLE = math.radians(32)
SHIFT = .24


def pose_warp(images, pose):
    angle, dx, dy = pose.unbind(dim=-1)
    cosine, sine = (angle * ANGLE).cos(), (angle * ANGLE).sin()
    matrix = torch.stack((cosine, -sine, SHIFT * dx, sine, cosine, SHIFT * dy), dim=-1)
    grid = torch.nn.functional.affine_grid(matrix.reshape(-1, 2, 3), images.shape, align_corners=False)
    return torch.nn.functional.grid_sample(images, grid, align_corners=False, padding_mode="border")


def inverse_pose(pose):
    angle, dx, dy = pose.unbind(dim=-1)
    cosine, sine = (angle * ANGLE).cos(), (angle * ANGLE).sin()
    return torch.stack((-angle, -cosine * dx - sine * dy, sine * dx - cosine * dy), dim=-1)


def parameters(model, images):
    center, log_std = model.network(images).chunk(2, dim=-1)
    return center, log_std.clamp(-3.5, .6)


def normal_nll(center, log_std, target):
    return .5 * ((target - center) * (-log_std).exp()).square() + log_std


def fit_canonicalizer(source, teacher, basis, odd, args, seed, hidden_weight):
    torch.manual_seed(seed)
    model = PoseGenerator().to(source.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    ordering = torch.Generator().manual_seed(seed)
    random = torch.Generator(device=source.device).manual_seed(seed + 100)
    teacher.requires_grad_(False)
    for _ in range(args.generator_epochs):
        for indices in batches(len(source), args.batch_size, ordering):
            clean = source[indices]
            reflected = torch.rand(len(clean), generator=random, device=clean.device) < .5
            clean = torch.where(reflected[:, None, None, None], clean.flip(-1), clean)
            pose = 2 * torch.rand(len(clean), 3, generator=random, device=clean.device) - 1
            noise = torch.randn(len(clean), 3, generator=random, device=clean.device)
            distorted = pose_warp(clean, pose)
            location, log_std = parameters(model, distorted)
            loss = normal_nll(location, log_std, inverse_pose(pose)).mean()
            if hidden_weight:
                with torch.no_grad():
                    target = teacher.encode(clean) @ basis
                sampled = (location + log_std.exp() * noise).clamp(-2, 2)
                residual = (teacher.encode(pose_warp(distorted, sampled)) @ basis - target).square()
                hidden = (residual[:, odd].mean() + residual[:, ~odd].mean()) / 2
                loss = loss + hidden_weight * hidden
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval()


@torch.inference_mode()
def canonical_embeddings(student, models, images, args, device, seed):
    rng = torch.Generator(device=device).manual_seed(seed)
    output = {}
    for label, samples in images.items():
        order = torch.randperm(len(samples), generator=rng, device=device)[:args.test_per_class]
        samples = samples[order.cpu()]
        per_strength = {str(s): [] for s in args.strengths}
        for batch in samples.split(args.batch_size):
            batch = batch.to(device, dtype=torch.float32) / 255
            clean = torch.cat((batch, batch.flip(-1)))
            unit_pose = 2 * torch.rand(len(clean), 3, device=device, generator=rng) - 1
            noise = torch.randn(args.views // 2, len(clean), 3, device=device, generator=rng)
            noise = torch.cat((noise, -noise))
            # Permute parameters only within each of the two reflection conditions.
            perm = torch.cat((torch.randperm(len(batch), device=device, generator=rng),
                              len(batch) + torch.randperm(len(batch), device=device, generator=rng)))
            for strength in args.strengths:
                pose = unit_pose * strength
                inverse = inverse_pose(pose)
                observed = pose_warp(clean, pose)
                repeated = observed.repeat(args.views, 1, 1, 1)
                features = {"raw": student(observed), "oracle": student(pose_warp(observed, inverse))}
                calibration = {}
                for key, model in models.items():
                    location, log_std = parameters(model, observed)
                    sampled = (location + log_std.exp() * noise).clamp(-2, 2)
                    features[key + "_point"] = student(pose_warp(observed, location.clamp(-2, 2)))
                    generated = student(pose_warp(repeated, sampled.flatten(0, 1)))
                    features[key + "_posterior"] = generated.reshape(args.views, len(clean), -1).mean(0)
                    if key == "hidden":
                        shuffled = student(pose_warp(repeated, sampled[:, perm].flatten(0, 1)))
                        features["hidden_shuffled"] = shuffled.reshape(args.views, len(clean), -1).mean(0)
                    calibration[key] = {
                        "mse": (location - inverse).square().mean(0).cpu().numpy(),
                        "nll": normal_nll(location, log_std, inverse).mean(0).cpu().numpy(),
                    }
                per_strength[str(strength)].append((
                    {k: f.reshape(2, len(batch), -1).cpu().numpy() for k, f in features.items()}, calibration,
                ))
        output[label] = {
            strength: ({k: np.concatenate([f[k] for f, _ in chunks], axis=1) for k in chunks[0][0]},
                       {k: {m: np.average([c[k][m] for _, c in chunks], axis=0,
                                          weights=[f["raw"].shape[1] for f, _ in chunks]).tolist()
                            for m in ("mse", "nll")} for k in models})
            for strength, chunks in per_strength.items()
        }
    return output


def evaluate_task(pair, sign, support, query, tracks, kernels, args, rng):
    parity = np.concatenate((kernels["fixed_even"], kernels["fixed_odd"]))
    batches_by_strength = {}
    for strength in map(str, args.strengths):
        a, b = (query[label][strength][0] for label in pair)
        embeddings = {key: np.concatenate((a[key], b[key]), axis=1) for key in a}
        first_count, last_count = a["raw"].shape[1], b["raw"].shape[1]
        truth = np.r_[np.full(first_count, -1), np.full(last_count, 1)]
        batches_by_strength[strength] = (embeddings, truth, first_count)
    rows = {str(n): [] for n in args.shots}
    for _ in range(args.episodes):
        ids = []
        for label in pair:
            groups = tracks[label]
            selected = rng.choice(len(groups), max(args.shots), replace=False)
            ids.append([rng.choice(groups[j]) for j in selected])
        for count in args.shots:
            f = np.concatenate([support[label][j[:count]] for label, j in zip(pair, ids, strict=True)])
            y = np.repeat([-1, 1], count)
            # Compute the posterior head once, using only the clean support.
            width = f.shape[1]
            weights, probability = predict(parity, f, y, np.eye(width), args.noise, center_evidence=True)
            even, _ = predict(kernels["fixed_even"], f, y, np.eye(width), args.noise)
            values = {"posterior_same": float(probability[0]), "strengths": {}}
            for strength, (embeddings, truth, n_negative) in batches_by_strength.items():
                choices = {name: features @ weights for name, features in embeddings.items()}
                choices["hidden_posterior_even"] = embeddings["hidden_posterior"] @ even
                metrics = {}
                for key, logits in choices.items():
                    clean, reflected = np.where(logits >= 0, 1, -1)
                    measures = (clean == truth, reflected == sign * truth)
                    metrics[key] = {
                        name: float((value[:n_negative].mean() + value[n_negative:].mean()) / 2)
                        for name, value in zip(("clean", "reflected"), measures, strict=True)
                    }
                values["strengths"][strength] = metrics
            rows[str(count)].append(values)
    return {
        count: {
            "posterior_same": float(np.mean([ep["posterior_same"] for ep in episodes])),
            "strengths": {strength: {key: {metric: float(np.mean([
                ep["strengths"][strength][key][metric] for ep in episodes]))
                for metric in measures} for key, measures in methods.items()}
                for strength, methods in episodes[0]["strengths"].items()},
        } for count, episodes in rows.items()
    }


def run(data, args, device, seed):
    (source, labels), (probe, _), train, test, tracks = data
    source = source.to(device, dtype=torch.float32) / 255
    probe = probe.to(device, dtype=torch.float32) / 255
    teacher = fit_teacher(source, labels.to(device), args, seed)
    mean, basis, energies = discover_basis(teacher, probe, args)
    student = fit_student(source, teacher, basis, mean, args, seed + 100)
    odd = energies > .5
    models = {k: fit_canonicalizer(source, teacher, basis, odd, args, seed + 200, weight)
              for k, weight in (("pose", 0.), ("hidden", args.hidden_weight))}
    support = {label: features[0] for label, features in embed(student, train, args, device).items()}
    query = canonical_embeddings(student, models, test, args, device, seed + 300)
    rng = np.random.default_rng(seed)
    kernels = prior_kernels(energies.cpu().numpy(), rng)
    tasks = {family: {"-".join(map(str, pair)): evaluate_task(
        pair, 1 if family == "invariant" else -1, support, query, tracks, kernels, args, rng)
        for pair in pairs} for family, pairs in TASKS.items()}
    calibration = {strength: {
        method: {measure: np.mean([v[strength][1][method][measure] for v in query.values()], axis=0).tolist()
                 for measure in ("mse", "nll")}
        for method in models} for strength in map(str, args.strengths)}
    print(seed, "completed", flush=True)
    return {"seed": seed, "tasks": tasks, "pose_calibration": calibration}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/tmp/ibt-gtsrb-data"))
    parser.add_argument("--output", type=Path, default=HERE / "results/canonical_pose/metrics.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(81, 93)))
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4, 5])
    parser.add_argument("--strengths", type=float, nargs="+", default=[0., .5, 1., 1.5])
    parser.add_argument("--teacher-epochs", type=int, default=22)
    parser.add_argument("--student-epochs", type=int, default=18)
    parser.add_argument("--generator-epochs", type=int, default=12)
    parser.add_argument("--source-examples", type=int, default=300)
    parser.add_argument("--probe-examples", type=int, default=65)
    parser.add_argument("--test-per-class", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--views", type=int, default=6)
    parser.add_argument("--noise", type=float, default=.5)
    parser.add_argument("--hidden-weight", type=float, default=.5)
    args = parser.parse_args()
    if args.views < 2 or args.views % 2 or args.test_per_class < 1:
        parser.error("need a positive test size and an even number of at least two posterior views")
    if not all(math.isfinite(s) and s >= 0 for s in args.strengths):
        parser.error("strengths must be finite and nonnegative")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    data = preprocess(GTSRB(args.data, split="train", download=True),
                      GTSRB(args.data, split="test", download=True), args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
              "torch_version": torch.__version__, "target_tasks": TASKS, "runs": []}
    start = time.monotonic()
    for seed in args.seeds:
        output["runs"].append(run(data, args, device, seed))
        output["elapsed_seconds"] = time.monotonic() - start
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
