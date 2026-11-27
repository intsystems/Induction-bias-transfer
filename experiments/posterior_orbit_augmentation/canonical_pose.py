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
STEPS = (0., .25, .5, .75, 1.)
POSTERIOR_STEPS = (0., .5, 1.)


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


def parity_energy(values, odd):
    return torch.stack((values[..., ~odd].square().mean(-1), values[..., odd].square().mean(-1)), dim=-1)


def candidate_views(observed, model, teacher, basis, odd):
    """Describe the corrections available for an image before seeing its clean view."""
    location, log_std = parameters(model, observed)
    bounded = location.clamp(-2, 2)
    views = torch.cat([observed, *(pose_warp(observed, bounded * step) for step in STEPS[1:])])
    features = (teacher.encode(views) @ basis).reshape(len(STEPS), len(observed), -1)
    delta = parity_energy(features[1:] - features[:1], odd)
    pose_features = torch.cat((location.abs(), log_std, location.abs() * (-log_std).exp()), dim=1)
    response = torch.cat((delta.transpose(0, 1).flatten(1), parity_energy(features[0], odd)), dim=1)
    return views, features, pose_features, torch.log1p(response)


def fit_risk(features, losses, ridge):
    mean = features.mean(0)
    scale = features.std(0, correction=0).clamp_min(1e-6)
    predictors = ((features - mean) / scale).clamp(-6, 6)
    gram = predictors.T @ predictors / len(predictors)
    regularized = gram + ridge * torch.eye(len(gram), device=gram.device, dtype=gram.dtype)
    intercept = losses.mean(0)
    coefficients = torch.linalg.solve(regularized, predictors.T @ (losses - intercept) / len(predictors))
    return mean, scale, intercept, coefficients


def select_step(calibration, features):
    mean, scale, intercept, coefficients = calibration
    normalized = ((features - mean) / scale).clamp(-6, 6)
    return (intercept + normalized @ coefficients).argmin(1)


@torch.inference_mode()
def calibrate_selectors(probe, model, teacher, basis, odd, args, seed):
    random = torch.Generator(device=probe.device).manual_seed(seed)
    levels = probe.new_tensor(args.strengths)
    risk, poses, responses = [], [], []
    for samples in probe.split(args.batch_size):
        clean = torch.cat((samples, samples.flip(-1)))
        scale = levels[torch.randint(len(levels), (len(clean),), generator=random, device=probe.device)]
        pose = (2 * torch.rand(len(clean), 3, generator=random, device=probe.device) - 1) * scale[:, None]
        _, features, pose_features, response = candidate_views(pose_warp(clean, pose), model, teacher, basis, odd)
        expected = teacher.encode(clean) @ basis
        losses = parity_energy(features - expected, odd).mean(dim=-1)
        risk.append((losses - losses[:1]).T)
        poses.append(pose_features)
        responses.append(response)
    risk, poses, responses = (torch.cat(values).double() for values in (risk, poses, responses))
    fixed = int(risk.mean(0).argmin())
    return fixed, fit_risk(poses, risk, args.selector_ridge), fit_risk(
        torch.cat((poses, responses), dim=1), risk, args.selector_ridge)


@torch.inference_mode()
def selective_embeddings(student, model, teacher, basis, odd, calibration, images, args, device, seed):
    fixed, pose_model, semantic_model = calibration
    rng = torch.Generator(device=device).manual_seed(seed)
    output, policies = {}, {}
    steps = torch.tensor(STEPS, device=device)
    for label, samples in images.items():
        order = torch.randperm(len(samples), generator=rng, device=device)[:args.test_per_class]
        chunks = {str(s): [] for s in args.strengths}
        selections = {str(s): [] for s in args.strengths}
        for batch in samples[order.cpu()].split(args.batch_size):
            batch = batch.to(device, dtype=torch.float32) / 255
            clean = torch.cat((batch, batch.flip(-1)))
            unit_pose = 2 * torch.rand(len(clean), 3, device=device, generator=rng) - 1
            rows = torch.arange(len(clean), device=device)
            for strength in args.strengths:
                pose = unit_pose * strength
                observed = pose_warp(clean, pose)
                views, _, pose_features, response = candidate_views(observed, model, teacher, basis, odd)
                features = student(views).reshape(len(STEPS), len(clean), -1)
                pose_step = select_step(pose_model, pose_features.double())
                semantic_step = select_step(semantic_model, torch.cat((pose_features, response), dim=1).double())
                methods = {"raw": features[0], "pose_point": features[-1], "half": features[2],
                           "source_fixed": features[fixed], "pose_selector": features[pose_step, rows],
                           "teacher_selector": features[semantic_step, rows],
                           "oracle": student(pose_warp(observed, inverse_pose(pose)))}
                chunks[str(strength)].append(
                    {key: values.reshape(2, len(batch), -1).cpu().numpy() for key, values in methods.items()})
                chosen = torch.stack((pose_step, semantic_step))
                stats = torch.stack((steps[chosen].mean(dim=1), (chosen == 0).float().mean(dim=1)))
                selections[str(strength)].append((len(batch), stats.cpu().numpy()))
        output[label] = {s: ({k: np.concatenate([c[k] for c in values], axis=1)
                              for k in values[0]}, None) for s, values in chunks.items()}
        policies[label] = {s: np.average([x for _, x in values], axis=0, weights=[n for n, _ in values])
                           for s, values in selections.items()}
    summary = {}
    for strength in map(str, args.strengths):
        means, skipped = np.mean([v[strength] for v in policies.values()], axis=0)
        summary[strength] = {name: {"mean_step": float(step), "skipped": float(fraction)}
                             for name, step, fraction in zip(("pose_selector", "teacher_selector"),
                                                             means, skipped, strict=True)}
    return output, summary


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


@torch.inference_mode()
def global_pose_scale(probe, model, args, seed):
    rng = torch.Generator(device=probe.device).manual_seed(seed)
    levels = probe.new_tensor(args.strengths)
    variances = []
    for batch in probe.split(args.batch_size):
        samples = torch.cat((batch, batch.flip(-1)))
        levels_for_batch = levels[torch.randint(len(levels), (len(samples),), device=probe.device, generator=rng)]
        unit_pose = 2 * torch.rand(len(samples), 3, device=probe.device, generator=rng) - 1
        _, log_std = parameters(model, pose_warp(samples, unit_pose * levels_for_batch[:, None]))
        variances.append((2 * log_std).exp())
    return torch.cat(variances).mean(dim=0).sqrt()


@torch.inference_mode()
def posterior_selection_embeddings(student, model, global_scale, images, args, device, seed):
    rng = torch.Generator(device=device).manual_seed(seed)
    sampling = torch.Generator(device=device).manual_seed(seed + 1000)
    output = {}
    for label, samples in images.items():
        order = torch.randperm(len(samples), generator=rng, device=device)[:args.test_per_class]
        chunks = {str(s): [] for s in args.strengths}
        for batch in samples[order.cpu()].split(args.batch_size):
            batch = batch.to(device, dtype=torch.float32) / 255
            clean = torch.cat((batch, batch.flip(-1)))
            unit_pose = 2 * torch.rand(len(clean), 3, device=device, generator=rng) - 1
            noise = torch.randn(args.views // 2, len(clean), 3, device=device, generator=sampling)
            noise = torch.cat((noise, -noise))
            for strength in args.strengths:
                pose = strength * unit_pose
                observed = pose_warp(clean, pose)
                repeated = observed.repeat(args.views, 1, 1, 1)
                location, log_std = parameters(model, observed)
                centers = [location.clamp(-2, 2) * step for step in POSTERIOR_STEPS]
                points = torch.stack([student(observed), *(student(pose_warp(observed, c)) for c in centers[1:])],
                                     dim=1)
                features = {"centers": points, "raw": points[:, 0], "half": points[:, 1],
                            "pose_point": points[:, 2], "oracle": student(pose_warp(observed, inverse_pose(pose)))}
                for key, scale in (("conditional_draws", log_std.exp()), ("global_draws", global_scale)):
                    draws = [(center + noise * scale).clamp(-2, 2) for center in centers]
                    embeddings = [student(pose_warp(repeated, draw.flatten(0, 1)))
                                  .reshape(args.views, len(clean), -1).transpose(0, 1) for draw in draws]
                    features[key] = torch.stack(embeddings, dim=1)
                chunks[str(strength)].append({
                    key: value.reshape(2, len(batch), *value.shape[1:]).cpu().numpy()
                    for key, value in features.items()
                })
        output[label] = {s: ({k: np.concatenate([c[k] for c in values], axis=1)
                              for k in values[0]}, None) for s, values in chunks.items()}
    return output


def select_posterior_logits(choices):
    centers = choices.pop("centers")
    conditional = choices.pop("conditional_draws")
    marginal = choices.pop("global_draws")
    direction = np.where(centers >= 0, 1, -1)
    rankings = {"point_margin": np.abs(centers)}
    for key, logits in (("conditional_consensus", conditional), ("global_consensus", marginal)):
        rankings[key] = direction * logits.mean(axis=-1) - logits.std(axis=-1)
    steps = {}
    for key, scores in rankings.items():
        idx = scores.argmax(axis=-1)[..., None]
        choices[key] = np.take_along_axis(centers, idx, axis=-1).squeeze(-1)
        steps[key] = float(np.asarray(POSTERIOR_STEPS)[idx[1]].mean())
    return steps


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
                steps = select_posterior_logits(choices) if "centers" in choices else {}
                if "hidden_posterior" in embeddings:
                    choices["hidden_posterior_even"] = embeddings["hidden_posterior"] @ even
                metrics = {}
                for key, logits in choices.items():
                    clean, reflected = np.where(logits >= 0, 1, -1)
                    measures = (clean == truth, reflected == sign * truth)
                    metrics[key] = {
                        name: float((value[:n_negative].mean() + value[n_negative:].mean()) / 2)
                        for name, value in zip(("clean", "reflected"), measures, strict=True)
                    }
                    if key in steps:
                        metrics[key]["mean_step"] = steps[key]
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
    basis_probe = probe[::2] if args.selective or args.posterior_selection else probe
    mean, basis, energies = discover_basis(teacher, basis_probe, args)
    student = fit_student(source, teacher, basis, mean, args, seed + 100)
    odd = energies > .5
    pose_model = fit_canonicalizer(source, teacher, basis, odd, args, seed + 200, 0.)
    support = {label: features[0] for label, features in embed(student, train, args, device).items()}
    if args.posterior_selection:
        global_scale = global_pose_scale(probe[1::2], pose_model, args, seed + 250)
        query = posterior_selection_embeddings(student, pose_model, global_scale, test, args, device, seed + 300)
        diagnostics = {"global_pose_scale": global_scale.cpu().tolist()}
    elif args.selective:
        selection = calibrate_selectors(probe[1::2], pose_model, teacher, basis, odd, args, seed + 250)
        query, policy_stats = selective_embeddings(
            student, pose_model, teacher, basis, odd, selection, test, args, device, seed + 300)
        diagnostics = {"source_fixed_step": STEPS[selection[0]], "policy_statistics": policy_stats}
    else:
        models = {"pose": pose_model, "hidden": fit_canonicalizer(
            source, teacher, basis, odd, args, seed + 200, args.hidden_weight)}
        query = canonical_embeddings(student, models, test, args, device, seed + 300)
        diagnostics = {"pose_calibration": {strength: {
            method: {measure: np.mean([v[strength][1][method][measure] for v in query.values()], axis=0).tolist()
                     for measure in ("mse", "nll")}
            for method in models} for strength in map(str, args.strengths)}}
    rng = np.random.default_rng(seed)
    kernels = prior_kernels(energies.cpu().numpy(), rng)
    tasks = {family: {"-".join(map(str, pair)): evaluate_task(
        pair, 1 if family == "invariant" else -1, support, query, tracks, kernels, args, rng)
        for pair in pairs} for family, pairs in TASKS.items()}
    print(seed, "completed", flush=True)
    return {"seed": seed, "tasks": tasks, **diagnostics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/tmp/ibt-gtsrb-data"))
    parser.add_argument("--output", type=Path)
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
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--selective", action="store_true")
    selection.add_argument("--posterior-selection", action="store_true")
    parser.add_argument("--selector-ridge", type=float, default=.05)
    args = parser.parse_args()
    if args.output is None:
        if args.posterior_selection:
            name = "posterior_selection"
        else:
            name = "selective_pose" if args.selective else "canonical_pose"
        args.output = HERE / "results" / name / "metrics.json"
    if args.views < 2 or args.views % 2 or args.test_per_class < 1:
        parser.error("need a positive test size and an even number of at least two posterior views")
    if not all(math.isfinite(s) and s >= 0 for s in args.strengths):
        parser.error("strengths must be finite and nonnegative")
    if not math.isfinite(args.selector_ridge) or args.selector_ridge <= 0:
        parser.error("selector ridge must be finite and positive")
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
