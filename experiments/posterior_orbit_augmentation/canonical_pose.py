"""Transfer a conditional distribution of inverse poses to held-out sign classes."""

import argparse
import json
import math
import time
from collections import defaultdict
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from experiment import (GTSRB, HERE, TASKS, PoseGenerator, batches, discover_basis, embed,
                        fit_student, fit_teacher, predict, preprocess, prior_kernels)
from reflection_evidence.experiment import MIXTURE, paths_by_class

ANGLE = math.radians(32)
SHIFT = .24
STEPS = (0., .25, .5, .75, 1.)
POSTERIOR_STEPS = (0., .5, 1.)
MIXED_LEVELS = (0., .5, 1.)
ANGLE_GATE_MODELS = ("gate_fixed", "gate_adaptive", "gate_shuffled")
TRACK_PAIR_BUDGETS = (2, 8, 32)


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


def compose_pose(first, second):
    angle, dx, dy = first.unbind(dim=-1)
    turn, tx, ty = second.unbind(dim=-1)
    cosine, sine = (angle * ANGLE).cos(), (angle * ANGLE).sin()
    return torch.stack((angle + turn, dx + cosine * tx - sine * ty, dy + sine * tx + cosine * ty), dim=-1)


def reflect_pose(pose):
    return pose * pose.new_tensor((-1, -1, 1))


def parameters(model, images):
    center, log_std = model.network(images).chunk(2, dim=-1)
    return center, log_std.clamp(-3.5, .6)


def normal_nll(center, log_std, target):
    return .5 * ((target - center) * (-log_std).exp()).square() + log_std


def angle_gate_features(first, second):
    first_center, first_log_std = first
    second_center, second_log_std = second
    angle = first_center[:, 0]
    difference = second_center[:, 0] - angle
    return torch.stack((angle.abs(), difference.abs(), first_log_std[:, 0], second_log_std[:, 0]), dim=1)


@torch.inference_mode()
def calibrate_angle_gate(probe, models, args, seed):
    rng = torch.Generator(device=probe.device).manual_seed(seed)
    levels = probe.new_tensor(MIXED_LEVELS)
    samples = []
    for batch in probe.split(args.batch_size):
        clean = torch.cat((batch, batch.flip(-1)))
        strength = levels[torch.randint(len(levels), (len(clean),), device=probe.device, generator=rng)]
        pose = (2 * torch.rand(len(clean), 3, device=probe.device, generator=rng) - 1) * strength[:, None]
        observed = pose_warp(clean, pose)
        first, second = (parameters(models[name], observed) for name in ("pose", "angle_same"))
        samples.append((angle_gate_features(first, second), second[0][:, 0] - first[0][:, 0],
                        -pose[:, 0] - first[0][:, 0]))
    features, delta, target = (torch.cat(chunks) for chunks in zip(*samples, strict=True))
    mean = features.mean(0)
    scale = features.std(0, unbiased=False).clamp_min(1e-6)
    design = torch.cat((torch.ones_like(delta[:, None]), (features - mean) / scale), dim=1) * delta[:, None]
    ridge = torch.diag(design.new_tensor((1e-6, .01, .01, .01, .01)))
    coefficients = torch.linalg.solve(design.T @ design / len(design) + ridge,
                                     design.T @ target / len(design))
    fixed = ((delta @ target) / (delta @ delta).clamp_min(1e-8)).clamp(0, 1)
    return mean, scale, coefficients, fixed


def mix_angle_predictions(first, second, weight):
    first_center, first_log_std = first
    second_center, second_log_std = second
    difference = second_center[:, 0] - first_center[:, 0]
    angle = first_center[:, 0] + weight * difference
    variance = ((1 - weight) * (2 * first_log_std[:, 0]).exp() + weight * (2 * second_log_std[:, 0]).exp()
                + weight * (1 - weight) * difference.square())
    return (torch.cat((angle[:, None], first_center[:, 1:]), dim=1),
            torch.cat((variance.log()[:, None] / 2, first_log_std[:, 1:]), dim=1))


def fit_canonicalizer(source, teacher, basis, odd, args, seed, hidden_weight, *, partners=None,
                      pair_weights=(.25, .25)):
    torch.manual_seed(seed)
    model = PoseGenerator().to(source.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    ordering = torch.Generator().manual_seed(seed)
    random = torch.Generator(device=source.device).manual_seed(seed + 100)
    teacher.requires_grad_(False)
    for epoch in range(args.generator_epochs):
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
            if partners is not None:
                partner_indices = partners[indices]
                eligible = partner_indices >= 0
                if eligible.any():
                    natural = source[torch.cat((indices[eligible], partner_indices[eligible]))]
                    if epoch % 2:
                        natural = natural.flip(-1)
                    center, _ = parameters(model, natural)
                    projected = teacher.encode(pose_warp(natural, center.clamp(-2, 2))) @ basis
                    first, second = projected.chunk(2)
                    residual = (first - second).square()
                    pair_loss = pair_weights[0] * residual[:, ~odd].mean() + pair_weights[1] * residual[:, odd].mean()
                    loss = loss + pair_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval()


def fit_paired_canonicalizer(source, args, seed, pairing, *, angle_bias=False):
    torch.manual_seed(seed)
    model = PoseGenerator().to(source.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    ordering = torch.Generator().manual_seed(seed)
    rng = torch.Generator(device=source.device).manual_seed(seed + 100)
    for _ in range(args.generator_epochs):
        for indices in batches(len(source), args.batch_size, ordering):
            clean = source[indices]
            reflected = torch.rand(len(clean), device=clean.device, generator=rng) < .5
            clean = torch.where(reflected[:, None, None, None], clean.flip(-1), clean)
            pose = 2 * torch.rand(2, len(clean), 3, device=clean.device, generator=rng) - 1
            if angle_bias:
                pose[1, :, 0] = -pose[0, :, 0]
            images = pose_warp(clean.repeat(2, 1, 1, 1), pose.flatten(0, 1))
            center, log_std = parameters(model, images)
            target = inverse_pose(pose)
            loss = normal_nll(center, log_std, target.flatten(0, 1)).mean()
            if pairing != "absolute":
                first, second = center.reshape(2, len(clean), 3)
                y_first, y_second = target
                if pairing == "shuffled":
                    second = second.roll(1, dims=0)
                    y_second = y_second.roll(1, dims=0)
                if angle_bias:
                    first_error = first[:, 0] - y_first[:, 0]
                    second_error = second[:, 0] - y_second[:, 0]
                    if pairing == "point":
                        loss = loss + .25 * (first_error.square() + second_error.square()).mean()
                    else:
                        loss = loss + .25 * (first_error + second_error).square().mean()
                else:
                    estimated = compose_pose(first, inverse_pose(second))
                    relative = compose_pose(y_first, inverse_pose(y_second))
                    loss = loss + .5 * (estimated - relative).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval()


def mixture_parameters(model, images):
    output = model.network(images)
    return output[:, :3], output[:, 3:6].clamp(-3.5, .6), output[:, 6]


def fit_mixture_canonicalizer(source, args, seed, *, hurdle):
    torch.manual_seed(seed)
    model = PoseGenerator()
    model.network[-1] = torch.nn.Linear(model.network[-1].in_features, 7)
    torch.nn.init.zeros_(model.network[-1].weight)
    torch.nn.init.zeros_(model.network[-1].bias)
    model.to(source.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    ordering = torch.Generator().manual_seed(seed)
    rng = torch.Generator(device=source.device).manual_seed(seed + 100)
    levels = source.new_tensor(MIXED_LEVELS)
    for _ in range(args.generator_epochs):
        for indices in batches(len(source), args.batch_size, ordering):
            clean = source[indices]
            reflected = torch.rand(len(clean), generator=rng, device=source.device) < .5
            clean = torch.where(reflected[:, None, None, None], clean.flip(-1), clean)
            scale = levels[torch.randint(len(levels), (len(clean),), device=source.device, generator=rng)]
            no_warp = scale == 0
            pose = (2 * torch.rand(len(clean), 3, generator=rng, device=source.device) - 1) * scale[:, None]
            observed = torch.where(no_warp[:, None, None, None], clean, pose_warp(clean, pose))
            center, log_std, logit = mixture_parameters(model, observed)
            pose_loss = normal_nll(center, log_std, inverse_pose(pose)).mean(dim=-1)
            if hurdle:
                active = ~no_warp
                pose_loss = (pose_loss * active).sum() / active.sum().clamp_min(1)
            else:
                pose_loss = pose_loss.mean()
            identity_loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, no_warp.float())
            optimizer.zero_grad(set_to_none=True)
            (pose_loss + identity_loss).backward()
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
    return calibrated_prediction(calibration, features).argmin(1)


def calibrated_prediction(calibration, features):
    mean, scale, intercept, coefficients = calibration
    normalized = ((features - mean) / scale).clamp(-6, 6)
    return intercept + normalized @ coefficients


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
def canonical_embeddings(student, models, images, args, device, seed, *, angle_gate=None):
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
                predictions = {key: parameters(model, observed) for key, model in models.items()}
                if angle_gate is not None:
                    first, second = predictions["pose"], predictions["angle_same"]
                    mean, scale, coefficients, fixed = angle_gate
                    weight = (coefficients[0] + ((angle_gate_features(first, second) - mean) / scale)
                              @ coefficients[1:]).clamp(0, 1)
                    predictions["angle_same"] = mix_angle_predictions(first, second, torch.ones_like(weight))
                    for name, amount in zip(ANGLE_GATE_MODELS, (fixed, weight, weight[perm]), strict=True):
                        predictions[name] = mix_angle_predictions(first, second, amount)
                for key, (location, log_std) in predictions.items():
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
                            for m in ("mse", "nll")} for k in chunks[0][1]})
            for strength, chunks in per_strength.items()
        }
    return output


def split_natural_tracks(tracks, shots, limit, seed):
    rng = np.random.default_rng(seed)
    support, query = {}, {}
    for label, groups in tracks.items():
        count = max(max(shots), math.ceil(.6 * len(groups)))
        if count >= len(groups):
            raise ValueError(f"class {label} has too few tracks for {max(shots)} labels per class")
        order = rng.permutation(len(groups))
        support[label] = [groups[i] for i in order[:count]]
        query[label] = [groups[i] for i in order[count:count + limit]]
    return support, query


def natural_source_partners(paths):
    by_class = defaultdict(lambda: defaultdict(list))
    for i, path in enumerate(paths):
        track, frame = path.stem.rsplit("_", 1)
        by_class[path.parent][track].append((int(frame), i))
    same_track = np.full(len(paths), -1, dtype=np.int64)
    same_class = np.full(len(paths), -1, dtype=np.int64)
    for tracks in by_class.values():
        groups = [np.array([i for _, i in sorted(frames)]) for frames in tracks.values() if len(frames) >= 2]
        if not groups:
            continue
        shift = max(map(len, groups))
        combined = np.concatenate(groups)
        if 2 * shift > len(combined):
            continue
        for group in groups:
            same_track[group] = np.roll(group, len(group) // 2)
        same_class[combined] = np.roll(combined, shift)
    return {"same_track": torch.from_numpy(same_track), "same_class": torch.from_numpy(same_class)}


@torch.no_grad()
def teacher_pair_differences(teacher, source, partners, batch_size):
    latent = torch.cat([teacher.encode(batch) for batch in source.split(batch_size)])
    indices = torch.where(partners >= 0)[0]
    return latent[indices] - latent[partners[indices]]


@torch.no_grad()
def control_teacher_basis(teacher, basis, odd, source, partners, args, seed):
    rng = torch.Generator(device=source.device).manual_seed(seed)
    rotated = []
    for block in basis.chunk(2, dim=0):
        shape = (len(block), len(block))
        matrix = torch.randn(shape, device=source.device, dtype=basis.dtype, generator=rng)
        rotated.append(torch.linalg.qr(matrix).Q @ block)
    control = torch.cat(rotated)
    difference = teacher_pair_differences(teacher, source, partners, args.batch_size)
    original_error = (difference @ basis).square().mean(dim=0)
    control_error = (difference @ control).square().mean(dim=0)
    diagnostics = {}
    for name, block in (("even", ~odd), ("odd", odd)):
        original = original_error[block].mean()
        unscaled = control_error[block].mean()
        scale = (original / unscaled.clamp_min(1e-12)).sqrt()
        control[:, block] *= scale
        diagnostics[name] = {"original_mse": original.item(), "scaled_mse": (unscaled * scale.square()).item(),
                             "scale": scale.item()}
    return control, diagnostics


@torch.no_grad()
def calibrate_teacher_parity(teacher, basis, odd, source, partners, args):
    difference = teacher_pair_differences(teacher, source, partners, args.batch_size) @ basis
    energy = torch.stack((difference[:, ~odd].square().mean(), difference[:, odd].square().mean()))
    matched = (.25 * energy.sum() / energy.clamp_min(1e-12)).tolist()
    return {"even_teacher": (matched[0], 0.), "odd_teacher": (0., matched[1])}, energy.tolist()


@torch.no_grad()
def natural_track_head_priors(teacher, basis, odd, source, partners, args, seed, *,
                              source_paths=None, student=None, source_labels=None):
    rng = np.random.default_rng(seed)
    names = ["head_same_track", "head_same_class", "head_rotated"]
    width = basis.shape[1]
    odd = odd.cpu().numpy()
    eligible = np.flatnonzero(partners["same_track"].numpy() >= 0)
    latent = torch.cat([teacher.encode(batch) @ basis for batch in source.split(args.batch_size)]).cpu().numpy()
    residuals = [latent[eligible] - latent[partners[name].numpy()[eligible]]
                 for name in ("same_track", "same_class")]
    centroids = None
    if source_paths is not None:
        centroids = natural_centroid_differences(latent, source_paths, eligible)
        names.extend(("head_track_means", "head_split_means"))
    student_residuals = []
    if student is not None:
        student_latent = torch.cat([student(batch) for batch in source.split(args.batch_size)]).cpu().numpy()
        student_residuals = [student_latent[eligible] - student_latent[partners[name].numpy()[eligible]]
                             for name in ("same_track", "same_class")]
        names.extend(("head_student_track", "head_student_class"))
    budget_residuals = {}
    if source_labels is not None:
        labels = source_labels.numpy()[eligible]
        sampling = np.random.default_rng(seed + 1000)
        order = [sampling.permutation(np.flatnonzero(labels == label)) for label in np.unique(labels)]
        for budget in TRACK_PAIR_BUDGETS:
            subset = np.concatenate([index[:budget] for index in order])
            budget_residuals[f"head_pairs_{budget}"] = residuals[1][subset]
        names.extend(budget_residuals)
    priors = {name: np.zeros((2, width, width)) for name in names}
    for block, proportions in ((~odd, (1-MIXTURE[0], 1-MIXTURE[-1])), (odd, (MIXTURE[0], MIXTURE[-1]))):
        moments = []
        for residual in residuals:
            part = residual[:, block].astype(np.float64)
            moments.append(part.T @ part / len(part))
        energies, same = np.linalg.eigh(moments[0])
        _, other = np.linalg.eigh(moments[1])
        rotated, _ = np.linalg.qr(rng.normal(size=same.shape))
        bases = [same, other, rotated]
        if centroids is not None:
            whole, first, second = [member[:, block].astype(np.float64) for member in centroids]
            cross = first.T @ second
            bases.extend(np.linalg.eigh(moment)[1] for moment in (whole.T @ whole, (cross + cross.T) / 2))
        for residual in (*student_residuals, *budget_residuals.values()):
            part = residual[:, block].astype(np.float64)
            bases.append(np.linalg.eigh(part.T @ part)[1])
        spectrum = 1 / (.25 + np.maximum(energies, 0) / max(float(energies.mean()), 1e-12))
        spectrum /= spectrum.sum()
        index = np.ix_(block, block)
        for name, axes in zip(names, bases, strict=True):
            block_covariance = (axes * spectrum) @ axes.T
            for mode, proportion in enumerate(proportions):
                priors[name][mode][index] = proportion * block_covariance
    return priors


def natural_centroid_differences(latent, paths, eligible):
    by_class = defaultdict(lambda: defaultdict(list))
    for index in eligible:
        path = paths[index]
        track, frame = path.stem.rsplit("_", 1)
        by_class[path.parent][track].append((int(frame), index))
    centered = []
    for groups in by_class.values():
        frames = [latent[[i for _, i in sorted(group)]] for group in groups.values()]
        aggregates = [np.stack([track[part].mean(axis=0) for track in frames])
                      for part in (slice(None), slice(0, None, 2), slice(1, None, 2))]
        centered.append([means - means.mean(axis=0) for means in aggregates])
    return tuple(np.concatenate(parts) for parts in zip(*centered, strict=True))


@torch.inference_mode()
def natural_track_embeddings(student, models, images, tracks, args, device, *, angle_control=False):
    output = {}
    for label, groups in tracks.items():
        output[label] = {}
        for fraction in args.strengths:
            indices = [group[round((len(group) - 1) * fraction)] for group in groups]
            chunks = []
            for batch in images[label][indices].split(args.batch_size):
                batch = batch.to(device, dtype=torch.float32) / 255
                clean = torch.cat((batch, batch.flip(-1)))
                predictions = {name: parameters(model, clean)[0] for name, model in models.items()}
                if angle_control:
                    shifts = predictions["pose_point"][:, 1:]
                    angle = predictions["paired_full"][:, :1]
                    perm = torch.arange(len(batch), device=device).roll(1)
                    predictions["angle_same_point"] = torch.cat((angle, shifts), dim=1)
                    shuffled_angle = angle[torch.cat((perm, perm + len(batch)))]
                    predictions["angle_shuffled"] = torch.cat((shuffled_angle, shifts), dim=1)
                features = {"raw": student(clean)}
                for name, center in predictions.items():
                    features[name] = student(pose_warp(clean, center.clamp(-2, 2)))
                chunks.append({name: array.reshape(2, len(batch), -1).cpu().numpy()
                               for name, array in features.items()})
            output[label][str(fraction)] = ({
                name: np.concatenate([chunk[name] for chunk in chunks], axis=1) for name in chunks[0]
            }, None)
    return output


def select_track_corrections(query, odd, seed):
    steps = np.asarray((0., .5, 1.), dtype=np.float32)
    odd = odd.cpu().numpy()
    tracks = []
    for positions in query.values():
        ordered = list(positions.values())
        raw = np.stack([features["raw"] for features, _ in ordered])
        delta = np.stack([features["same_track"] for features, _ in ordered]) - raw
        candidates = raw[None] + steps[:, None, None, None, None] * delta[None]
        dispersion = 0.
        for block in (odd, ~odd):
            values = candidates[..., block]
            normalized = values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)
            dispersion += np.square(normalized - normalized.mean(axis=1, keepdims=True)).sum(axis=-1).mean(axis=1)
        selected = steps[dispersion.argmin(axis=0)]
        tracks.append((ordered, raw, delta, selected))
    weights = np.concatenate([selected for _, _, _, selected in tracks], axis=1)
    mean_weight = weights.mean(axis=1, keepdims=True)
    order = np.random.default_rng(seed).permutation(weights.shape[1])
    shuffled = np.empty_like(weights)
    shuffled[:, order] = weights[:, np.roll(order, 1)]
    offsets = np.cumsum([selected.shape[1] for _, _, _, selected in tracks])
    for (ordered, raw, delta, selected), control in zip(tracks, np.split(shuffled, offsets[:-1], axis=1), strict=True):
        for i, (features, _) in enumerate(ordered):
            for name, fraction in (("matched_weight", mean_weight),
                                   ("consistency_weight", selected),
                                   ("shuffled_weight", control)):
                features[name] = raw[i] + fraction[..., None] * delta[i]
    return mean_weight[:, 0].tolist()


@torch.inference_mode()
def pooled_track_embeddings(student, model, images, tracks, args, device, seed):
    frame_indices = {
        label: torch.tensor([[group[round((len(group) - 1) * fraction)] for group in groups]
                            for fraction in (0., .5, 1.)])
        for label, groups in tracks.items()
    }
    frames = torch.cat([images[label][index] for label, index in frame_indices.items()], dim=1)
    frames = frames.to(device, dtype=torch.float32) / 255
    clean = torch.stack((frames, frames.flip(-1)), dim=1)
    flat = clean.flatten(0, 2)
    centers = torch.cat([parameters(model, batch)[0] for batch in flat.split(args.batch_size)])
    centers = centers.clamp(-2, 2).reshape(*clean.shape[:3], 3)
    order = torch.randperm(frames.shape[1], generator=torch.Generator(device=device).manual_seed(seed), device=device)
    permutation = torch.empty_like(order)
    permutation[order] = order.roll(1)
    corrected = pose_warp(flat, centers.flatten(0, 2)).reshape_as(clean)
    shuffled = pose_warp(flat, centers[:, :, permutation].flatten(0, 2)).reshape_as(clean)

    def encoded(batch):
        features = [student(chunk) for chunk in batch.reshape(-1, *batch.shape[-3:]).split(args.batch_size)]
        return torch.cat(features).reshape(*batch.shape[:-3], -1)

    raw_features, corrected_features = encoded(clean), encoded(corrected)
    clips = {label: {} for label in tracks}
    for count in args.strengths:
        variants = []
        for take in combinations(range(3), int(count)):
            index = list(take)
            variants.append({
                "raw": raw_features[index].mean(dim=0),
                "feature_corrected": corrected_features[index].mean(dim=0),
                "pixel_raw": encoded(clean[index].mean(dim=0)),
                "pixel_corrected": encoded(corrected[index].mean(dim=0)),
                "pixel_shuffled": encoded(shuffled[index].mean(dim=0)),
            })
        start = 0
        for label, groups in tracks.items():
            end = start + len(groups)
            clips[label][str(count)] = ({
                key: torch.cat([variant[key][:, start:end] for variant in variants], dim=1).cpu().numpy()
                for key in variants[0]
            }, None)
            start = end
    return clips


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


def pose_passes(model, observed):
    first = parameters(model, observed)[0].clamp(-2, 2)
    intermediate = pose_warp(observed, first)
    residual = parameters(model, intermediate)[0].clamp(-2, 2)
    return first, intermediate, residual


def fit_residual_stage(source, first_model, args, seed, *, residual_training):
    model = deepcopy(first_model).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    ordering = torch.Generator().manual_seed(seed)
    rng = torch.Generator(device=source.device).manual_seed(seed + 100)
    levels = source.new_tensor(MIXED_LEVELS)
    for _ in range(args.refiner_epochs):
        for indices in batches(len(source), args.batch_size, ordering):
            with torch.no_grad():
                clean = source[indices]
                reflected = torch.rand(len(clean), device=clean.device, generator=rng) < .5
                clean = torch.where(reflected[:, None, None, None], clean.flip(-1), clean)
                amplitude = levels[torch.randint(len(levels), (len(clean),), device=source.device, generator=rng)]
                pose = (2 * torch.rand(len(clean), 3, device=source.device, generator=rng) - 1) * amplitude[:, None]
                observed = pose_warp(clean, pose)
                target = inverse_pose(pose)
                if residual_training:
                    first = parameters(first_model, observed)[0].clamp(-2, 2)
                    observed = pose_warp(observed, first)
                    target = compose_pose(inverse_pose(first), target)
            location, log_std = parameters(model, observed)
            loss = normal_nll(location, log_std, target).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval()


@torch.inference_mode()
def fit_pose_refinement(probe, model, args, seed):
    rng = torch.Generator(device=probe.device).manual_seed(seed)
    levels = probe.new_tensor(MIXED_LEVELS)
    points, passes, targets = [], [], []
    for batch in probe.split(args.batch_size):
        samples = torch.cat((batch, batch.flip(-1)))
        scale = levels[torch.randint(len(levels), (len(samples),), device=probe.device, generator=rng)]
        pose = (2 * torch.rand(len(samples), 3, device=probe.device, generator=rng) - 1) * scale[:, None]
        first, _, residual = pose_passes(model, pose_warp(samples, pose))
        points.append(first)
        passes.append(torch.cat((first, residual), dim=1))
        targets.append(compose_pose(inverse_pose(first), inverse_pose(pose)))
    target = torch.cat(targets).double()
    return tuple(fit_risk(torch.cat(features).double(), target, args.selector_ridge) for features in (points, passes))


@torch.inference_mode()
def refinement_embeddings(student, model, images, args, device, seed, *, calibration=None, stages=None):
    rng = torch.Generator(device=device).manual_seed(seed)
    output, errors = {}, {str(s): [] for s in args.strengths}
    for label, samples in images.items():
        order = torch.randperm(len(samples), generator=rng, device=device)[:args.test_per_class]
        chunks = {str(s): [] for s in args.strengths}
        class_errors = {str(s): [] for s in args.strengths}
        for batch in samples[order.cpu()].split(args.batch_size):
            batch = batch.to(device, dtype=torch.float32) / 255
            clean = torch.cat((batch, batch.flip(-1)))
            unit_pose = 2 * torch.rand(len(clean), 3, device=device, generator=rng) - 1
            for strength in args.strengths:
                key = str(strength)
                pose = strength * unit_pose
                inverse = inverse_pose(pose)
                observed = pose_warp(clean, pose)
                first, intermediate, residual = pose_passes(model, observed)
                centers = {"pose_point": first, "iterative": compose_pose(first, residual),
                           "damped": compose_pose(first, residual / 2)}
                if calibration is not None:
                    first_adjustment = calibrated_prediction(calibration[0], first.double()).to(first.dtype)
                    both = torch.cat((first, residual), dim=1).double()
                    both_adjustment = calibrated_prediction(calibration[1], both).to(first.dtype)
                    centers["calibrated_first"] = compose_pose(first, first_adjustment)
                    centers["calibrated_refinement"] = compose_pose(first, both_adjustment)
                if stages is not None:
                    for name, stage in stages.items():
                        estimate = parameters(stage, intermediate)[0].clamp(-2, 2)
                        centers[name] = compose_pose(first, estimate)
                    centers["continued_point"] = parameters(stages["continued_two"], observed)[0]
                features = {"raw": student(observed), "oracle": student(pose_warp(observed, inverse))}
                for name, location in centers.items():
                    features[name] = student(pose_warp(observed, location.clamp(-2, 2)))
                chunks[key].append({name: values.reshape(2, len(batch), -1).cpu().numpy()
                                    for name, values in features.items()})
                class_errors[key].append((len(batch), {
                    name: (location.clamp(-2, 2) - inverse).square().mean(0).cpu().numpy()
                    for name, location in centers.items()
                }))
        output[label] = {s: ({name: np.concatenate([c[name] for c in values], axis=1)
                              for name in values[0]}, None) for s, values in chunks.items()}
        for strength, values in class_errors.items():
            errors[strength].append({
                name: np.average([scores[name] for _, scores in values], axis=0, weights=[n for n, _ in values])
                for name in values[0][1]
            })
    return output, {strength: {name: np.mean([value[name] for value in values], axis=0).tolist()
                               for name in values[0]} for strength, values in errors.items()}


@torch.inference_mode()
def symmetric_pose_embeddings(student, model, second_model, images, args, device, seed):
    rng = torch.Generator(device=device).manual_seed(seed)
    sampling = torch.Generator(device=device).manual_seed(seed + 1000)
    output, errors = {}, {str(s): [] for s in args.strengths}
    for label, samples in images.items():
        order = torch.randperm(len(samples), generator=rng, device=device)[:args.test_per_class]
        chunks = {str(s): [] for s in args.strengths}
        class_errors = {str(s): [] for s in args.strengths}
        for batch in samples[order.cpu()].split(args.batch_size):
            batch = batch.to(device, dtype=torch.float32) / 255
            clean = torch.cat((batch, batch.flip(-1)))
            unit_pose = 2 * torch.rand(len(clean), 3, device=device, generator=rng) - 1
            noise = torch.randn(args.views // 2, len(clean), 3, device=device, generator=sampling)
            noise = torch.cat((noise, -noise))
            for strength in args.strengths:
                key = str(strength)
                pose = strength * unit_pose
                inverse = inverse_pose(pose)
                observed = pose_warp(clean, pose)
                center, log_std = parameters(model, observed)
                reflected, reflected_log_std = parameters(model, observed.flip(-1))
                partner = reflect_pose(reflected)
                independent, _ = parameters(second_model, observed)
                delta = center - partner
                symmetric = (center + partner) / 2
                weight = (2 * (reflected_log_std - log_std)).sigmoid()
                precision = weight * center + (1 - weight) * partner
                mixture_variance = ((2 * log_std).exp() + (2 * reflected_log_std).exp()) / 2 + delta.square() / 4
                mixture_scale = mixture_variance.sqrt()
                centers = {"pose_point": center, "reflected_point": partner, "symmetric_point": symmetric,
                           "symmetric_precision": precision, "independent_point": (center + independent) / 2}
                features = {"raw": student(observed), "oracle": student(pose_warp(observed, inverse))}
                for name, location in centers.items():
                    features[name] = student(pose_warp(observed, location.clamp(-2, 2)))
                repeated = observed.repeat(args.views, 1, 1, 1)
                for name, location, scale in (("pose_posterior", center, log_std.exp()),
                                              ("symmetric_posterior", symmetric, mixture_scale)):
                    samples_pose = (location + noise * scale).clamp(-2, 2)
                    sampled = student(pose_warp(repeated, samples_pose.flatten(0, 1)))
                    features[name] = sampled.reshape(args.views, len(clean), -1).mean(0)
                chunks[key].append({name: value.reshape(2, len(batch), -1).cpu().numpy()
                                    for name, value in features.items()})
                class_errors[key].append((len(batch), {
                    name: (location.clamp(-2, 2) - inverse).square().mean(0).cpu().numpy()
                    for name, location in centers.items()
                }))
        output[label] = {s: ({name: np.concatenate([c[name] for c in values], axis=1)
                              for name in values[0]}, None) for s, values in chunks.items()}
        for strength, values in class_errors.items():
            errors[strength].append({
                name: np.average([scores[name] for _, scores in values], axis=0, weights=[n for n, _ in values])
                for name in values[0][1]
            })
    return output, {strength: {name: np.mean([value[name] for value in values], axis=0).tolist()
                               for name in values[0]} for strength, values in errors.items()}


@torch.inference_mode()
def identity_pose_embeddings(student, gaussian, hurdle, images, args, device, seed):
    rng = torch.Generator(device=device).manual_seed(seed)
    sampling = torch.Generator(device=device).manual_seed(seed + 1000)
    output, diagnostics = {}, {str(s): [] for s in args.strengths}
    for label, samples in images.items():
        order = torch.randperm(len(samples), generator=rng, device=device)[:args.test_per_class]
        chunks = {str(s): [] for s in args.strengths}
        label_stats = {str(s): [] for s in args.strengths}
        for batch in samples[order.cpu()].split(args.batch_size):
            batch = batch.to(device, dtype=torch.float32) / 255
            clean = torch.cat((batch, batch.flip(-1)))
            unit_pose = 2 * torch.rand(len(clean), 3, device=device, generator=rng) - 1
            noise = torch.randn(args.views // 2, len(clean), 3, device=device, generator=sampling)
            noise = torch.cat((noise, -noise))
            for strength in args.strengths:
                key = str(strength)
                pose = strength * unit_pose
                inverse = inverse_pose(pose)
                observed = clean if strength == 0 else pose_warp(clean, pose)
                gaussian_mean, gaussian_log_std, gaussian_logit = mixture_parameters(gaussian, observed)
                slab_mean, slab_log_std, logit = mixture_parameters(hurdle, observed)
                no_warp = logit.sigmoid().unsqueeze(1)
                centers = {"gaussian_point": gaussian_mean, "slab_point": slab_mean,
                           "fixed_prior": slab_mean * (1 - 1 / len(MIXED_LEVELS)),
                           "identity_mean": slab_mean * (1 - no_warp)}
                features = {"raw": student(observed), "oracle": student(pose_warp(observed, inverse))}
                for name, center in centers.items():
                    features[name] = student(pose_warp(observed, center.clamp(-2, 2)))
                repeated = observed.repeat(args.views, 1, 1, 1)
                for name, location, log_std in (("gaussian_posterior", gaussian_mean, gaussian_log_std),
                                                 ("identity_posterior", slab_mean, slab_log_std)):
                    draws = (location + noise * log_std.exp()).clamp(-2, 2)
                    generated = student(pose_warp(repeated, draws.flatten(0, 1)))
                    features[name] = generated.reshape(args.views, len(clean), -1).mean(0)
                features["identity_posterior"] = no_warp * features["raw"] + (1 - no_warp) * features["identity_posterior"]
                chunks[key].append({name: value.reshape(2, len(batch), -1).cpu().numpy()
                                    for name, value in features.items()})
                measures = {name: (center.clamp(-2, 2) - inverse).square().mean(0).cpu().numpy()
                            for name, center in centers.items()}
                measures.update(hurdle_p_zero=no_warp.mean().item(), gaussian_p_zero=gaussian_logit.sigmoid().mean().item())
                label_stats[key].append((len(batch), measures))
        output[label] = {s: ({name: np.concatenate([c[name] for c in values], axis=1)
                              for name in values[0]}, None) for s, values in chunks.items()}
        for strength, values in label_stats.items():
            diagnostics[strength].append({
                name: np.average([scores[name] for _, scores in values], axis=0, weights=[n for n, _ in values])
                for name in values[0][1]
            })
    return output, {strength: {name: np.mean([value[name] for value in values], axis=0).tolist()
                               for name in values[0]} for strength, values in diagnostics.items()}


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


def evaluate_task(pair, sign, support, query, tracks, kernels, args, rng, *, head_kernels=None):
    parity = np.concatenate((kernels["fixed_even"], kernels["fixed_odd"]))
    if args.track_head_evidence:
        geometry = ("head_same_track", "head_same_class")
        joint = np.concatenate((parity, *(head_kernels[name] for name in geometry)))
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
            identity = np.eye(f.shape[1])
            weights, probability = predict(parity, f, y, identity, args.noise, center_evidence=True)
            even, _ = predict(kernels["fixed_even"], f, y, identity, args.noise)
            heads = {name: predict(prior, f, y, identity, args.noise, center_evidence=True)[0]
                     for name, prior in (head_kernels or {}).items()}
            values = {"posterior_same": float(probability[0]), "strengths": {}}
            if args.track_head_evidence:
                adaptive, joint_mass = predict(joint, f, y, identity, args.noise, center_evidence=True)
                heads["head_evidence"] = adaptive
                heads["head_equal_mix"] = np.mean([weights, *(heads[name] for name in geometry)], axis=0)
                values["geometry_mass"] = joint_mass.reshape(3, 2).sum(axis=1)
            if args.track_parity_policy:
                values["correct_action"] = float((probability[0] >= .5) == (sign == 1))
            for strength, (embeddings, truth, n_negative) in batches_by_strength.items():
                choices = {name: features @ weights for name, features in embeddings.items()}
                choices.update({name: embeddings["raw"] @ head for name, head in heads.items()})
                if args.track_parity_policy:
                    p_same = probability[0]
                    even_logits, odd_logits = (choices[name] for name in ("even_teacher", "odd_teacher"))
                    choices.update({
                        "parity_soft": p_same * even_logits + (1 - p_same) * odd_logits,
                        "parity_fixed": (even_logits + odd_logits) / 2,
                        "parity_inverted": (1 - p_same) * even_logits + p_same * odd_logits,
                        "parity_hard": even_logits if p_same >= .5 else odd_logits,
                        "parity_oracle": even_logits if sign == 1 else odd_logits,
                    })
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
            **({"correct_action": float(np.mean([ep["correct_action"] for ep in episodes]))}
               if args.track_parity_policy else {}),
            **({"geometry_mass": np.mean([ep["geometry_mass"] for ep in episodes], axis=0).tolist()}
               if args.track_head_evidence else {}),
            "strengths": {strength: {key: {metric: float(np.mean([
                ep["strengths"][strength][key][metric] for ep in episodes]))
                for metric in measures} for key, measures in methods.items()}
                for strength, methods in episodes[0]["strengths"].items()},
        } for count, episodes in rows.items()
    }


def run(data, args, device, seed, *, source_partners=None, source_paths=None):
    (source, labels), (probe, _), train, test, tracks = data
    source = source.to(device, dtype=torch.float32) / 255
    probe = probe.to(device, dtype=torch.float32) / 255
    teacher = fit_teacher(source, labels.to(device), args, seed)
    basis_probe = (probe[::2] if args.selective or args.posterior_selection or args.pose_refinement or args.angle_gate
                   else probe)
    mean, basis, energies = discover_basis(teacher, basis_probe, args)
    student = fit_student(source, teacher, basis, mean, args, seed + 100)
    odd = energies > .5
    if not (args.identity_mixture or args.relative_pose or args.paired_angle_bias or args.angle_gate
            or args.natural_track_pose):
        pose_model = fit_canonicalizer(source, teacher, basis, odd, args, seed + 200, 0.)
    support = {label: features[0] for label, features in embed(student, train, args, device).items()}
    head_kernels = None
    if args.natural_track_pose:
        tracks, natural_queries = split_natural_tracks(tracks, args.shots, args.test_per_class, seed + 400)
        teacher_rotation = None
        teacher_parity = None
        if args.track_head_prior:
            models = {}
            head_kernels = natural_track_head_priors(
                teacher, basis, odd, source, source_partners, args, seed + 250, source_paths=source_paths,
                student=student if args.track_head_student else None,
                source_labels=labels if args.track_head_budget else None)
        elif source_partners is None:
            models = {name: fit_paired_canonicalizer(source, args, seed + 200, pairing, angle_bias=True)
                      for name, pairing in (("pose_point", "absolute"), ("paired_full", "same"))}
        else:
            if args.track_consistency or args.track_pooling or args.track_teacher_basis or args.track_teacher_parity:
                source_partners = {"same_track": source_partners["same_track"]}
            model_partners = source_partners if args.track_pooling else {"pose_point": None, **source_partners}
            models = {name: fit_canonicalizer(source, teacher, basis, odd, args, seed + 200, 0.,
                                             partners=partners)
                      for name, partners in model_partners.items()}
            if args.track_teacher_basis:
                partners = source_partners["same_track"]
                control, teacher_rotation = control_teacher_basis(teacher, basis, odd, source, partners, args, seed + 250)
                models["rotated_teacher"] = fit_canonicalizer(
                    source, teacher, control, odd, args, seed + 200, 0., partners=partners)
            if args.track_teacher_parity:
                partners = source_partners["same_track"]
                weights, energy = calibrate_teacher_parity(teacher, basis, odd, source, partners, args)
                teacher_parity = {"weights": weights, "raw_pair_mse": energy}
                for name, pair_weights in weights.items():
                    models[name] = fit_canonicalizer(
                        source, teacher, basis, odd, args, seed + 200, 0.,
                        partners=partners, pair_weights=pair_weights)
        if args.track_pooling:
            query = pooled_track_embeddings(student, models["same_track"], train, natural_queries, args, device, seed + 500)
        else:
            query = natural_track_embeddings(student, models, train, natural_queries, args, device,
                                             angle_control=source_partners is None and not args.track_head_prior)
        diagnostics = {"track_counts": {label: {"support": len(groups), "query": len(natural_queries[label])}
                                         for label, groups in tracks.items()}}
        if args.track_consistency:
            diagnostics["mean_track_weight"] = select_track_corrections(query, odd, seed + 500)
        if teacher_rotation is not None:
            diagnostics["teacher_rotation"] = teacher_rotation
        if teacher_parity is not None:
            diagnostics["teacher_parity"] = teacher_parity
    elif args.pose_refinement:
        calibration = fit_pose_refinement(probe[1::2], pose_model, args, seed + 250)
        query, errors = refinement_embeddings(student, pose_model, test, args, device, seed + 300,
                                              calibration=calibration)
        diagnostics = {"refinement_pose_mse": errors}
    elif args.train_residual:
        stages = {name: fit_residual_stage(source, pose_model, args, seed + 250, residual_training=target)
                  for name, target in (("continued_two", False), ("residual_trained", True))}
        query, errors = refinement_embeddings(student, pose_model, test, args, device, seed + 300, stages=stages)
        diagnostics = {"refinement_pose_mse": errors}
    elif args.identity_mixture:
        gaussian = fit_mixture_canonicalizer(source, args, seed + 200, hurdle=False)
        hurdle = fit_mixture_canonicalizer(source, args, seed + 200, hurdle=True)
        query, calibration = identity_pose_embeddings(student, gaussian, hurdle, test, args, device, seed + 300)
        diagnostics = {"identity_calibration": calibration}
    elif args.symmetrize_pose:
        second_model = fit_canonicalizer(source, teacher, basis, odd, args, seed + 1200, 0.)
        query, errors = symmetric_pose_embeddings(student, pose_model, second_model, test, args, device, seed + 300)
        diagnostics = {"symmetric_pose_mse": errors}
    elif args.posterior_selection:
        global_scale = global_pose_scale(probe[1::2], pose_model, args, seed + 250)
        query = posterior_selection_embeddings(student, pose_model, global_scale, test, args, device, seed + 300)
        diagnostics = {"global_pose_scale": global_scale.cpu().tolist()}
    elif args.selective:
        selection = calibrate_selectors(probe[1::2], pose_model, teacher, basis, odd, args, seed + 250)
        query, policy_stats = selective_embeddings(
            student, pose_model, teacher, basis, odd, selection, test, args, device, seed + 300)
        diagnostics = {"source_fixed_step": STEPS[selection[0]], "policy_statistics": policy_stats}
    else:
        angle_gate = None
        if args.angle_gate:
            models = {name: fit_paired_canonicalizer(source, args, seed + 200, pairing, angle_bias=True)
                      for name, pairing in (("pose", "absolute"), ("angle_same", "same"))}
            angle_gate = calibrate_angle_gate(probe[1::2], models, args, seed + 250)
        elif args.paired_angle_bias:
            models = {name: fit_paired_canonicalizer(source, args, seed + 200, pairing, angle_bias=True)
                      for name, pairing in (("pose", "absolute"), ("angle_same", "same"),
                                            ("angle_shuffled", "shuffled"), ("angle_point", "point"))}
        elif args.relative_pose:
            models = {name: fit_paired_canonicalizer(source, args, seed + 200, pairing)
                      for name, pairing in (("pose", "absolute"), ("relative", "same"), ("relative_shuffled", "shuffled"))}
        else:
            models = {"pose": pose_model, "hidden": fit_canonicalizer(
                source, teacher, basis, odd, args, seed + 200, args.hidden_weight)}
        query = canonical_embeddings(student, models, test, args, device, seed + 300, angle_gate=angle_gate)
        pose_methods = (*models, *ANGLE_GATE_MODELS) if angle_gate is not None else models
        diagnostics = {"pose_calibration": {strength: {
            method: {measure: np.mean([v[strength][1][method][measure] for v in query.values()], axis=0).tolist()
                     for measure in ("mse", "nll")}
            for method in pose_methods} for strength in map(str, args.strengths)}}
        if angle_gate is not None:
            diagnostics["fixed_angle_weight"] = angle_gate[-1].item()
    rng = np.random.default_rng(seed)
    kernels = prior_kernels(energies.cpu().numpy(), rng)
    tasks = {family: {"-".join(map(str, pair)): evaluate_task(
        pair, 1 if family == "invariant" else -1, support, query, tracks, kernels, args, rng, head_kernels=head_kernels)
        for pair in pairs} for family, pairs in TASKS.items()}
    print(seed, "completed", flush=True)
    return {"seed": seed, "tasks": tasks, **diagnostics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/tmp/ibt-gtsrb-data"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(81, 93)))
    parser.add_argument("--shots", type=int, nargs="+")
    parser.add_argument("--strengths", type=float, nargs="+")
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
    selection.add_argument("--symmetrize-pose", action="store_true")
    selection.add_argument("--identity-mixture", action="store_true")
    selection.add_argument("--pose-refinement", action="store_true")
    selection.add_argument("--train-residual", action="store_true")
    selection.add_argument("--relative-pose", action="store_true")
    selection.add_argument("--paired-angle-bias", action="store_true")
    selection.add_argument("--angle-gate", action="store_true")
    selection.add_argument("--natural-track-pose", action="store_true")
    selection.add_argument("--train-track-canonicalizer", action="store_true")
    selection.add_argument("--track-consistency", action="store_true")
    selection.add_argument("--track-pooling", action="store_true")
    selection.add_argument("--track-teacher-basis", action="store_true")
    selection.add_argument("--track-teacher-parity", action="store_true")
    selection.add_argument("--track-parity-policy", action="store_true")
    selection.add_argument("--track-head-prior", action="store_true")
    selection.add_argument("--track-head-between", action="store_true")
    selection.add_argument("--track-head-student", action="store_true")
    selection.add_argument("--track-head-evidence", action="store_true")
    selection.add_argument("--track-head-budget", action="store_true")
    parser.add_argument("--refiner-epochs", type=int, default=8)
    parser.add_argument("--selector-ridge", type=float, default=.05)
    args = parser.parse_args()
    if args.track_head_between or args.track_head_student or args.track_head_evidence or args.track_head_budget:
        args.track_head_prior = True
    if args.track_parity_policy:
        args.track_teacher_parity = True
    if args.track_consistency or args.track_pooling or args.track_teacher_basis or args.track_teacher_parity:
        args.train_track_canonicalizer = True
    if args.train_track_canonicalizer or args.track_head_prior:
        args.natural_track_pose = True
    if args.shots is None:
        args.shots = [1, 2, 3] if args.natural_track_pose else [1, 2, 4, 5]
    if args.strengths is None:
        if args.track_pooling:
            args.strengths = [1., 2., 3.]
        else:
            args.strengths = [0., .5, 1.] if args.natural_track_pose else [0., .5, 1., 1.5]
    if args.output is None:
        if args.track_head_budget:
            name = "track_head_budget"
        elif args.track_head_evidence:
            name = "track_head_evidence"
        elif args.track_head_student:
            name = "track_head_student"
        elif args.track_head_between:
            name = "track_head_between"
        elif args.track_head_prior:
            name = "track_head_prior"
        elif args.track_parity_policy:
            name = "track_parity_policy"
        elif args.track_teacher_parity:
            name = "track_teacher_parity"
        elif args.track_teacher_basis:
            name = "track_teacher_basis"
        elif args.track_pooling:
            name = "track_pooling"
        elif args.track_consistency:
            name = "track_consistency"
        elif args.train_track_canonicalizer:
            name = "trained_track_pose"
        elif args.natural_track_pose:
            name = "natural_track_pose"
        elif args.angle_gate:
            name = "angle_gate"
        elif args.paired_angle_bias:
            name = "paired_angle_bias"
        elif args.relative_pose:
            name = "relative_pose"
        elif args.train_residual:
            name = "trained_residual_pose"
        elif args.pose_refinement:
            name = "pose_refinement"
        elif args.identity_mixture:
            name = "identity_mixture"
        elif args.symmetrize_pose:
            name = "symmetric_pose"
        elif args.posterior_selection:
            name = "posterior_selection"
        else:
            name = "selective_pose" if args.selective else "canonical_pose"
        args.output = HERE / "results" / name / "metrics.json"
    if args.views < 2 or args.views % 2 or args.test_per_class < 1:
        parser.error("need a positive test size and an even number of at least two posterior views")
    if not all(math.isfinite(s) and s >= 0 for s in args.strengths):
        parser.error("strengths must be finite and nonnegative")
    if args.track_pooling and any(s not in {1., 2., 3.} for s in args.strengths):
        parser.error("track pooling requires between one and three frames")
    if args.natural_track_pose and not args.track_pooling and any(s > 1 for s in args.strengths):
        parser.error("natural track frame fractions must be in [0, 1]")
    if args.track_consistency and len(set(args.strengths)) < 2:
        parser.error("track consistency requires at least two frame positions")
    if not math.isfinite(args.selector_ridge) or args.selector_ridge <= 0:
        parser.error("selector ridge must be finite and positive")
    if args.refiner_epochs < 1:
        parser.error("refiner epochs must be positive")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    train_dataset = GTSRB(args.data, split="train", download=True)
    use_source_paths = args.train_track_canonicalizer or args.track_head_prior
    data = preprocess(train_dataset, GTSRB(args.data, split="test", download=True), args,
                      include_source_paths=use_source_paths)
    source_partners = None
    source_paths = None
    if use_source_paths:
        data, source_paths = data[:-1], data[-1]
        source_partners = natural_source_partners(source_paths)
    if args.natural_track_pose:
        paths = paths_by_class(train_dataset)
        for label, groups in data[-1].items():
            for group in groups:
                group.sort(key=lambda i: int(paths[label][i].stem.rsplit("_", 1)[1]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
              "torch_version": torch.__version__, "target_tasks": TASKS, "runs": []}
    if source_partners is not None:
        count = int((source_partners["same_track"] >= 0).sum())
        if not count:
            parser.error("training on tracks requires at least two source tracks with multiple selected frames")
        output["source_pair_count"] = count
        if args.track_head_budget:
            eligible = source_partners["same_track"].numpy() >= 0
            counts = np.unique(data[0][1].numpy()[eligible], return_counts=True)[1]
            output["source_budget_counts"] = {
                str(budget): int(np.minimum(counts, budget).sum()) for budget in TRACK_PAIR_BUDGETS
            }
    retained_paths = source_paths if args.track_head_between else None
    start = time.monotonic()
    for seed in args.seeds:
        output["runs"].append(run(data, args, device, seed, source_partners=source_partners,
                                  source_paths=retained_paths))
        output["elapsed_seconds"] = time.monotonic() - start
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
