"""Transfer a conditional distribution of inverse poses to held-out sign classes."""

import argparse
import json
import math
import time
from collections import defaultdict
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiment import (GTSRB, HERE, TASKS, PoseGenerator, batches, discover_basis, embed,
                        fit_student, fit_teacher, predict, preprocess, prior_kernels)
from reflection_evidence.experiment import FoldedReflectionStudent, MIXTURE, paths_by_class

ANGLE = math.radians(32)
SHIFT = .24
STEPS = (0., .25, .5, .75, 1.)
POSTERIOR_STEPS = (0., .5, 1.)
MIXED_LEVELS = (0., .5, 1.)
ANGLE_GATE_MODELS = ("gate_fixed", "gate_adaptive", "gate_shuffled")
TRACK_PAIR_BUDGETS = (2, 8, 32)
DISTILLATION_TAILS = {"sixth": 6, "third": 3, "half": 2}
PROJECTION_PENALTIES = {"penalty_zero": 0., "penalty_quarter": .25, "penalty_four": 4.}


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


def track_prior_spectrum(energies):
    scale = max(float(energies.mean()), 1e-12)
    values = 1 / (.25 + np.maximum(energies, 0) / scale)
    return values / values.sum()


def attach_track_cross_parity(prior, residuals, odd):
    even = ~odd
    first, second = residuals[:, even].astype(np.float64), residuals[:, odd].astype(np.float64)

    def regularized_whitener(values):
        moment = values.T @ values / len(values)
        energies, basis = np.linalg.eigh(moment)
        ridge = max(.1 * float(energies.mean()), 1e-12)
        return (basis * (np.maximum(energies, 0) + ridge)**-.5) @ basis.T

    correlation = regularized_whitener(first) @ (first.T @ second / len(first)) @ regularized_whitener(second)
    aligned, reversed_cross = prior.copy(), prior.copy()
    even_index, odd_index = np.ix_(even, even), np.ix_(odd, odd)
    cross_index, reverse_index = np.ix_(even, odd), np.ix_(odd, even)
    for mode, matrix in enumerate(prior):
        roots = []
        for index in (even_index, odd_index):
            energies, basis = np.linalg.eigh(matrix[index])
            roots.append((basis * np.sqrt(np.maximum(energies, 0))) @ basis.T)
        cross = .5 * roots[0] @ correlation @ roots[1]
        aligned[mode][cross_index] = -cross
        aligned[mode][reverse_index] = -cross.T
        reversed_cross[mode][cross_index] = cross
        reversed_cross[mode][reverse_index] = cross.T
    return ({"head_cross_aligned": aligned, "head_cross_reversed": reversed_cross},
            {"max_correlation": float(np.linalg.norm(correlation, ord=2))})


@torch.no_grad()
def natural_track_head_priors(teacher, basis, odd, source, partners, args, seed, *,
                              source_paths=None, student=None, source_labels=None, probe_data=None):
    rng = np.random.default_rng(seed)
    names = ["head_same_track", "head_same_class", "head_rotated"]
    spectrum_names = ("head_track_cross_spectrum", "head_class_cross_spectrum", "head_rotated_cross_spectrum")
    include_synthetic = args.track_head_synthetic or args.track_distillation_synthetic
    width = basis.shape[1]
    odd = odd.cpu().numpy()
    eligible = np.flatnonzero(partners["same_track"].numpy() >= 0)
    latent = torch.cat([teacher.encode(batch) @ basis for batch in source.split(args.batch_size)]).cpu().numpy()
    residuals = [latent[eligible] - latent[partners[name].numpy()[eligible]]
                 for name in ("same_track", "same_class")]
    centroids = None
    if args.track_head_between:
        centroids = natural_centroid_differences(latent, source_paths, eligible)
        names.extend(("head_track_means", "head_split_means"))
    student_residuals = []
    if student is not None:
        student_latent = torch.cat([student(batch) for batch in source.split(args.batch_size)]).cpu().numpy()
        student_residuals = [student_latent[eligible] - student_latent[partners[name].numpy()[eligible]]
                             for name in ("same_track", "same_class")]
        names.extend(("head_student_track", "head_student_class"))
    additional_residuals = {}
    pair_diagnostics = None
    if source_labels is not None and probe_data is None:
        labels = source_labels.numpy()[eligible]
        sampling = np.random.default_rng(seed + 1000)
        order = [sampling.permutation(np.flatnonzero(labels == label)) for label in np.unique(labels)]
        if args.track_head_budget:
            for budget in TRACK_PAIR_BUDGETS:
                subset = np.concatenate([index[:budget] for index in order])
                additional_residuals[f"head_pairs_{budget}"] = residuals[1][subset]
        if args.track_head_pair_control:
            grouped = np.concatenate(order)
            shift = max(map(len, order))
            if 2 * shift > len(grouped):
                raise ValueError("too many frames from one source class for cross-class matching")
            shuffled = sampling.permutation(len(grouped))
            values = latent[eligible]
            additional_residuals["head_cross_class"] = values[grouped] - values[np.roll(grouped, shift)]
            additional_residuals["head_random_pairs"] = values[shuffled] - values[np.roll(shuffled, 1)]
        if args.track_head_distance:
            energy = np.square(residuals[1]).mean(axis=1)
            ranked = [group[np.argsort(energy[group], kind="stable")] for group in order]
            near = np.concatenate([group[:len(group)//2] for group in ranked])
            far = np.concatenate([group[len(group)-len(group)//2:] for group in ranked])
            random_half = np.concatenate([group[:len(group)//2] for group in order])
            additional_residuals.update(head_near=residuals[1][near], head_far=residuals[1][far],
                                        head_random_half=residuals[1][random_half])
    if args.track_head_neighbors or args.track_head_predicted:
        logits = None
        if args.track_head_predicted:
            logits = torch.cat([teacher(batch) for batch in source.split(args.batch_size)]).cpu().numpy()
        neighbor_residuals, pair_diagnostics = natural_track_neighbor_residuals(
            latent, source_paths, eligible, odd, logits=logits)
        additional_residuals.update(neighbor_residuals)
    if probe_data is not None:
        probe, probe_labels, probe_partners = probe_data
        probe_latent = torch.cat([teacher.encode(batch) @ basis for batch in probe.split(args.batch_size)]).cpu().numpy()
        matched, pair_diagnostics = matched_source_track_residuals(
            residuals[1], source_labels.numpy()[eligible], probe_latent, probe_labels.numpy(),
            probe_partners["same_class"].numpy(), seed + 1100)
        additional_residuals.update(matched)
    if include_synthetic:
        rng_pose = torch.Generator(device=source.device).manual_seed(seed + 1200)
        distorted = []
        for indices in torch.from_numpy(eligible).split(args.batch_size):
            images = source[indices]
            pose = torch.rand(len(images), 3, device=source.device, generator=rng_pose) - .5
            distorted.append(teacher.encode(pose_warp(images, pose)) @ basis)
        additional_residuals["head_synthetic_pairs"] = latent[eligible] - torch.cat(distorted).cpu().numpy()
    names.extend(additional_residuals)
    if include_synthetic:
        names.append("head_natural_synthetic")
    if args.track_head_spectrum:
        names.extend(spectrum_names)
    priors = {name: np.zeros((2, width, width)) for name in names}
    for block, proportions in ((~odd, (1-MIXTURE[0], 1-MIXTURE[-1])), (odd, (MIXTURE[0], MIXTURE[-1]))):
        moments = []
        for residual in residuals:
            part = residual[:, block].astype(np.float64)
            moments.append(part.T @ part / len(part))
        energies, same = np.linalg.eigh(moments[0])
        cross_energies, other = np.linalg.eigh(moments[1])
        rotated, _ = np.linalg.qr(rng.normal(size=same.shape))
        bases = [same, other, rotated]
        if centroids is not None:
            whole, first, second = [member[:, block].astype(np.float64) for member in centroids]
            cross = first.T @ second
            bases.extend(np.linalg.eigh(moment)[1] for moment in (whole.T @ whole, (cross + cross.T) / 2))
        for residual in (*student_residuals, *additional_residuals.values()):
            part = residual[:, block].astype(np.float64)
            bases.append(np.linalg.eigh(part.T @ part)[1])
        if include_synthetic:
            synthetic = additional_residuals["head_synthetic_pairs"][:, block].astype(np.float64)
            synthetic_moment = synthetic.T @ synthetic / len(synthetic)
            mixed = moments[1] / max(float(np.trace(moments[1])), 1e-12)
            mixed += synthetic_moment / max(float(np.trace(synthetic_moment)), 1e-12)
            bases.append(np.linalg.eigh(mixed)[1])
        spectrum = track_prior_spectrum(energies)
        spectra = {name: spectrum for name in names}
        if args.track_head_spectrum:
            bases.extend((same, other, rotated))
            cross_spectrum = track_prior_spectrum(cross_energies)
            spectra.update({name: cross_spectrum for name in spectrum_names})
        index = np.ix_(block, block)
        for name, axes in zip(names, bases, strict=True):
            block_covariance = (axes * spectra[name]) @ axes.T
            for mode, proportion in enumerate(proportions):
                priors[name][mode][index] = proportion * block_covariance
    if args.track_head_blocks or args.track_distillation_blocks:
        for name, mask in (("head_learned_even", ~odd), ("head_learned_odd", odd)):
            keep = mask[:, None] & mask[None, :]
            priors[name] = np.where(keep, priors["head_same_class"], priors["head_rotated"])
    if args.track_head_cross_parity:
        cross_priors, pair_diagnostics = attach_track_cross_parity(priors["head_same_class"], residuals[1], odd)
        priors.update(cross_priors)
    return priors, pair_diagnostics


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


def natural_track_neighbor_residuals(latent, paths, eligible, odd, *, logits=None):
    frames = defaultdict(list)
    for index in eligible:
        path = paths[index]
        track, frame = path.stem.rsplit("_", 1)
        frames[path.parent, track].append((int(frame), index))
    groups = [np.array([index for _, index in sorted(paths_in_track)]) for paths_in_track in frames.values()]
    centroids = np.stack([latent[group].mean(axis=0) for group in groups])
    labels = np.array([key[0].name for key in frames])
    residuals, diagnostics = {}, {}
    even = centroids[:, ~odd]
    methods = [("head_neighbor_even", even, None)]
    if logits is None:
        methods.append(("head_neighbor_full", centroids, None))
    else:
        predicted = np.array([logits[group].mean(axis=0).argmax() for group in groups])
        methods.extend((("head_predicted_class", even, predicted), ("head_known_class_nearest", even, labels)))
    for name, features, classes in methods:
        normalized = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
        similarities = normalized @ normalized.T
        np.fill_diagonal(similarities, -np.inf)
        fallback = np.zeros(len(groups), dtype=bool)
        if classes is not None:
            permitted = classes[:, None] == classes[None, :]
            np.fill_diagonal(permitted, False)
            fallback = ~permitted.any(axis=1)
            similarities = np.where(permitted | fallback[:, None], similarities, -np.inf)
        neighbors = similarities.argmax(axis=1)
        differences = []
        for group, partner in zip(groups, neighbors, strict=True):
            other = groups[partner]
            positions = np.rint(np.linspace(0, len(other) - 1, len(group))).astype(int)
            differences.append(latent[group] - latent[other[positions]])
        residuals[name] = np.concatenate(differences)
        diagnostics[name] = {
            "same_class_fraction": float(np.average(labels == labels[neighbors], weights=list(map(len, groups)))),
            "unique_partner_tracks": len(np.unique(neighbors)),
            "fallback_fraction": float(np.average(fallback, weights=list(map(len, groups)))),
        }
    return residuals, diagnostics


def matched_source_track_residuals(train, train_labels, probe, probe_labels, partners, seed):
    eligible = partners >= 0
    probe_residual = probe[eligible] - probe[partners[eligible]]
    probe_labels = probe_labels[eligible]
    rng = np.random.default_rng(seed)
    train_subset, probe_subset, count_by_class = [], [], {}
    for label in np.intersect1d(train_labels, probe_labels):
        first = np.flatnonzero(train_labels == label)
        second = np.flatnonzero(probe_labels == label)
        count = min(len(first), len(second))
        train_subset.append(train[rng.permutation(first)[:count]])
        probe_subset.append(probe_residual[rng.permutation(second)[:count]])
        count_by_class[str(label)] = count
    if not count_by_class:
        raise ValueError("no shared source classes with track pairs")
    return ({"head_matched_train": np.concatenate(train_subset), "head_matched_probe": np.concatenate(probe_subset)},
            {"pair_count": sum(count_by_class.values()), "class_count": len(count_by_class)})


def repeated_source_indices(labels, per_class, seed):
    rng = np.random.default_rng(seed)
    indices = np.empty(len(labels), dtype=np.int64)
    for label in np.unique(labels):
        group = np.flatnonzero(labels == label)
        kept = rng.permutation(group)[:per_class]
        duplicates = rng.choice(kept, len(group) - len(kept), replace=True)
        indices[group] = rng.permutation(np.r_[kept, duplicates])
    return indices


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


def reflect_projected_views(views, action):
    return (views + views[::-1] @ action) / 2


def fit_reflection_involution(clean, reflected, odd_count, reference=None):
    cross = clean.T @ reflected
    symmetric = (cross + cross.T) / 2
    if reference is not None:
        symmetric = symmetric / max(np.linalg.norm(symmetric, "fro"), 1e-12)
        symmetric += reference / np.linalg.norm(reference, "fro")
    _, basis = np.linalg.eigh(symmetric)
    negative = basis[:, :odd_count]
    return np.eye(len(basis)) - 2 * negative @ negative.T


@torch.inference_mode()
def calibrate_track_projection(student, images, odd_count, batch_size, *, sparse=None, reference=None):
    pairs = [student(torch.cat((batch, batch.flip(-1)))).reshape(2, len(batch), -1).cpu().numpy()
             for batch in images.split(batch_size)]
    clean, reflected = np.concatenate(pairs, axis=1).astype(np.float64)
    actions = {"calibrated": fit_reflection_involution(clean, reflected, odd_count)}
    if sparse is not None:
        actions["sparse"] = fit_reflection_involution(clean[sparse], reflected[sparse], odd_count)
        actions["shrunk"] = fit_reflection_involution(clean[sparse], reflected[sparse], odd_count, reference)
    return actions


def add_projected_track_readouts(name, support, original, query, actions, prior, distilled_support):
    for variant, action in actions.items():
        key = f"{name}_{variant}"
        distilled_support[key] = ({label: reflect_projected_views(views, action)[0]
                                   for label, views in support.items()}, prior)
        for label, positions in original.items():
            for strength, (values, _) in positions.items():
                query[label][strength][0][key] = reflect_projected_views(values["raw"], action)


def add_track_readout(name, support, original, query, prior, distilled_support):
    distilled_support[name] = support, prior
    for label, positions in original.items():
        for strength, (values, _) in positions.items():
            query[label][strength][0][name] = values["raw"]


@torch.inference_mode()
def compare_folded_projection(student, folded, probe, odd, device):
    sign = torch.where(odd, -1., 1.)

    def two_views(batch):
        clean, reflected = student(torch.cat((batch, batch.flip(-1)))).chunk(2)
        return (clean + reflected * sign) / 2

    timings = {}
    max_error = 0.
    for size in sorted({min(size, len(probe)) for size in (16, 64, 128)}):
        batch = probe[:size]
        max_error = max(max_error, float((two_views(batch) - folded(batch)).abs().max()))
        samples = {"two_view": [], "folded": []}
        methods = (("two_view", two_views), ("folded", folded))
        for _, method in methods:
            for _ in range(10):
                method(batch)
        if device.type == "cuda":
            stream = torch.cuda.current_stream(device)
            torch.cuda.synchronize(device)
        for index in range(6):
            for name, method in (methods if index % 2 == 0 else methods[::-1]):
                if device.type == "cuda":
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record(stream)
                    for _ in range(30):
                        method(batch)
                    end.record(stream)
                    end.synchronize()
                    milliseconds = start.elapsed_time(end)
                else:
                    start = time.perf_counter()
                    for _ in range(30):
                        method(batch)
                    milliseconds = 1000 * (time.perf_counter() - start)
                samples[name].append(milliseconds / 30)
        timings[str(len(batch))] = {name: float(np.median(times)) for name, times in samples.items()}
    return {"latency_ms": timings, "max_prediction_difference": max_error}


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


def evaluate_task(pair, sign, support, query, tracks, kernels, args, rng, *, head_kernels=None, distilled_support=None):
    parity = np.concatenate((kernels["fixed_even"], kernels["fixed_odd"]))
    prediction_priors = {} if args.track_parity_prior_route else (head_kernels or {})
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
                     for name, prior in prediction_priors.items()}
            distilled_heads = {}
            for name, (features, prior) in (distilled_support or {}).items():
                design = np.concatenate([features[label][j[:count]] for label, j in zip(pair, ids, strict=True)])
                selected_prior = parity if prior is None else prior
                distilled_heads[name] = predict(selected_prior, design, y, identity, args.noise, center_evidence=True)[0]
            values = {"posterior_same": float(probability[0]), "strengths": {}}
            if args.track_head_evidence:
                adaptive, joint_mass = predict(joint, f, y, identity, args.noise, center_evidence=True)
                heads["head_evidence"] = adaptive
                heads["head_equal_mix"] = np.mean([weights, *(heads[name] for name in geometry)], axis=0)
                values["geometry_mass"] = joint_mass.reshape(3, 2).sum(axis=1)
            if args.track_parity_policy:
                values["correct_action"] = float((probability[0] >= .5) == (sign == 1))
            if args.track_parity_prior_route:
                _, route_probability = predict(head_kernels["head_same_class"], f, y, identity,
                                               args.noise, center_evidence=True)
                route_weight = route_probability[0]
                values["prior_correct_action"] = float((route_weight >= .5) == (sign == 1))
            for strength, (embeddings, truth, n_negative) in batches_by_strength.items():
                choices = {name: features @ distilled_heads.get(name, weights) for name, features in embeddings.items()}
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
                    if args.track_parity_prior_route:
                        choices["parity_prior_soft"] = route_weight * even_logits + (1 - route_weight) * odd_logits
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
            **({"prior_correct_action": float(np.mean([ep["prior_correct_action"] for ep in episodes]))}
               if args.track_parity_prior_route else {}),
            **({"geometry_mass": np.mean([ep["geometry_mass"] for ep in episodes], axis=0).tolist()}
               if args.track_head_evidence else {}),
            "strengths": {strength: {key: {metric: float(np.mean([
                ep["strengths"][strength][key][metric] for ep in episodes]))
                for metric in measures} for key, measures in methods.items()}
                for strength, methods in episodes[0]["strengths"].items()},
        } for count, episodes in rows.items()
    }


def run(data, args, device, seed, *, source_partners=None, source_paths=None, probe_partners=None):
    (source, labels), (probe, probe_labels), train, test, tracks = data
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
    support_views = embed(student, train, args, device)
    support = {label: views[0] for label, views in support_views.items()}
    head_kernels = None
    distilled_support = None
    if args.natural_track_pose:
        tracks, natural_queries = split_natural_tracks(tracks, args.shots, args.test_per_class, seed + 400)
        teacher_rotation = None
        teacher_parity = None
        pair_diagnostics = None
        folding = None
        if args.track_distillation:
            models = {}
        elif args.track_head_prior:
            models = {}
            head_kernels, pair_diagnostics = natural_track_head_priors(
                teacher, basis, odd, probe if args.track_head_predicted else source, source_partners,
                args, seed + 250, source_paths=source_paths,
                student=student if args.track_head_student else None,
                source_labels=labels if (args.track_head_budget or args.track_head_pair_control
                                         or args.track_head_distance or args.track_head_source_split) else None,
                probe_data=(probe, probe_labels, probe_partners) if args.track_head_source_split else None)
        elif source_partners is None:
            models = {name: fit_paired_canonicalizer(source, args, seed + 200, pairing, angle_bias=True)
                      for name, pairing in (("pose_point", "absolute"), ("paired_full", "same"))}
        else:
            model_partners = source_partners
            if args.track_consistency or args.track_pooling or args.track_teacher_basis or args.track_teacher_parity:
                model_partners = {"same_track": source_partners["same_track"]}
            if not (args.track_pooling or args.track_parity_prior_route):
                model_partners = {"pose_point": None, **model_partners}
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
            if args.track_parity_prior_route:
                routing_priors, _ = natural_track_head_priors(
                    teacher, basis, odd, source, source_partners, args, seed + 250)
                head_kernels = {"head_same_class": routing_priors["head_same_class"]}
        if args.track_pooling:
            query = pooled_track_embeddings(student, models["same_track"], train, natural_queries, args, device, seed + 500)
        else:
            query = natural_track_embeddings(student, models, train, natural_queries, args, device,
                                             angle_control=source_partners is None and not args.track_head_prior
                                             and not args.track_distillation)
        if args.track_distillation:
            priors, _ = natural_track_head_priors(teacher, basis, odd, source, source_partners, args, seed + 250)
            distilled_support = {}
            if args.track_distillation_joint:
                head_kernels = {key: priors[key] for key in ("head_same_class", "head_rotated")}
            variants = [("track_distilled", "head_same_class", source, None),
                        ("rotated_distilled", "head_rotated", source, None)]
            if args.track_distillation_blocks:
                variants.extend((("even_distilled", "head_learned_even", source, None),
                                 ("odd_distilled", "head_learned_odd", source, None)))
            if args.track_distillation_synthetic:
                variants.extend((("synthetic_distilled", "head_synthetic_pairs", source, None),
                                 ("mixed_distilled", "head_natural_synthetic", source, None)))
            if args.track_distillation_budget:
                indices = repeated_source_indices(labels.numpy(), 32, seed + 600)
                limited = source[torch.from_numpy(indices)]
                variants.extend((("budget_uniform", None, limited, None),
                                 ("budget_track", "head_same_class", limited, None),
                                 ("budget_rotated", "head_rotated", limited, None)))
            if args.track_distillation_schedule:
                half = args.student_epochs // 2
                early, late = range(half), range(args.student_epochs - half, args.student_epochs)
                variants = [(f"{phase}_{name}", key, source, epochs)
                            for phase, epochs in (("early", early), ("late", late))
                            for name, key in (("track", "head_same_class"), ("rotated", "head_rotated"))]
            if args.track_distillation_tail:
                for phase, divisor in DISTILLATION_TAILS.items():
                    start = args.student_epochs - args.student_epochs // divisor
                    epochs = range(start, args.student_epochs)
                    variants.extend(((f"{phase}_track", "head_same_class", source, epochs),
                                     (f"{phase}_rotated", "head_rotated", source, epochs)))
            frozen_epochs = {}
            if args.track_distillation_late_head:
                late = range(args.student_epochs - args.student_epochs // 2, args.student_epochs)
                metric_keys = (("track", "head_same_class"), ("rotated", "head_rotated"))
                variants = [(f"late_{name}", key, source, late) for name, key in metric_keys]
                head_variants = [(f"head_{name}", key, source, late)
                                 for name, key in (*metric_keys, ("uniform", None))]
                variants.extend(head_variants)
                frozen_epochs = {name: late for name, *_ in head_variants}
            projection_actions = {}
            projection_prior = None
            projection_distances = {}
            projection_penalties = {}
            reflection_schedules = {}
            if args.track_distillation_projection:
                late = range(args.student_epochs - args.student_epochs // 2, args.student_epochs)
                variants = [("late_track", "head_same_class", source, late)]
                reflection_sign = np.where(odd.cpu().numpy(), -1., 1.)
                projection_actions = {"aligned": np.diag(reflection_sign)}
                if args.track_distillation_projection_train:
                    variants.extend((("trained_uniform", None, source, None),
                                     ("trained_late", "head_same_class", source, late)))
                    projection_penalties = {"trained_uniform": 0., "trained_late": 0.}
                elif args.track_distillation_projection_penalty:
                    variants.extend((method, "head_same_class", source, late) for method in PROJECTION_PENALTIES)
                    projection_penalties = PROJECTION_PENALTIES
                elif args.track_distillation_projection_schedule:
                    projection_penalties = {"project_early": 0., "project_late": 0., "project_all": 0.}
                    reflection_schedules = {"project_early": range(len(late)), "project_late": late}
                    variants.extend((method, "head_same_class", source, late) for method in projection_penalties)
                elif args.track_distillation_projection_timing_weights:
                    for geometry, key in (("uniform", None), ("natural", "head_same_class")):
                        for phase, epochs in (("early", range(len(late))), ("late", late)):
                            method = f"{geometry}_{phase}"
                            variants.append((method, key, source, late))
                            projection_penalties[method] = 0.
                            reflection_schedules[method] = epochs
                elif args.track_distillation_projection_compress:
                    variants.append(("trained_late", "head_same_class", source, late))
                    projection_penalties = {"trained_late": 0.}
                elif args.track_distillation_projection_folded:
                    variants = [("trained_late", "head_same_class", source, late)]
                    projection_penalties = {"trained_late": 0.}
                else:
                    rotation, _ = np.linalg.qr(np.random.default_rng(seed + 950).standard_normal(
                        (len(reflection_sign), len(reflection_sign))))
                    projection_actions["rotated"] = (rotation * reflection_sign) @ rotation.T
                projection_prior = np.eye(len(reflection_sign))[None] / len(reflection_sign)
                distilled_support["raw"] = support, projection_prior
                raw_actions = projection_actions
                if args.track_distillation_projection_calibrated:
                    probe_pairs = probe[::2]
                    odd_count = int(odd.sum().item())
                    sparse = None
                    reference = projection_actions["aligned"]
                    if args.track_distillation_projection_shrink:
                        labels_for_pairs = probe_labels[::2].numpy()
                        selection_rng = np.random.default_rng(seed + 960)
                        sparse = np.concatenate([
                            selection_rng.permutation(np.flatnonzero(labels_for_pairs == label))[:2]
                            for label in np.unique(labels_for_pairs)
                        ])
                    calibrated = calibrate_track_projection(student, probe_pairs, odd_count, args.batch_size,
                                                            sparse=sparse, reference=reference)
                    raw_actions = {**projection_actions, **calibrated}
                    distance = np.linalg.norm(calibrated["calibrated"] - reference, "fro")
                    projection_distances["raw"] = float(distance / np.sqrt(len(reflection_sign)))
                add_projected_track_readouts("raw", support_views, query, query, raw_actions,
                                             projection_prior, distilled_support)
            narrow_variants = {}
            if args.track_distillation_width:
                narrow_variants = {"narrow_uniform": None, "narrow_track": "head_same_class", "narrow_rotated": "head_rotated"}
                variants.extend((name, key, source, None) for name, key in narrow_variants.items())
            for name, key, training_source, metric_epochs in variants:
                metric = None
                if key is not None:
                    matrix = priors[key].sum(axis=0)
                    metric = source.new_tensor(matrix / np.diag(matrix).mean())
                during_training = name in projection_penalties
                learned = fit_student(training_source, teacher, basis, mean, args, seed + 100,
                                      metric=metric, metric_epochs=metric_epochs,
                                      hidden_width=48 if name in narrow_variants else 192,
                                      frozen_feature_epochs=frozen_epochs.get(name),
                                      reflection_action=source.new_tensor(projection_actions["aligned"])
                                      if during_training else None,
                                      reflection_penalty=projection_penalties.get(name, 0.),
                                      reflection_epochs=reflection_schedules.get(name))
                features = embed(learned, train, args, device)
                variant_support = {label: values[0] for label, values in features.items()}
                transformed = natural_track_embeddings(learned, {}, train, natural_queries, args, device)
                evaluators = [] if during_training else [(name, projection_prior)]
                if args.track_distillation_joint:
                    evaluators.extend(((name + "_track_head", priors["head_same_class"]),
                                       (name + "_rotated_head", priors["head_rotated"])))
                for method, prior in evaluators:
                    add_track_readout(method, variant_support, transformed, query, prior, distilled_support)
                if projection_actions:
                    learned_actions = projection_actions
                    if args.track_distillation_projection_calibrated:
                        calibrated = calibrate_track_projection(learned, probe_pairs, odd_count, args.batch_size,
                                                                sparse=sparse, reference=reference)
                        learned_actions = {**projection_actions, **calibrated}
                        distance = np.linalg.norm(calibrated["calibrated"] - reference, "fro")
                        projection_distances[name] = float(distance / np.sqrt(len(reflection_sign)))
                    add_projected_track_readouts(name, features, transformed, query, learned_actions,
                                                 projection_prior, distilled_support)
                if args.track_distillation_projection_compress:
                    action = source.new_tensor(projection_actions["aligned"])

                    def encode_projected(batch, model=learned):
                        return (model(batch) + model(teacher.view(batch)) @ action) / 2

                    projected_teacher = SimpleNamespace(encode=encode_projected, view=teacher.view)
                    width = len(action)
                    identity = torch.eye(width, device=device, dtype=source.dtype)
                    compressed = fit_student(source, projected_teacher, identity, source.new_zeros((width,)),
                                             args, seed + 700)
                    compressed_support = {label: views[0] for label, views in embed(compressed, train, args, device).items()}
                    compressed_query = natural_track_embeddings(compressed, {}, train, natural_queries, args, device)
                    add_track_readout(f"{name}_compressed", compressed_support, compressed_query, query,
                                      projection_prior, distilled_support)
                if args.track_distillation_projection_folded:
                    folded = FoldedReflectionStudent(learned, odd).eval()
                    folded_support = {label: views[0] for label, views in embed(folded, train, args, device).items()}
                    folded_query = natural_track_embeddings(folded, {}, train, natural_queries, args, device)
                    add_track_readout(f"{name}_folded", folded_support, folded_query, query,
                                      projection_prior, distilled_support)
                    folding = compare_folded_projection(learned, folded, probe, odd, device)
        diagnostics = {"track_counts": {label: {"support": len(groups), "query": len(natural_queries[label])}
                                         for label, groups in tracks.items()}}
        if args.track_distillation_budget:
            diagnostics["student_distillation_budget"] = {"full": len(source), "limited": len(np.unique(indices))}
        if args.track_distillation_projection_calibrated:
            diagnostics["student_reflection_calibration"] = {"pair_count": len(probe_pairs),
                                                              "distance": projection_distances}
            if args.track_distillation_projection_shrink:
                diagnostics["student_reflection_calibration"]["sparse_pair_count"] = len(sparse)
        if folding is not None:
            diagnostics["folded_projection"] = folding
        if args.track_consistency:
            diagnostics["mean_track_weight"] = select_track_corrections(query, odd, seed + 500)
        if teacher_rotation is not None:
            diagnostics["teacher_rotation"] = teacher_rotation
        if teacher_parity is not None:
            diagnostics["teacher_parity"] = teacher_parity
        if pair_diagnostics is not None:
            key = "neighbor_pairing"
            if args.track_head_source_split:
                key = "matched_source"
            elif args.track_head_cross_parity:
                key = "cross_parity"
            diagnostics[key] = pair_diagnostics
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
        pair, 1 if family == "invariant" else -1, support, query, tracks, kernels, args, rng,
        head_kernels=head_kernels, distilled_support=distilled_support)
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
    selection.add_argument("--track-parity-prior-route", action="store_true")
    selection.add_argument("--track-head-prior", action="store_true")
    selection.add_argument("--track-head-between", action="store_true")
    selection.add_argument("--track-head-student", action="store_true")
    selection.add_argument("--track-head-evidence", action="store_true")
    selection.add_argument("--track-head-budget", action="store_true")
    selection.add_argument("--track-head-blocks", action="store_true")
    selection.add_argument("--track-head-pair-control", action="store_true")
    selection.add_argument("--track-head-distance", action="store_true")
    selection.add_argument("--track-head-neighbors", action="store_true")
    selection.add_argument("--track-head-spectrum", action="store_true")
    selection.add_argument("--track-head-predicted", action="store_true")
    selection.add_argument("--track-head-source-split", action="store_true")
    selection.add_argument("--track-head-synthetic", action="store_true")
    selection.add_argument("--track-head-cross-parity", action="store_true")
    selection.add_argument("--track-distillation", action="store_true")
    selection.add_argument("--track-distillation-budget", action="store_true")
    selection.add_argument("--track-distillation-joint", action="store_true")
    selection.add_argument("--track-distillation-blocks", action="store_true")
    selection.add_argument("--track-distillation-schedule", action="store_true")
    selection.add_argument("--track-distillation-width", action="store_true")
    selection.add_argument("--track-distillation-synthetic", action="store_true")
    selection.add_argument("--track-distillation-tail", action="store_true")
    selection.add_argument("--track-distillation-late-head", action="store_true")
    selection.add_argument("--track-distillation-projection", action="store_true")
    selection.add_argument("--track-distillation-projection-train", action="store_true")
    selection.add_argument("--track-distillation-projection-calibrated", action="store_true")
    selection.add_argument("--track-distillation-projection-penalty", action="store_true")
    selection.add_argument("--track-distillation-projection-shrink", action="store_true")
    selection.add_argument("--track-distillation-projection-schedule", action="store_true")
    selection.add_argument("--track-distillation-projection-compress", action="store_true")
    selection.add_argument("--track-distillation-projection-timing-weights", action="store_true")
    selection.add_argument("--track-distillation-projection-folded", action="store_true")
    parser.add_argument("--refiner-epochs", type=int, default=8)
    parser.add_argument("--selector-ridge", type=float, default=.05)
    args = parser.parse_args()
    if args.track_distillation_projection_shrink:
        args.track_distillation_projection_calibrated = True
    if (args.track_distillation_projection_train or args.track_distillation_projection_calibrated
            or args.track_distillation_projection_penalty or args.track_distillation_projection_schedule
            or args.track_distillation_projection_compress or args.track_distillation_projection_timing_weights
            or args.track_distillation_projection_folded):
        args.track_distillation_projection = True
    if (args.track_distillation_budget or args.track_distillation_joint or args.track_distillation_blocks
            or args.track_distillation_schedule or args.track_distillation_width or args.track_distillation_synthetic
            or args.track_distillation_tail or args.track_distillation_late_head or args.track_distillation_projection):
        args.track_distillation = True
    if args.track_parity_prior_route:
        args.track_parity_policy = True
    if (args.track_head_between or args.track_head_student or args.track_head_evidence or args.track_head_budget
            or args.track_head_blocks or args.track_head_pair_control or args.track_head_distance
            or args.track_head_neighbors or args.track_head_spectrum or args.track_head_predicted
            or args.track_head_source_split or args.track_head_synthetic or args.track_head_cross_parity):
        args.track_head_prior = True
    if args.track_parity_policy:
        args.track_teacher_parity = True
    if args.track_consistency or args.track_pooling or args.track_teacher_basis or args.track_teacher_parity:
        args.train_track_canonicalizer = True
    if args.train_track_canonicalizer or args.track_head_prior or args.track_distillation:
        args.natural_track_pose = True
    if args.shots is None:
        args.shots = [1, 2, 3] if args.natural_track_pose else [1, 2, 4, 5]
    if args.strengths is None:
        if args.track_pooling:
            args.strengths = [1., 2., 3.]
        else:
            args.strengths = [0., .5, 1.] if args.natural_track_pose else [0., .5, 1., 1.5]
    if args.output is None:
        if args.track_distillation_width:
            name = "track_distillation_width"
        elif args.track_distillation_synthetic:
            name = "track_distillation_synthetic"
        elif args.track_distillation_tail:
            name = "track_distillation_tail"
        elif args.track_distillation_late_head:
            name = "track_distillation_late_head"
        elif args.track_distillation_projection_train:
            name = "track_distillation_projection_train"
        elif args.track_distillation_projection_shrink:
            name = "track_distillation_projection_shrink"
        elif args.track_distillation_projection_calibrated:
            name = "track_distillation_projection_calibrated"
        elif args.track_distillation_projection_penalty:
            name = "track_distillation_projection_penalty"
        elif args.track_distillation_projection_schedule:
            name = "track_distillation_projection_schedule"
        elif args.track_distillation_projection_compress:
            name = "track_distillation_projection_compress"
        elif args.track_distillation_projection_timing_weights:
            name = "track_distillation_projection_timing_weights"
        elif args.track_distillation_projection_folded:
            name = "track_distillation_projection_folded"
        elif args.track_distillation_projection:
            name = "track_distillation_projection"
        elif args.track_distillation_schedule:
            name = "track_distillation_schedule"
        elif args.track_distillation_blocks:
            name = "track_distillation_blocks"
        elif args.track_distillation_joint:
            name = "track_distillation_joint"
        elif args.track_distillation_budget:
            name = "track_distillation_budget"
        elif args.track_distillation:
            name = "track_distillation"
        elif args.track_parity_prior_route:
            name = "track_parity_prior_route"
        elif args.track_head_cross_parity:
            name = "track_head_cross_parity"
        elif args.track_head_synthetic:
            name = "track_head_synthetic"
        elif args.track_head_source_split:
            name = "track_head_source_split"
        elif args.track_head_predicted:
            name = "track_head_predicted"
        elif args.track_head_spectrum:
            name = "track_head_spectrum"
        elif args.track_head_neighbors:
            name = "track_head_neighbors"
        elif args.track_head_distance:
            name = "track_head_distance"
        elif args.track_head_pair_control:
            name = "track_head_pair_control"
        elif args.track_head_blocks:
            name = "track_head_blocks"
        elif args.track_head_budget:
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
    if (args.track_distillation_schedule or args.track_distillation_late_head
            or args.track_distillation_projection) and args.student_epochs < 2:
        parser.error("the weighted schedule requires at least two student epochs")
    if args.track_distillation_tail and args.student_epochs < max(DISTILLATION_TAILS.values()):
        parser.error("the weighted tail comparison requires at least six student epochs")
    if not math.isfinite(args.selector_ridge) or args.selector_ridge <= 0:
        parser.error("selector ridge must be finite and positive")
    if args.refiner_epochs < 1:
        parser.error("refiner epochs must be positive")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    train_dataset = GTSRB(args.data, split="train", download=True)
    use_source_paths = args.train_track_canonicalizer or args.track_head_prior or args.track_distillation
    data = preprocess(train_dataset, GTSRB(args.data, split="test", download=True), args,
                      include_source_paths=use_source_paths and not args.track_head_predicted,
                      include_probe_paths=args.track_head_predicted or args.track_head_source_split)
    source_partners = None
    source_paths = None
    probe_partners = None
    if args.track_head_source_split:
        data, source_paths, probe_paths = data[:-2], data[-2], data[-1]
        source_partners = natural_source_partners(source_paths)
        probe_partners = natural_source_partners(probe_paths)
    elif use_source_paths:
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
    if args.track_distillation_tail:
        output["distillation_tail_epochs"] = {phase: args.student_epochs // divisor
                                               for phase, divisor in DISTILLATION_TAILS.items()}
    if args.track_distillation_projection_penalty:
        output["projection_penalties"] = {**PROJECTION_PENALTIES, "late_track": 1.}
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
    retained_paths = source_paths if args.track_head_between or args.track_head_neighbors or args.track_head_predicted else None
    start = time.monotonic()
    for seed in args.seeds:
        output["runs"].append(run(data, args, device, seed, source_partners=source_partners,
                                  source_paths=retained_paths, probe_partners=probe_partners))
        output["elapsed_seconds"] = time.monotonic() - start
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
