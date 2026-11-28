"""Compare a learned quarter-turn projection in distillation and inference."""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
from scipy.linalg import schur
import torch

from experiment import (CIFAR10, HERE, class_split, data_partitions, discover_basis,
                        evaluate, fit_student, fit_teacher, predict)
from reflection_evidence.experiment import class_target_means

RESULTS = HERE / "results" / "quarter_turn_projection"
VIEWS = tuple(partial(torch.rot90, k=rotation, dims=(-2, -1)) for rotation in range(1, 4))
QUARTER_TURNS = np.array([[[1, 0], [0, 1]], [[0, -1], [1, 0]],
                          [[-1, 0], [0, -1]], [[0, 1], [-1, 0]]])
VIEW_BUDGETS = {"one": (0,), "opposite": (0, 2), "adjacent": (0, 1), "four": (0, 1, 2, 3)}
VARIANCE_WEIGHTS = {"variance_quarter_training": .25, "variance_unit_training": 1.}
PAIR_VARIANCE_WEIGHTS = {"opposite_variance_training": 1., "adjacent_variance_training": 1.}


def dihedral_view(images, rotation, mirrored):
    rotated = torch.rot90(images, rotation, (-2, -1))
    return rotated.flip(-1) if mirrored else rotated


DIHEDRAL_VIEWS = tuple(partial(dihedral_view, rotation=turn, mirrored=mirror)
                       for turn in range(4) for mirror in (False, True) if turn or mirror)


def rotation_cross_moment(orbit):
    left = orbit.reshape(-1, orbit.shape[-1])
    return left.T @ np.roll(orbit, -1, axis=0).reshape(left.shape)


def rotation_from_pairs(orbit):
    u, _, vt = np.linalg.svd(rotation_cross_moment(orbit), full_matrices=False)
    return u @ vt


def round_quarter_turn(action, fraction=0.):
    blocks, basis = schur(action, output="real")
    selected = np.zeros_like(blocks)
    offset = 0
    while offset < len(blocks):
        if offset + 1 < len(blocks) and abs(blocks[offset + 1, offset]) > 1e-10:
            subspace = slice(offset, offset + 2)
            angle = np.arctan2(blocks[offset + 1, offset], blocks[offset, offset])
            quarter = int(np.rint(angle / (np.pi / 2)))
            if fraction:
                angle = (1 - fraction) * quarter * (np.pi / 2) + fraction * angle
                cosine, sine = np.cos(angle), np.sin(angle)
                selected[subspace, subspace] = ((cosine, -sine), (sine, cosine))
            else:
                selected[subspace, subspace] = QUARTER_TURNS[quarter % 4]
            offset += 2
        else:
            selected[offset, offset] = 1 if blocks[offset, offset] >= 0 else -1
            offset += 1
    return basis @ selected @ basis.T


def rotation_powers(action):
    return np.stack([np.linalg.matrix_power(action, turns) for turns in range(4)])


def dihedral_rotation(orbit, odd):
    cross = rotation_cross_moment(orbit)
    between = (cross - cross.T)[np.ix_(~odd, odd)]
    even_axes, _, odd_axes_t = np.linalg.svd(between, full_matrices=True)
    frame = np.zeros_like(cross)
    even_count = int((~odd).sum())
    frame[np.ix_(~odd, np.arange(even_count))] = even_axes
    frame[np.ix_(odd, np.arange(even_count, len(odd)))] = odd_axes_t.T
    scores = frame.T @ cross @ frame
    selected = np.diag(np.where(np.diag(scores) < 0, -1., 1.))
    for even_index in range(min(even_count, len(odd) - even_count)):
        pair = np.ix_([even_index, even_count + even_index], [even_index, even_count + even_index])
        block = scores[pair]
        best = int(np.argmax([np.sum(block * candidate) for candidate in (selected[pair], QUARTER_TURNS[1], QUARTER_TURNS[3])]))
        if best:
            selected[pair] = QUARTER_TURNS[1 if best == 1 else 3]
    return frame @ selected @ frame.T


def dihedral_actions(rotation, reflection):
    powers = rotation_powers(rotation)
    return np.stack((powers, powers @ reflection), axis=1).reshape(8, len(rotation), len(rotation))


def subgroup_actions(rotation, reflection):
    powers = rotation_powers(rotation)
    rotations = np.stack((powers, powers[[0, 3, 2, 1]]))
    rotation_order = (0, 2, 4, 6, 1, 3, 5, 7)
    reflections = np.repeat(np.stack((np.eye(len(reflection)), reflection))[None], 4, axis=0)
    return (("rotations_only", rotations, rotation_order), ("reflection_only", reflections, None))


@torch.inference_mode()
def teacher_orbit(teacher, images, basis, mean, batch_size):
    parts = []
    for batch in images.split(batch_size):
        views = torch.cat((batch, *(view(batch) for view in VIEWS)))
        parts.append(((teacher.encode(views) - mean) @ basis).reshape(4, len(batch), -1).cpu().numpy())
    return np.concatenate(parts, axis=1).astype(np.float64)


@torch.inference_mode()
def invariant_embeddings(student, dataset, fixed, args, device, *, views=VIEWS, return_orbit=False):
    result = {}
    for label, images in dataset.items():
        parts = []
        for batch in images.split(args.batch_size):
            batch = batch.to(device, dtype=torch.float32) / 255
            image_views = torch.cat((batch, *(view(batch) for view in views)))
            orbit = student(image_views).reshape(len(views) + 1, len(batch), -1)
            embedding = orbit if return_orbit else orbit.mean(dim=0)
            parts.append((embedding @ fixed).cpu().numpy())
        result[label] = np.concatenate(parts, axis=1) if return_orbit else np.stack([np.concatenate(parts)] * 2)
    return result


def pool_rotation_views(orbit, offsets):
    if len(offsets) == len(orbit):
        return np.broadcast_to(orbit.mean(axis=0), orbit.shape)
    return np.mean([np.roll(orbit, -offset, axis=0) for offset in offsets], axis=0)


def evaluate_view_budgets(prior, support, queries, target_classes, args, seed):
    query_orbit = np.concatenate([queries[label] for label in target_classes], axis=1).astype(np.float64)
    truth = np.repeat(np.arange(len(target_classes)), [queries[label].shape[1] for label in target_classes])
    fixed_queries = {name: pool_rotation_views(query_orbit, offsets) for name, offsets in VIEW_BUDGETS.items()}
    fixed_support = {name: {label: pool_rotation_views(orbit, offsets)[0] for label, orbit in support.items()}
                     for name, offsets in VIEW_BUDGETS.items()}
    rng = np.random.default_rng(seed)
    observations = {str(count): {name: [] for name in VIEW_BUDGETS} for count in args.shots}
    for _ in range(args.episodes):
        chosen = [rng.choice(support[label].shape[1], max(args.shots), replace=False) for label in target_classes]
        for count in args.shots:
            labels = np.repeat(np.eye(len(target_classes)), count, axis=0) - 1 / len(target_classes)
            for name, pooled_support in fixed_support.items():
                train = np.concatenate([pooled_support[label][indices[:count]]
                                        for label, indices in zip(target_classes, chosen, strict=True)]).astype(np.float64)
                predictions = predict(prior, train, labels, fixed_queries[name], args.noise).argmax(axis=-1)
                correct = predictions == truth
                agreement = (predictions == predictions[:1]).all(axis=0).mean()
                observations[str(count)][name].append((correct.mean(), correct.all(axis=0).mean(), agreement))
    return {count: {name: dict(zip(("mean_accuracy", "all_rotations_accuracy", "rotation_agreement"),
                                  np.mean(rows, axis=0).tolist(), strict=True))
                    for name, rows in measurements.items()} for count, measurements in observations.items()}


def run(train, test, args, seed, device):
    source_classes, target_classes = class_split(seed, vary_classes=True)
    source, probe, target, query = data_partitions(train, test, args, seed, device,
                                                   source_classes, target_classes)
    labels = torch.arange(len(source_classes), device=device).repeat_interleave(args.source_examples)
    teacher = fit_teacher(source, labels, args, seed, len(source_classes), equivariant=not args.ordinary_teacher)
    basis_probe = probe[::2] if args.ordinary_teacher else probe
    mean, basis, parity = discover_basis(teacher, basis_probe, args)
    probe_orbit = teacher_orbit(teacher, probe, basis, mean, args.batch_size)
    if args.dihedral_action:
        odd = parity.cpu().numpy() > .5
        reflection = np.diag(np.where(odd, -1., 1.))
        action = dihedral_rotation(probe_orbit[:, ::2], odd)
        orbit_views = DIHEDRAL_VIEWS
        powers = dihedral_actions(action, reflection)
    else:
        fitted = rotation_from_pairs(probe_orbit[:, ::2])
        action = round_quarter_turn(fitted)
        orbit_views = VIEWS
        powers = rotation_powers(action)
    candidates = {"learned": action}
    if args.approximate_action:
        candidates.update(midpoint=round_quarter_turn(fitted, .5), unrounded=fitted)
    elif not (args.view_budget or args.dihedral_subgroups or args.ordinary_teacher_class_targets):
        generator = np.random.default_rng(seed + 250)
        rotation, _ = np.linalg.qr(generator.standard_normal(action.shape))
        candidates["rotated"] = rotation @ action @ rotation.T
    projector = powers.mean(axis=0)
    eig, directions = np.linalg.eigh((projector + projector.T) / 2)
    fixed = directions[:, eig > .5]
    if not fixed.shape[1]:
        raise RuntimeError("the learned rotation has no invariant coordinates")
    prior = np.eye(fixed.shape[1]) / fixed.shape[1]
    fixed_tensor = source.new_tensor(fixed)

    shots = {str(count): {} for count in args.shots}
    variants = [("inference_only", None, None)]
    for name, action_option in candidates.items():
        if args.dihedral_action:
            reflection_option = reflection if name == "learned" else rotation @ reflection @ rotation.T
            actions = dihedral_actions(action_option, reflection_option)
        else:
            actions = rotation_powers(action_option)
        variants.append((f"{name}_training", actions, None))
    if args.dihedral_subgroups:
        variants.extend(subgroup_actions(action, reflection))
    if args.training_view_budget:
        pairs = np.repeat(powers[[0, 2]][None], 2, axis=0)
        variants.append(("half_turn_training", pairs, (0, 2, 1, 3)))
    if args.paired_view_alignment:
        pairs = np.repeat(powers[[0, 1]][None], 2, axis=0)
        variants.append(("adjacent_training", pairs, ((0, 1, 2, 3), (1, 2, 3, 0))))
    variance_weights = VARIANCE_WEIGHTS if args.invariant_variance else {}
    variants.extend((name, powers, None) for name in variance_weights)
    variance_projector = None
    if args.pair_variance:
        variance_weights = PAIR_VARIANCE_WEIGHTS
        variance_projector = source.new_tensor(projector)
        pair_variants = {"half_turn_training": "opposite_variance_training",
                         "adjacent_training": "adjacent_variance_training"}
        variants.extend([(pair_variants[name], actions, order) for name, actions, order in variants
                         if name in pair_variants])
    target_transforms = {}
    if args.ordinary_teacher_class_targets:
        class_targets = partial(class_target_means, labels=labels, restore_variance=False)
        target_transforms = {"class_inference_only": class_targets, "class_learned_training": class_targets,
                             "scaled_class_learned_training": partial(class_target_means, labels=labels,
                                                                      restore_variance=True)}
        variants.extend((("class_inference_only", None, None), ("class_learned_training", powers, None),
                         ("scaled_class_learned_training", powers, None)))
    for name, actions, action_order in variants:
        student = fit_student(source, teacher, basis, mean, args, seed + 100, training_views=orbit_views,
                              orbit_actions=None if actions is None else source.new_tensor(actions),
                              orbit_order=action_order, invariant_variance_weight=variance_weights.get(name, 0.),
                              variance_projector=variance_projector, target_transform=target_transforms.get(name))
        support, queries = (invariant_embeddings(student, part, fixed_tensor, args, device, views=orbit_views,
                                                 return_orbit=args.view_budget)
                            for part in (target, query))
        if args.view_budget:
            scores = evaluate_view_budgets(prior, support, queries, target_classes, args, seed + 500)
            for count, row in scores.items():
                shots[count][name] = row
        else:
            scores = evaluate({name: prior}, support, queries, target_classes, args,
                              np.random.default_rng(seed + 500))
            for count, row in scores.items():
                shots[count].update(row)
    holdout = probe_orbit[:, 1::2]
    energy = max(float(np.square(holdout).mean()), 1e-12)
    diagnostics = {name: float(np.square(np.roll(holdout, -1, axis=0) - holdout @ generator).mean() / energy)
                   for name, generator in candidates.items()}
    identity = np.eye(len(action))
    order_errors = {name: float(np.max(np.abs(np.linalg.matrix_power(generator, 4) - identity)))
                    for name, generator in candidates.items()}
    result = {"seed": seed, "source_classes": source_classes, "target_classes": target_classes,
              "latent_width": int(len(action)), "invariant_width": int(fixed.shape[1]),
              "action_error": diagnostics, "order_four_error": order_errors["learned"], "shots": shots}
    if args.approximate_action:
        result["approximate_order_four_errors"] = {name: order_errors[name] for name in ("midpoint", "unrounded")}
    if args.dihedral_action:
        result["reflection_rotation_error"] = float(np.max(np.abs(reflection @ action @ reflection - action.T)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/tmp/ibt-cifar-data"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--shots", type=int, nargs="+", default=[5, 20, 60])
    parser.add_argument("--teacher-epochs", type=int, default=26)
    parser.add_argument("--student-epochs", type=int, default=22)
    parser.add_argument("--source-examples", type=int, default=1400)
    parser.add_argument("--probe-examples", type=int, default=320)
    parser.add_argument("--target-pool", type=int, default=600)
    parser.add_argument("--test-per-class", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--noise", type=float, default=.45)
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument("--approximate-action", action="store_true")
    comparison.add_argument("--view-budget", action="store_true")
    comparison.add_argument("--dihedral-action", action="store_true")
    comparison.add_argument("--dihedral-subgroups", action="store_true")
    comparison.add_argument("--training-view-budget", action="store_true")
    comparison.add_argument("--invariant-variance", action="store_true")
    comparison.add_argument("--paired-view-alignment", action="store_true")
    comparison.add_argument("--pair-variance", action="store_true")
    comparison.add_argument("--ordinary-teacher", action="store_true")
    comparison.add_argument("--ordinary-teacher-action-rounding", action="store_true")
    comparison.add_argument("--ordinary-teacher-class-targets", action="store_true")
    args = parser.parse_args()
    if args.ordinary_teacher_action_rounding:
        args.ordinary_teacher = args.approximate_action = True
    if args.ordinary_teacher_class_targets:
        args.ordinary_teacher = True
    if args.dihedral_subgroups:
        args.dihedral_action = True
    if args.pair_variance:
        args.paired_view_alignment = True
    if args.paired_view_alignment:
        args.training_view_budget = True
    if args.training_view_budget or args.invariant_variance:
        args.view_budget = True
    dataset_name, first_seed = RESULTS.name, 1249
    if args.ordinary_teacher_class_targets:
        dataset_name, first_seed = "quarter_turn_ordinary_teacher_class_targets", 1425
    elif args.ordinary_teacher_action_rounding:
        dataset_name, first_seed = "quarter_turn_ordinary_teacher_rounding", 1409
    elif args.ordinary_teacher:
        dataset_name, first_seed = "quarter_turn_ordinary_teacher", 1393
    elif args.approximate_action:
        dataset_name, first_seed = "quarter_turn_approximate", 1265
    elif args.pair_variance:
        dataset_name, first_seed = "quarter_turn_pair_variance", 1377
    elif args.paired_view_alignment:
        dataset_name, first_seed = "quarter_turn_paired_view_alignment", 1361
    elif args.invariant_variance:
        dataset_name, first_seed = "quarter_turn_invariant_variance", 1345
    elif args.training_view_budget:
        dataset_name, first_seed = "quarter_turn_training_view_budget", 1329
    elif args.view_budget:
        dataset_name, first_seed = "quarter_turn_view_budget", 1281
    elif args.dihedral_subgroups:
        dataset_name, first_seed = "dihedral_subgroups", 1313
    elif args.dihedral_action:
        dataset_name, first_seed = "dihedral_projection", 1297
    if args.output is None:
        args.output = RESULTS.with_name(dataset_name) / "metrics.json"
    if args.seeds is None:
        args.seeds = list(range(first_seed, first_seed + 16))
    if not 0 < min(args.shots) <= max(args.shots) <= args.target_pool <= 5000:
        parser.error("shots must fit a nonempty target pool of at most 5000 examples per class")
    if args.probe_examples < 2 or not 0 < args.source_examples <= 5000 - args.probe_examples:
        parser.error("need distinct source and probe samples with at least two probe examples per class")
    if not 0 < args.test_per_class <= 1000 or min(args.batch_size, args.teacher_epochs, args.student_epochs,
                                                   args.episodes) < 1 or not 0 < args.noise < float("inf"):
        parser.error("invalid test count, batch size, training steps, target episodes or head noise")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    train, test = (CIFAR10(args.data, train=is_train, download=True) for is_train in (True, False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = {"config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "torch_version": torch.__version__, "runs": []}
    if args.invariant_variance:
        data["invariant_variance_weights"] = VARIANCE_WEIGHTS
    if args.pair_variance:
        data["invariant_variance_weights"] = PAIR_VARIANCE_WEIGHTS
    start = time.monotonic()
    for seed in args.seeds:
        data["runs"].append(run(train, test, args, seed, device))
        data["elapsed_seconds"] = time.monotonic() - start
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)
        print(seed, "completed", flush=True)


if __name__ == "__main__":
    main()
