"""Transfer joint generator constraints to held-out image classes."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiment import (CIFAR10, GTSRB, HERE, TRAFFIC_SOURCE, budget_weights, class_split,
                        data_partitions, discover_basis, fit_student, fit_teacher, paths_by_class,
                        predict, traffic_source)
from reflection_evidence.experiment import Teacher, reflect

ROTATION = (2, 7, 4, 1, 6, 3, 0, 5)
REFLECTION = (1, 0, 3, 2, 5, 4, 7, 6)
COMPOSITION = tuple(REFLECTION[i] for i in ROTATION)
TEST_VIEWS = {"reflection": 1, "rotation": 2, "composition": 3, "half_turn": 4}
ORBIT_ORDERS = {"horizontal": tuple(range(8)), "vertical": (0, 5, 2, 7, 4, 1, 6, 3)}
TARGET_PAIR_PRIORS = ("isotropic", "split_balanced", "composition_mean_moment")


def vertical_reflect(x):
    return x.flip(-2)


@torch.inference_mode()
def embed_orbit(student, images, device, batch_size):
    views = [[] for _ in range(8)]
    for batch in images.split(batch_size):
        batch = batch.to(device=device, dtype=torch.float32)
        if images.dtype == torch.uint8:
            batch = batch / 255
        for k in range(4):
            rotated = torch.rot90(batch, k, (-2, -1))
            views[2 * k].append(student(rotated).cpu().numpy())
            views[2 * k + 1].append(student(rotated.flip(-1)).cpu().numpy())
    return np.stack([np.concatenate(view) for view in views]).astype(np.float64)


def generator_moments(orbit, complete, actions=(ROTATION, REFLECTION)):
    outputs = []
    start = np.arange(8) if complete else np.array([0])
    for action in actions:
        error = ((orbit[start] - orbit[np.asarray(action)[start]]) / 2).reshape(-1, orbit.shape[-1])
        moment = error.T @ error / len(error)
        outputs.append(normalize_moment(moment))
    return np.stack(outputs)


def normalize_moment(moment):
    return moment / max(np.trace(moment) / len(moment), 1e-12)


def composition_moments(orbit, path_composition=False):
    """Build composition moments from disjoint sets of rotation and reflection pairs."""
    width = orbit.shape[-1]
    folds = np.arange(orbit.shape[1]) % 4
    ridge = .001 * np.eye(width)
    names = ("plug_in", "additive", "transported")
    if path_composition:
        names += ("reverse_path", "mean_prediction", "mean_moment", "wrong_prediction")
    moments = {key: np.zeros((width, width)) for key in names}
    composition_operator = np.zeros((width, width))
    for holdout in (folds < 2, folds >= 2):
        operators, innovations = [], []
        for parity, action in enumerate((ROTATION, REFLECTION)):
            use_action = folds % 2 == parity
            train = orbit[:, ~holdout & use_action]
            x = train.reshape(-1, width)
            y = train[np.asarray(action)].reshape(-1, width)
            gram = x.T @ x / len(x)
            cross = x.T @ y / len(x)
            operator = np.linalg.solve(gram + ridge, cross)
            if path_composition and parity == 0:
                # Reversing the pairs preserves the Gram matrix on each complete orbit.
                inverse_rotation = np.linalg.solve(gram + ridge, cross.T)
            validation = orbit[:, holdout & use_action]
            residual = (validation[np.asarray(action)] - validation @ operator).reshape(-1, width)
            operators.append(operator)
            innovations.append(residual.T @ residual / len(residual))
        rotation, reflection = operators
        rotation_error, reflection_error = innovations
        points = orbit[:, holdout].reshape(-1, width)
        difference = points - points @ rotation @ reflection
        plug_in = difference.T @ difference / len(difference)
        moments["plug_in"] += plug_in
        moments["additive"] += plug_in + rotation_error + reflection_error
        moments["transported"] += plug_in + reflection.T @ rotation_error @ reflection + reflection_error
        if path_composition:
            composition_operator += (rotation @ reflection + reflection @ inverse_rotation) / 4
            reverse = points - points @ reflection @ inverse_rotation
            mean = (difference + reverse) / 2
            wrong = (difference + points - points @ reflection @ rotation) / 2
            reverse_moment = reverse.T @ reverse / len(reverse)
            moments["reverse_path"] += reverse_moment
            moments["mean_prediction"] += mean.T @ mean / len(mean)
            moments["mean_moment"] += (plug_in + reverse_moment) / 2
            moments["wrong_prediction"] += wrong.T @ wrong / len(wrong)
    return {key: normalize_moment(moment) for key, moment in moments.items()}, composition_operator


def budget_covariance(moment, budget):
    energies, axes = np.linalg.eigh(moment)
    weights, _, _ = budget_weights(np.maximum(energies, 0), budget)
    return (axes * weights) @ axes.T


def generator_balanced(moments, requested):
    """Solve the two trace constraints by searching the mixing weight of their dual."""
    def solve(budget):
        def candidate(weight):
            covariance = budget_covariance(weight * moments[0] + (1 - weight) * moments[1], budget)
            return covariance, np.einsum("gij,ij->g", moments, covariance)

        left, right = 0., 1.
        cov, energy = candidate(left)
        if energy[0] <= energy[1]:
            return cov, energy, left
        cov, energy = candidate(right)
        if energy[1] <= energy[0]:
            return cov, energy, right
        for _ in range(28):
            mid = (left + right) / 2
            cov, energy = candidate(mid)
            if energy[0] > energy[1]:
                left = mid
            else:
                right = mid
        return cov, energy, mid

    covariance, achieved, weight = solve(requested)
    # If the two budgets have no common interior, relax only on source data.
    budget = requested
    if achieved.max() > requested + 1e-5:
        low, high = requested, 1.
        for _ in range(22):
            mid = (low + high) / 2
            _, energy, _ = solve(mid)
            if energy.max() <= mid + 1e-5:
                high = mid
            else:
                low = mid
        budget = high
        covariance, achieved, weight = solve(budget)
    return covariance, achieved.tolist(), float(weight), float(budget)


def source_priors(probe, budget, seed, generator_controls=False, operator_composition=False,
                  path_composition=False):
    width = probe.shape[-1]
    flat = probe.reshape(-1, width)
    ridge = 1e-4 * max(float(np.mean(flat ** 2)), 1e-12)
    metric = flat.T @ flat / len(flat) + ridge * np.eye(width)
    whitener = np.linalg.solve(np.linalg.cholesky(metric).T, np.eye(width))
    orbit = probe @ whitener
    natural = {"isotropic": np.eye(width) / width}
    diagnostics = {}
    residual_model = None
    rng = np.random.default_rng(seed + 407)
    for key, complete in (("canonical", False), ("orbit", True)):
        moments = generator_moments(orbit, complete)
        joint, achieved, mixture, feasible = generator_balanced(moments, budget)
        average = budget_covariance(moments.mean(axis=0), budget)
        natural[f"{key}_average"] = average
        natural[f"{key}_balanced"] = joint
        diagnostics[key] = {"balanced": achieved, "rotation_weight": mixture, "budget": feasible,
                            "average": np.einsum("gij,ij->g", moments, average).tolist()}
        if complete:
            for i in range(4):
                rotation, _ = np.linalg.qr(rng.normal(size=(width, width)))
                natural[f"orbit_rotated_{i}"] = rotation @ joint @ rotation.T
            if generator_controls or operator_composition or path_composition:
                composition, = generator_moments(orbit, complete, (COMPOSITION,))
                action_moments = np.concatenate([moments, composition[None]], axis=0)
                for key, moment in zip(("rotation", "reflection", "composition"), action_moments, strict=True):
                    control = budget_covariance(moment, budget)
                    natural[f"orbit_{key}_only"] = control
                    diagnostics["orbit"][f"{key}_only"] = np.einsum("gij,ij->g", action_moments, control).tolist()
                diagnostics["orbit"]["balanced_composition"] = float(np.sum(joint * composition))
                diagnostics["orbit"]["average_composition"] = float(np.sum(average * composition))
    if operator_composition or path_composition:
        split_moments = np.concatenate([
            generator_moments(orbit[:, parity::2], True, (action,))
            for parity, action in enumerate((ROTATION, REFLECTION))
        ])
        natural["split_balanced"], _, _, _ = generator_balanced(split_moments, budget)
        inferred, composition_operator = composition_moments(orbit, path_composition)
        if path_composition:
            residual = whitener @ (np.eye(width) - composition_operator) / np.sqrt(2)
            residual_model = np.linalg.solve(whitener.T, residual.T).T, metric
        for key, moment in inferred.items():
            natural[f"composition_{key}"] = budget_covariance(moment, budget)
        for i in range(4):
            axes, _ = np.linalg.qr(rng.normal(size=(width, width)))
            natural[f"transported_rotated_{i}"] = axes @ natural["composition_transported"] @ axes.T
            if path_composition:
                natural[f"path_rotated_{i}"] = axes @ natural["composition_mean_prediction"] @ axes.T
        composition_scale = max(float(np.linalg.norm(composition)), 1e-12)
        diagnostics["composition_operators"] = {
            "relative_moment_error": {key: float(np.linalg.norm(moment - composition) / composition_scale)
                                      for key, moment in inferred.items()},
            "actual_composition_variance": {key: float(np.sum(prior * composition)) for key, prior in natural.items()},
            "estimated_composition_variance": {key: float(np.sum(natural[f"composition_{key}"] * moment))
                                               for key, moment in inferred.items()},
        }
    return {k: whitener @ v @ whitener.T for k, v in natural.items()}, diagnostics, residual_model


def measure_orbit_prior(student, probe, support, test_views, target, args, seed, device, *, both_axes=False):
    source_views = embed_orbit(student, probe, device, args.batch_size)
    center = source_views.mean(axis=(0, 1))
    support_views = {k: embed_orbit(student, v, device, args.batch_size) for k, v in support.items()}
    supports = {k: v[0] - center for k, v in support_views.items()}
    queries = np.concatenate([embed_orbit(student, test_views[k], device, args.batch_size)
                              for k in target], axis=1) - center
    truth = np.repeat(np.arange(len(target)), args.test_per_class)
    source_views -= center
    evaluations = {}
    for axis, order in ORBIT_ORDERS.items():
        if axis == "vertical" and not both_axes:
            continue
        paired = None
        if args.target_pairs:
            paired = {k: v[order[TEST_VIEWS["composition"]]] - center for k, v in support_views.items()}
        order = list(order)
        evaluations[axis] = evaluate_orbit(source_views[order], supports, queries[order], truth,
                                           target, args, seed, paired=paired)
    return evaluations if both_axes else evaluations["horizontal"]


def target_pair_designs(inputs, labels, counterparts, paired_mask):
    transformed = counterparts[paired_mask]
    augmented = np.concatenate([inputs, transformed]), np.concatenate([labels, labels[paired_mask]])
    constrained = target_pair_constraint(inputs, labels, transformed, paired_mask)
    return {"augment": augmented, "constraint": constrained}


def target_pair_constraint(inputs, labels, transformed, paired_mask):
    original = inputs[paired_mask]
    # The averaged feature retains one labeled observation per original example.
    means = inputs.copy()
    means[paired_mask] = (original + transformed) / 2
    # Keep the same difference precision as independent augmentation.
    differences = (original - transformed) / np.sqrt(2)
    return np.concatenate([means, differences]), np.concatenate([labels, np.zeros_like(labels[paired_mask])])


def orbit_accuracy(prior, inputs, labels, queries, truth, noise):
    correct = predict(prior, inputs, labels, queries, noise).argmax(axis=-1) == truth
    return {"clean": float(correct[0].mean()), "all_views": float(correct.all(axis=0).mean()),
            **{view: float((correct[0] & correct[index]).mean()) for view, index in TEST_VIEWS.items()}}


def active_pair_scores(inputs, labels, prior, residual_model, noise, *, design=None):
    operator, metric = residual_model
    differences = inputs @ operator
    x, y = (inputs, labels) if design is None else design
    precision = np.linalg.solve(prior, np.eye(len(prior))) + x.T @ x / noise ** 2
    weight = np.linalg.solve(precision, x.T @ y / noise ** 2)
    moved = np.linalg.solve(precision, differences.T).T
    leverage = np.sum(differences * moved, axis=1)
    displacement = np.sum((moved @ metric) * moved, axis=1)
    violation = np.sum((differences @ weight) ** 2, axis=1)
    update = violation * displacement / (noise ** 2 + leverage) ** 2
    clean_covariance = np.linalg.solve(precision, inputs.T).T
    clean_leverage = np.sum(inputs * clean_covariance, axis=1)
    return {"expected_update": update, "action_leverage": leverage, "clean_leverage": clean_leverage}


def full_pair_update_scores(inputs, labels, prior, metric, noise, predicted, observed):
    precision = np.linalg.solve(prior, np.eye(len(prior))) + inputs.T @ inputs / noise ** 2
    weight = np.linalg.solve(precision, inputs.T @ labels / noise ** 2)
    targets = np.stack((labels, labels, np.zeros_like(labels)), axis=1)
    # Downdate the original row, then add the pair mean and difference.
    signs = np.diag([-1., 1., 1.]) * noise ** 2
    scores = {}
    for name, counterparts in (("full_predicted", predicted), ("full_observed", observed)):
        means = (inputs + counterparts) / 2
        differences = (inputs - counterparts) / np.sqrt(2)
        directions = np.stack((inputs, means, differences), axis=-1)
        rhs = directions.transpose(1, 0, 2).reshape(len(prior), -1)
        root = np.linalg.solve(precision, rhs).reshape(len(prior), len(inputs), 3).transpose(1, 0, 2)
        dual = directions.swapaxes(1, 2)
        change = root @ np.linalg.solve(signs + dual @ root, targets - dual @ weight)
        scores[name] = np.sum(change * (metric @ change), axis=(1, 2))
    return scores


def compare_active_pairs(inputs, labels, counterparts, priors, residual_model, queries, truth, args, n):
    metrics = {}
    if args.sequential_target_pairs or args.exact_pair_update:
        predicted = inputs - np.sqrt(2) * (inputs @ residual_model[0])
    for key in TARGET_PAIR_PRIORS[1:]:
        prior = priors[key]
        orders = active_pair_scores(inputs, labels, prior, residual_model, args.noise)
        if args.exact_pair_update:
            orders.update(full_pair_update_scores(
                inputs, labels, prior, residual_model[1], args.noise, predicted, counterparts
            ))
        ranks = {policy: np.argsort(np.argsort(-scores.reshape(-1, n), axis=1, kind="stable"), axis=1).ravel()
                 for policy, scores in orders.items()}
        static_rank = ranks["expected_update"]
        counts = {1, min(3, n), min(5, n)}
        if args.sequential_target_pairs:
            counts.add(min(2, n))
        counts = sorted(counts)
        for policy, rank in ranks.items():
            for count in counts:
                mask = rank < count
                x_pair, y_pair = target_pair_constraint(inputs, labels, counterparts[mask], mask)
                result = orbit_accuracy(prior, x_pair, y_pair, queries, truth, args.noise)
                if policy in ("full_predicted", "full_observed"):
                    result["static_overlap"] = float(np.mean(static_rank[mask] < count))
                metrics[f"{key}_{policy}_{count}"] = result
        if args.sequential_target_pairs:
            for policy, feedback in (("actual_feedback", counterparts), ("predicted_feedback", predicted)):
                masks = sequential_pair_masks(inputs, labels, feedback, prior, residual_model,
                                              args.noise, n, counts, orders["expected_update"])
                for count, mask in masks.items():
                    x_pair, y_pair = target_pair_constraint(inputs, labels, counterparts[mask], mask)
                    result = orbit_accuracy(prior, x_pair, y_pair, queries, truth, args.noise)
                    result["static_overlap"] = float(np.mean(static_rank[mask] < count))
                    metrics[f"{key}_{policy}_{count}"] = result
    return metrics


def sequential_pair_masks(inputs, labels, feedback, prior, residual_model, noise, n, counts, initial_scores):
    chosen = np.zeros(len(inputs), dtype=bool)
    starts = np.arange(0, len(inputs), n)
    scores = initial_scores
    masks = {}
    for count in range(1, max(counts) + 1):
        available = np.where(chosen, -np.inf, scores).reshape(-1, n)
        chosen[starts + available.argmax(axis=1)] = True
        if count in counts:
            masks[count] = chosen.copy()
        if count < max(counts):
            design = target_pair_constraint(inputs, labels, feedback[chosen], chosen)
            scores = active_pair_scores(inputs, labels, prior, residual_model, noise, design=design)["expected_update"]
    return masks


def evaluate_orbit(source_views, supports, queries, truth, target, args, seed, *, paired=None):
    priors, diagnostics, residual_model = source_priors(
        source_views, args.budget, seed, args.generator_controls, args.operator_composition, args.path_composition
    )
    if args.compare_teacher_views:
        difference = (source_views - source_views[list(REFLECTION)]) / 2
        diagnostics["reflection_feature_energy"] = float(np.mean(difference ** 2)
                                                         / max(float(np.mean(source_views ** 2)), 1e-12))
    rows = {str(n): [] for n in args.shots}
    rng = np.random.default_rng(seed + 422)
    for _ in range(args.episodes):
        chosen = {k: rng.choice(len(supports[k]), max(args.shots), replace=False) for k in target}
        for n in args.shots:
            inputs = np.concatenate([supports[k][chosen[k][:n]] for k in target])
            labels = np.repeat(np.eye(len(target)), n, axis=0) - 1 / len(target)
            methods = {key: orbit_accuracy(prior, inputs, labels, queries, truth, args.noise)
                       for key, prior in priors.items()}
            if paired is not None:
                counterparts = np.concatenate([paired[k][chosen[k][:n]] for k in target])
                counts = {1, min(5, n), n}
                if args.active_target_pairs:
                    counts.add(min(3, n))
                    if args.sequential_target_pairs:
                        counts.add(min(2, n))
                    methods.update(compare_active_pairs(
                        inputs, labels, counterparts, priors, residual_model, queries, truth, args, n
                    ))
                for count in sorted(counts):
                    mask = np.tile(np.arange(n) < count, len(target))
                    for rule, (x_pair, y_pair) in target_pair_designs(inputs, labels, counterparts, mask).items():
                        for key in TARGET_PAIR_PRIORS:
                            methods[f"{key}_{rule}_{count}"] = orbit_accuracy(
                                priors[key], x_pair, y_pair, queries, truth, args.noise
                            )
            rows[str(n)].append(methods)
    return {"latent_width": source_views.shape[-1], "diagnostics": diagnostics, "shots": rows}


def match_teacher_basis(basis, odd, ranks, seed):
    rng = np.random.default_rng(seed)
    blocks = []
    for is_odd, rank in zip((False, True), ranks, strict=True):
        block = basis[:, odd == is_odd]
        axes, _ = np.linalg.qr(rng.normal(size=(block.shape[1], rank)))
        blocks.append(block @ torch.as_tensor(axes, device=basis.device, dtype=basis.dtype))
    return torch.cat(blocks, dim=1)


def compare_reflection_teachers(x, probe, support, test_views, source, target, args, seed, device):
    views = {"horizontal": reflect, "vertical": vertical_reflect}
    teachers, bases, ranks_by_axis = {}, {}, {}
    for axis, view in views.items():
        torch.manual_seed(seed)
        teacher = Teacher(len(source), view=view).to(device).eval()
        mean, basis, energies = discover_basis(teacher, probe[::2], args)
        odd = energies > .5
        teachers[axis] = teacher
        bases[axis] = (mean, basis, odd)
        ranks_by_axis[axis] = (int((~odd).sum()), int(odd.sum()))
    ranks = tuple(map(min, zip(*ranks_by_axis.values(), strict=True)))
    measurements = {}
    for axis, teacher in teachers.items():
        mean, basis, odd = bases[axis]
        basis = match_teacher_basis(basis, odd, ranks, seed + 727)
        student = fit_student(x, teacher, basis, mean, args, seed + 100, training_views=tuple(views.values()))
        measurements[axis] = measure_orbit_prior(
            student, probe[1::2], support, test_views, target, args, seed, device, both_axes=True
        )
    return {"teacher_views": measurements, "teacher_ranks": {"matched": ranks, **ranks_by_axis}}


def run(train, test, args, seed, device, traffic_paths=None):
    source, target = class_split(seed, True)
    x, probe, support, test_views = data_partitions(train, test, args, seed, device, source, target)
    if traffic_paths is not None:
        x, probe = traffic_source(traffic_paths, args, seed, device)
        source = TRAFFIC_SOURCE
    result = {"seed": seed, "source_classes": source, "target_classes": target}
    if args.compare_teacher_views:
        result.update(compare_reflection_teachers(x, probe, support, test_views, source, target, args, seed, device))
        print(seed, "completed", flush=True)
        return result
    training_labels = torch.arange(len(source), device=device).repeat_interleave(args.source_examples)
    teacher = fit_teacher(x, training_labels, args, seed, classes=len(source))
    mean, basis, energies = discover_basis(teacher, probe[::2], args)
    if args.compare_teacher_training:
        torch.manual_seed(seed)
        frozen = Teacher(len(source)).to(device).eval()
        frozen_mean, frozen_basis, frozen_energies = discover_basis(frozen, probe[::2], args)
        odd, frozen_odd = energies > .5, frozen_energies > .5
        learned_ranks = (int((~odd).sum()), int(odd.sum()))
        frozen_ranks = (int((~frozen_odd).sum()), int(frozen_odd.sum()))
        ranks = tuple(min(a, b) for a, b in zip(learned_ranks, frozen_ranks, strict=True))
        basis = match_teacher_basis(basis, odd, ranks, seed + 727)
        frozen_basis = match_teacher_basis(frozen_basis, frozen_odd, ranks, seed + 727)
        frozen_student = fit_student(x, frozen, frozen_basis, frozen_mean, args, seed + 100)
        result["initialized_teacher"] = measure_orbit_prior(
            frozen_student, probe[1::2], support, test_views, target, args, seed, device
        )
        result["teacher_ranks"] = {"matched": ranks, "learned": learned_ranks, "initialized": frozen_ranks}
    student = fit_student(x, teacher, basis, mean, args, seed + 100)
    result.update(measure_orbit_prior(student, probe[1::2], support, test_views, target, args, seed, device))
    print(seed, "completed", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/tmp/ibt-cifar-data"))
    parser.add_argument("--source-domain", choices=("cifar", "gtsrb"), default="cifar")
    parser.add_argument("--gtsrb-data", type=Path, default=Path("/tmp/ibt-gtsrb-data"))
    parser.add_argument("--output", type=Path, default=HERE / "results/generator_budget/metrics.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(701, 717)))
    parser.add_argument("--shots", type=int, nargs="+", default=[5, 20, 60])
    parser.add_argument("--source-examples", type=int, default=900)
    parser.add_argument("--probe-examples", type=int, default=128)
    parser.add_argument("--teacher-epochs", type=int, default=26)
    parser.add_argument("--student-epochs", type=int, default=22)
    parser.add_argument("--target-pool", type=int, default=600)
    parser.add_argument("--test-per-class", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--budget", type=float, default=.25)
    parser.add_argument("--generator-controls", action="store_true",
                        help="include full-orbit priors estimated from each action alone")
    parser.add_argument("--operator-composition", action="store_true",
                        help="infer composition moments from disjoint generator pairs")
    parser.add_argument("--path-composition", action="store_true",
                        help="compare two equivalent paths estimated from disjoint generator pairs")
    parser.add_argument("--target-pairs", action="store_true",
                        help="measure the value of direct composition pairs in labeled target support")
    parser.add_argument("--active-target-pairs", action="store_true",
                        help="select composition pairs using only clean support and source operators")
    parser.add_argument("--sequential-target-pairs", action="store_true",
                        help="compare real and predicted feedback after revealing each batch of pairs")
    parser.add_argument("--exact-pair-update", action="store_true",
                        help="score pair acquisition by the full change in the posterior mean")
    teacher_comparison = parser.add_mutually_exclusive_group()
    teacher_comparison.add_argument("--compare-teacher-training", action="store_true",
                                    help="compare trained and initialized teachers with matched parity ranks")
    teacher_comparison.add_argument("--compare-teacher-views", action="store_true",
                                    help="compare horizontal and vertical parity with the same training views")
    parser.add_argument("--noise", type=float, default=.45)
    args = parser.parse_args()
    if args.sequential_target_pairs or args.exact_pair_update:
        args.active_target_pairs = True
    if args.active_target_pairs:
        args.target_pairs = True
    if not 0 < args.budget < 1 or args.noise <= 0:
        parser.error("need a variance budget in (0, 1) and positive noise")
    if (args.operator_composition or args.path_composition) and args.probe_examples < 2:
        parser.error("operator composition needs at least two probe examples per source class")
    if args.compare_teacher_views and not args.path_composition:
        parser.error("teacher view comparison needs --path-composition")
    if args.target_pairs and not args.path_composition:
        parser.error("target pair comparison needs --path-composition")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    train, test = (CIFAR10(args.data, train=flag, download=True) for flag in (True, False))
    traffic_paths = (paths_by_class(GTSRB(args.gtsrb_data, split="train", download=True))
                     if args.source_domain == "gtsrb" else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "runs": []}
    start = time.monotonic()
    for seed in args.seeds:
        output["runs"].append(run(train, test, args, seed, device, traffic_paths))
        output["elapsed_seconds"] = time.monotonic() - start
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(output, allow_nan=False) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
