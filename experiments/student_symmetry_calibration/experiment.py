"""Calibrate a teacher-derived reflection prior using the student's orbit errors."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR10

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from orbit_spectral_prior.experiment import balanced_indices
from reflection_evidence.experiment import discover_basis, embed, fit_student, fit_teacher

SOURCE = (0, 2, 6, 8)
TARGET = (1, 3, 4, 5, 7, 9)


def as_float(x, device):
    return x.to(device, dtype=torch.float32) / 255


def data_partitions(train, test, args, seed, device):
    rng = torch.Generator().manual_seed(seed)
    train_images = torch.from_numpy(train.data).permute(0, 3, 1, 2)
    test_images = torch.from_numpy(test.data).permute(0, 3, 1, 2)
    train_labels = torch.tensor(train.targets)
    test_labels = torch.tensor(test.targets)
    idx = balanced_indices(train_labels, SOURCE, args.source_examples + args.probe_examples, rng)
    source = as_float(train_images[idx[:, :args.source_examples].reshape(-1)], device)
    probe = as_float(train_images[idx[:, args.source_examples:].reshape(-1)], device)
    target_idx = balanced_indices(train_labels, TARGET, args.target_pool, rng)
    test_idx = balanced_indices(test_labels, TARGET, args.test_per_class, rng)
    target_pool = {label: train_images[row] for label, row in zip(TARGET, target_idx, strict=True)}
    target_test = {label: test_images[row] for label, row in zip(TARGET, test_idx, strict=True)}
    return source, probe, target_pool, target_test


def trace_normalized(matrix):
    return matrix / np.trace(matrix)


def prior_options(probe, odd, args, rng):
    clean, flipped = probe.astype(np.float64)
    delta = (flipped - clean) / 2
    within = delta.T @ delta / len(delta)
    width = within.shape[0]
    identity = np.eye(width)
    within /= max(np.trace(within) / width, 1e-8)
    empirical = np.linalg.solve(identity + args.strength * within, identity)
    gate = np.diag(np.where(odd, np.sqrt(args.odd_ratio), 1.0))
    basis, _ = np.linalg.qr(rng.normal(size=(width, width)))
    prior = {
        "isotropic": identity,
        "teacher": gate @ gate,
        "student": empirical,
        "joint": gate @ empirical @ gate,
        "shuffled_student": gate @ basis @ empirical @ basis.T @ gate,
        "two_view": identity,
    }
    if args.precision_strengths:
        centers = (clean + flipped) / 2
        centered = centers - centers.mean(axis=0)
        between = centered.T @ centered / len(centered)
        between /= max(np.trace(between) / width, 1e-8)
        prior["between"] = np.linalg.solve(identity + args.strength * between, identity)
        for strength in args.precision_strengths:
            covariance = np.linalg.solve(identity + strength * within, identity)
            prior[f"orbit_strength_{strength:g}"] = covariance
            prior[f"rotated_strength_{strength:g}"] = basis @ covariance @ basis.T
    return {key: trace_normalized(value) for key, value in prior.items()}, np.linalg.eigvalsh(within)


def coordinate_priors(h, strength, rng):
    total = h.reshape(-1, h.shape[-1])
    total_covariance = total.T @ total / len(total)
    residual = (h[0] - h[1]) / 2
    orbit_covariance = residual.T @ residual / len(residual)
    width = h.shape[-1]
    identity = np.eye(width)
    whitener = np.linalg.solve(np.linalg.cholesky(total_covariance).T, identity)
    relative_orbit = whitener.T @ orbit_covariance @ whitener
    ordinary = orbit_covariance / max(np.trace(orbit_covariance) / width, 1e-8)
    natural = np.linalg.solve(identity + strength * relative_orbit, identity)
    rotation, _ = np.linalg.qr(rng.normal(size=(width, width)))
    normalized = trace_normalized(natural)
    return {
        "ordinary_iso": trace_normalized(identity),
        "ordinary_orbit": trace_normalized(np.linalg.solve(identity + strength * ordinary, identity)),
        "natural_iso": whitener @ trace_normalized(identity) @ whitener.T,
        "natural_orbit": whitener @ normalized @ whitener.T,
        "natural_rotated": whitener @ rotation @ normalized @ rotation.T @ whitener.T,
    }


def gauge_evaluation(probe, support, query, args, seed):
    width = probe.shape[-1]
    rng = np.random.default_rng(seed + 253)
    left, _ = np.linalg.qr(rng.normal(size=(width, width)))
    right, _ = np.linalg.qr(rng.normal(size=(width, width)))
    identity = np.eye(width)
    change = (left * np.geomspace(0.25, 4.0, width)) @ right
    mean = probe.mean(axis=(0, 1), dtype=np.float64)
    metrics_by_basis = {}
    kernels = {}
    for basis_name, basis in (("native", identity), ("mixed", change)):
        h = (probe - mean) @ basis
        prior = coordinate_priors(h, args.strength, np.random.default_rng(seed + 221))
        if basis_name == "native":
            rotated_native = prior["natural_rotated"]
        else:
            inverse = np.linalg.solve(basis, identity)
            prior["natural_rotated"] = inverse @ rotated_native @ inverse.T
        centered_support = {k: (v - mean) @ basis for k, v in support.items()}
        centered_query = {k: (v - mean) @ basis for k, v in query.items()}
        metrics_by_basis[basis_name] = evaluate(
            prior, centered_support, centered_query, args, np.random.default_rng(seed + 422)
        )
        fixed_query = np.concatenate([centered_query[k][0, :12] for k in TARGET])
        kernels[basis_name] = {name: (fixed_query @ matrix) @ fixed_query.T
                               for name, matrix in prior.items()}
    relative_errors = {}
    for name, matrix in kernels["native"].items():
        relative_errors[name] = float(np.linalg.norm(matrix - kernels["mixed"][name]) /
                                      max(np.linalg.norm(matrix), 1e-12))
    return metrics_by_basis, {"condition_number": float(np.linalg.cond(change)),
                              "test_kernel_relative_error": relative_errors}


def predict(prior, train, labels, test, noise):
    precision = np.linalg.solve(prior, np.eye(len(prior)))
    covariance_inverse = precision + train.T @ train / noise ** 2
    weight = np.linalg.solve(covariance_inverse, train.T @ labels / noise ** 2)
    return test @ weight


def accuracy(predictions, truth):
    clean, mirror = predictions.argmax(axis=-1)
    return {
        "clean_accuracy": float(np.mean(clean == truth)),
        "mirror_accuracy": float(np.mean(mirror == truth)),
        "matched_view_accuracy": float(np.mean((clean == truth) & (mirror == truth))),
        "agreement": float(np.mean(clean == mirror)),
    }


def evaluate(priors, supports, queries, args, rng):
    query_features = np.concatenate([queries[label] for label in TARGET], axis=1)
    query_labels = np.concatenate([np.full(len(queries[label][0]), index)
                                   for index, label in enumerate(TARGET)])
    query_sym = query_features.mean(axis=0)
    measurements = {str(count): [] for count in args.shots}
    for _ in range(args.episodes):
        selected = {label: rng.choice(len(supports[label][0]), max(args.shots), replace=False)
                    for label in TARGET}
        for count in args.shots:
            x = np.concatenate([supports[label][:, selected[label][:count]] for label in TARGET],
                               axis=1).astype(np.float64)
            targets = np.repeat(np.eye(len(TARGET)), count, axis=0) - 1 / len(TARGET)
            row = {}
            for key, prior in priors.items():
                if key == "two_view":
                    prediction = predict(prior, x.mean(axis=0), targets, query_sym, args.noise)
                    scores = np.stack((prediction, prediction))
                else:
                    scores = predict(prior, x[0], targets, query_features, args.noise)
                row[key] = accuracy(scores, query_labels)
            measurements[str(count)].append(row)
    return {
        count: {method: {metric: float(np.mean([r[method][metric] for r in rows]))
                         for metric in scores}
                for method, scores in rows[0].items()}
        for count, rows in measurements.items()
    }


def run(train, test, args, seed, device):
    rng = np.random.default_rng(seed)
    source, probe, target, target_test = data_partitions(train, test, args, seed, device)
    labels = torch.arange(len(SOURCE), device=device).repeat_interleave(args.source_examples)
    teacher = fit_teacher(source, labels, args, seed, len(SOURCE))
    mean, basis, energies = discover_basis(teacher, probe, args)
    student = fit_student(source, teacher, basis, mean, args, seed + 100)
    odd = energies.cpu().numpy() > 0.5
    with torch.inference_mode():
        pred = torch.cat([teacher(batch).argmax(dim=-1) for batch in probe.split(args.batch_size)])
        truth = torch.arange(len(SOURCE), device=device).repeat_interleave(args.probe_examples)
        accuracy_source = pred.eq(truth).float().mean().item()
        h = torch.cat([student(batch) for batch in probe.split(args.batch_size)]).cpu().numpy()
        reflected = torch.cat([student(batch.flip(-1)) for batch in probe.split(args.batch_size)])
        probe_z = np.stack((h, reflected.cpu().numpy()))
    priors, spectrum = prior_options(probe_z, odd, args, rng)
    supports = embed(student, target, args, device)
    queries = embed(student, target_test, args, device)
    main = evaluate(priors, supports, queries, args, rng)
    if args.gauge_check:
        comparison, coordinate_diagnostics = gauge_evaluation(probe_z, supports, queries, args, seed)
    else:
        comparison, coordinate_diagnostics = {}, {}
    print(seed, round(accuracy_source, 4), len(odd), main[str(max(args.shots))]["joint"], flush=True)
    return {"seed": seed, "teacher_probe_accuracy": accuracy_source, "latent_width": len(odd),
            "teacher_odd_count": int(odd.sum()), "student_orbit_spectrum": spectrum.tolist(),
            "source_even_flip_mse": float(((probe_z[0, :, ~odd] - probe_z[1, :, ~odd]) ** 2).mean()),
            "shots": main, "coordinate_comparison": comparison,
            "coordinate_diagnostics": coordinate_diagnostics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/tmp/ibt-cifar-data"))
    parser.add_argument("--output", type=Path, default=HERE / "results/metrics.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--shots", type=int, nargs="+", default=[5, 20, 60])
    parser.add_argument("--teacher-epochs", type=int, default=26)
    parser.add_argument("--student-epochs", type=int, default=22)
    parser.add_argument("--source-examples", type=int, default=1400)
    parser.add_argument("--probe-examples", type=int, default=320)
    parser.add_argument("--target-pool", type=int, default=600)
    parser.add_argument("--test-per-class", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--noise", type=float, default=0.45)
    parser.add_argument("--strength", type=float, default=4.0)
    parser.add_argument("--precision-strengths", type=float, nargs="+")
    parser.add_argument("--gauge-check", action="store_true")
    parser.add_argument("--odd-ratio", type=float, default=0.02)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    train = CIFAR10(args.data, train=True, download=True)
    test = CIFAR10(args.data, train=False, download=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "torch_version": torch.__version__, "source_classes": SOURCE, "target_classes": TARGET,
            "runs": []}
    start = time.monotonic()
    for seed in args.seeds:
        data["runs"].append(run(train, test, args, seed, device))
        data["elapsed_seconds"] = time.monotonic() - start
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
