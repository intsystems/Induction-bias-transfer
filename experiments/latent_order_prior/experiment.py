"""Discover a cyclic action and adapt its strength with target marginal likelihood."""

import argparse
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DIM = 10
INVARIANT = 4
ORDERS = (4, 5, 6, 7, 8, 9, 10)
MIXTURE = (0.01, 0.08, 0.25, 0.55, 0.85)


def orthogonal(rng, dimensions):
    q, r = np.linalg.qr(rng.normal(size=(dimensions, dimensions)))
    return q * np.sign(np.diag(r))


def action(order, gauge):
    operator = np.eye(DIM)
    for i, harmonic in enumerate((1, 2, 3)):
        angle = harmonic * 2 * np.pi / order
        c, s = np.cos(angle), np.sin(angle)
        start = INVARIANT + 2 * i
        operator[start:start + 2, start:start + 2] = [[c, -s], [s, c]]
    return gauge.T @ operator @ gauge


def procrustes(first, second):
    u, _, vt = np.linalg.svd(first.T @ second)
    return u @ vt


def discover(first, second):
    estimated = procrustes(first, second)
    identity = np.eye(DIM)
    loss = {
        k: np.linalg.norm(np.linalg.matrix_power(estimated, k) - identity, "fro")
        / (k * np.sqrt(DIM)) + 0.002 * k
        for k in ORDERS
    }
    order = min(loss, key=loss.get)
    split = len(first) // 2
    projectors = []
    for part in (slice(None, split), slice(split, None)):
        q = procrustes(first[part], second[part])
        cosines, vectors = np.linalg.eigh((q + q.T) / 2)
        # Invariant frequencies lie closer to 0 than to the next phase.
        selected = vectors[:, cosines > np.cos(np.pi / order)]
        projectors.append(selected @ selected.T)
    # Averaging independently fitted projectors preserves split uncertainty.
    projector = (projectors[0] + projectors[1]) / 2
    return order, projector, loss


def observation(latents, mixing):
    mixed = latents @ mixing
    return np.concatenate((np.tanh(mixed), np.sin(mixed), np.tanh(mixed * np.roll(mixed, 1, axis=1))),
                          axis=1)


def fit_encoder(rng, mixing, count):
    weight = rng.normal(size=(3 * DIM, 96)) / np.sqrt(3 * DIM)
    bias = rng.uniform(-np.pi, np.pi, 96)

    def features(x):
        return np.concatenate((x, np.cos(x @ weight + bias)), axis=1)

    h = rng.normal(size=(count, DIM))
    features_source = features(observation(h, mixing))
    ridge = 0.02 * np.eye(features_source.shape[1])
    adapter = np.linalg.solve(features_source.T @ features_source + ridge, features_source.T @ h)

    def encode(latents):
        return features(observation(latents, mixing)) @ adapter

    return encode


def covariance(projector, broken_fraction):
    identity = np.eye(DIM)
    trace = projector.trace()
    if min(trace, DIM - trace) <= 1e-8:
        return identity / DIM
    return (1 - broken_fraction) * projector / trace + broken_fraction * (identity - projector) / (DIM - trace)


def prior_bank(projector, random_projector, oracle_projector):
    priors = {"isotropic": [np.eye(DIM) / DIM], "fixed_invariant": [covariance(projector, 0.01)]}
    for name, kernel in {
        "adaptive": projector,
        "random_adaptive": random_projector,
        "oracle_adaptive": oracle_projector,
    }.items():
        priors[name] = [covariance(kernel, q) for q in MIXTURE]
    return {name: [(np.linalg.inv(k), np.linalg.slogdet(k)[1]) for k in kernels]
            for name, kernels in priors.items()}


def posterior(priors, gram, cross):
    precision, logdet_prior = priors
    updated = precision + gram
    means = np.linalg.solve(updated, np.broadcast_to(cross[:, None, :], updated.shape[:-1])[..., None])[..., 0]
    improvement = np.einsum("bkd,bd->bk", means, cross)
    log_evidence = (improvement - np.linalg.slogdet(updated)[1] - logdet_prior) / 2
    probs = np.exp(log_evidence - log_evidence.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    return np.einsum("bki,bk->bi", means, probs), probs


def restricted_observations(rng, shape, gauge):
    values = rng.normal(size=(*shape, DIM))
    values[..., INVARIANT] = np.abs(values[..., INVARIANT]) + 0.5
    values[..., INVARIANT + 1] *= 0.35
    return values @ gauge


def task_benchmark(rng, args, encode, operator, oracle, gauge, banks, fraction):
    direction = rng.normal(size=(args.tasks, DIM))
    stable = direction @ oracle
    changing = direction - stable
    stable /= np.linalg.norm(stable, axis=1, keepdims=True)
    changing /= np.linalg.norm(changing, axis=1, keepdims=True)
    coefficients = np.sqrt(1 - fraction) * stable + np.sqrt(fraction) * changing
    training = restricted_observations(rng, (args.tasks, max(args.shots)), gauge)
    noise = rng.normal(scale=args.noise, size=training.shape[:2])
    labels = np.einsum("bnd,bd->bn", training, coefficients) + noise
    training_features = encode(training.reshape(-1, DIM)).reshape(training.shape)

    clean = restricted_observations(rng, (args.test_size,), gauge)
    shifted = clean @ np.linalg.matrix_power(operator, 2)
    clean_features, shifted_features = encode(clean), encode(shifted)
    clean_truth = clean @ coefficients.T
    shifted_truth = shifted @ coefficients.T
    noise_var = args.noise ** 2
    output = {}
    for count in args.shots:
        f = training_features[:, :count]
        y = labels[:, :count]
        gram = (f.transpose(0, 2, 1) @ f)[:, None] / noise_var
        cross = (f.transpose(0, 2, 1) @ y[..., None])[..., 0] / noise_var
        comparisons = {}
        for name, entries in banks.items():
            precision, logdet = map(np.stack, zip(*entries, strict=True))
            beta, probs = posterior((precision, logdet), gram, cross)
            clean_error = ((clean_features @ beta.T - clean_truth) ** 2).mean(axis=0)
            shifted_error = ((shifted_features @ beta.T - shifted_truth) ** 2).mean(axis=0)
            comparisons[name] = {
                "clean_mse": float(clean_error.mean()),
                "shifted_mse": float(shifted_error.mean()),
                "task_shifted_mse": shifted_error.tolist(),
            }
            if len(entries) > 1:
                comparisons[name]["posterior_mean_breaking"] = float((probs @ MIXTURE).mean())
        output[str(count)] = comparisons
    return output


def run(args, seed):
    rng = np.random.default_rng(seed)
    gauge = orthogonal(rng, DIM)
    mixing = orthogonal(rng, DIM)
    actual_order = ORDERS[(seed - 1) % len(ORDERS)]
    operator = action(actual_order, gauge)
    oracle = gauge[:INVARIANT].T @ gauge[:INVARIANT]
    encoded = fit_encoder(rng, mixing, args.encoder_samples)
    diagnostic = rng.normal(size=(args.test_size, DIM))
    reconstruction_mse = float(((encoded(diagnostic) - diagnostic) ** 2).mean())
    source = rng.normal(size=(args.pairs, DIM))
    ideal = source @ operator
    impostors = rng.normal(size=ideal.shape)
    uniform = rng.uniform(size=len(source))
    rotation = orthogonal(rng, DIM)
    result = {"seed": seed, "actual_order": actual_order, "encoder_mse": reconstruction_mse,
              "corruption": {}}
    for error_rate in args.corruption:
        destination = np.where((uniform < error_rate)[:, None], impostors, ideal)
        order, projector, closure = discover(source, destination)
        randomized = rotation @ projector @ rotation.T
        banks = prior_bank(projector, randomized, oracle)
        # Use the same target tasks for each corruption at this seed.
        benchmark_rng = np.random.default_rng(seed + 2000)
        tasks = {str(q): task_benchmark(benchmark_rng, args, encoded, operator, oracle, gauge, banks, q)
                 for q in args.breaking}
        result["corruption"][str(error_rate)] = {
            "recovered_order": order, "closure_loss": closure,
            "projector_error": float(np.linalg.norm(projector - oracle, "fro")),
            "estimated_rank": float(projector.trace()), "breaking": tasks,
        }
    print(seed, actual_order, [arm["recovered_order"] for arm in result["corruption"].values()], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results/metrics.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 15)))
    parser.add_argument("--corruption", type=float, nargs="+", default=[0.0, 0.35, 0.65])
    parser.add_argument("--breaking", type=float, nargs="+", default=[0.0, 0.15, 0.65])
    parser.add_argument("--shots", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--noise", type=float, default=0.20)
    parser.add_argument("--encoder-samples", type=int, default=2800)
    parser.add_argument("--pairs", type=int, default=384)
    parser.add_argument("--test-size", type=int, default=384)
    parser.add_argument("--tasks", type=int, default=25)
    args = parser.parse_args()
    results = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "mixture": MIXTURE, "runs": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        results["runs"].append(run(args, seed))
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(results, indent=2) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
