"""Transfer two latent reflections and infer their commuting actions on six classes."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from reflection_label_matching.experiment import edge_evidence, gaussian_head, matching_posterior

SIZE = 6
WIDTH = 12
SIGNS = np.array(((1, 1), (-1, 1), (1, -1), (-1, -1)))


def involutions(indices, result):
    if not indices:
        yield result
        return
    a, *rest = indices
    yield from involutions(rest, result)
    for b in rest:
        mapped = result.copy()
        mapped[a], mapped[b] = b, a
        yield from involutions([i for i in rest if i != b], mapped)


def projectors(p, q):
    identity = np.eye(p.shape[-1])
    return np.stack([
        ((identity + a * p) @ (identity + b * q) + (identity + b * q) @ (identity + a * p)) / 8
        for a, b in SIGNS
    ], axis=-3)


def candidate_bank():
    identity = np.eye(SIZE)
    actions = np.stack(list(involutions(list(range(SIZE)), np.arange(SIZE))))
    first, second = [], []
    for p in actions:
        valid = np.all(p[actions] == actions[:, p], axis=1)
        first.extend([p] * int(valid.sum()))
        second.extend(actions[valid])
    first, second = np.array(first), np.array(second)
    projector = projectors(identity[first], identity[second])
    moved = ((first != np.arange(SIZE)).sum(axis=1) + (second != np.arange(SIZE)).sum(axis=1)) / 2
    return first, second, projector.reshape(len(first), -1), -moved * np.log(SIZE - 1)


def covariance_basis(rng):
    basis, _ = np.linalg.qr(rng.normal(size=(WIDTH, WIDTH)))
    latent = np.stack([basis[:, k * 3:(k + 1) * 3] @ basis[:, k * 3:(k + 1) * 3].T
                       for k in range(len(SIGNS))])
    action = np.stack([np.einsum("s,sij->ij", SIGNS[:, axis], latent) for axis in range(2)])
    return latent, action


def discover(x, inverse_pairs):
    candidates = []
    for y in inverse_pairs:
        u, _, vh = np.linalg.svd(x.T @ y)
        q = u @ vh
        candidates.append((q + q.T) / 2)
    values, axes = np.linalg.eigh(candidates[0] + 2 * candidates[1])
    assignment = np.argmin(np.abs(values[:, None] - SIGNS @ [1, 2]), axis=1)
    return np.stack([axes[:, assignment == k] @ axes[:, assignment == k].T for k in range(len(SIGNS))])


def encoder(rng, samples):
    transform, _ = np.linalg.qr(rng.normal(size=(WIDTH, WIDTH)))
    random = rng.normal(size=(WIDTH, 72)) / np.sqrt(WIDTH)
    offset = rng.uniform(-np.pi, np.pi, size=72)

    def raw(latent):
        x = np.tanh(latent @ transform)
        return np.concatenate((x, np.cos(x @ random + offset)), axis=1)

    train = rng.normal(size=(samples, WIDTH))
    design = raw(train)
    w = np.linalg.solve(design.T @ design + 0.05 * np.eye(design.shape[1]), design.T @ train)

    def predict(latent):
        return raw(latent) @ w

    return predict


def kernels(projector):
    priors = 0.98 * projector + 0.02 * np.eye(WIDTH)[None] / 4
    return priors / np.trace(priors, axis1=1, axis2=2)[:, None, None]


def truth_actions(rng):
    p = np.array([1, 0, 3, 2, 4, 5])
    q = np.array([2, 3, 0, 1, 4, 5])
    mode = rng.integers(1, 4)
    if mode & 1:
        p[4:6] = [5, 4]
    if mode & 2:
        q[4:6] = [5, 4]
    permutation = rng.permutation(SIZE)
    inverse = np.argsort(permutation)
    return inverse[p[permutation]], inverse[q[permutation]]


def observations(rng, p, q, action, shots, queries, noise):
    latent = (np.eye(WIDTH), action[0], action[1], action[0] @ action[1])
    permutations = (np.arange(SIZE), p, q, p[q])
    means = np.zeros((SIZE, WIDTH))
    remaining = set(range(SIZE))
    while remaining:
        representative = min(remaining)
        prototype = rng.normal(size=WIDTH) * 0.65
        orbit = set()
        for transform, label_map in zip(latent, permutations, strict=True):
            label = label_map[representative]
            means[label] += prototype @ transform
            orbit.add(label)
        normalization = len(orbit) / len(latent)
        means[list(orbit)] *= normalization
        remaining -= orbit
    nuisance = rng.normal(size=WIDTH) * 0.65
    train = means[:, None] + nuisance + rng.normal(scale=noise, size=(SIZE, shots, WIDTH))
    test = means[:, None] + nuisance + rng.normal(scale=noise, size=(SIZE, queries, WIDTH))
    return train, test.reshape(-1, WIDTH) @ latent[-1]


def estimates(support, labels, query, priors, bank, noise):
    first, second, candidates, penalties = bank
    heads = [gaussian_head(k, support, labels, query, noise) for k in priors]
    scores = np.stack([labels.T @ head[1] + head[2] * np.eye(SIZE) for head in heads])
    logits = -(candidates @ scores.ravel()) / 2 + penalties
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    posterior = (probabilities @ candidates).reshape(len(SIGNS), SIZE, SIZE)
    best = int(np.argmax(logits))
    combined = {"joint": (posterior, first[best], second[best])}

    individual = []
    for axis in range(2):
        pos = gaussian_head(priors[SIGNS[:, axis] == 1].mean(axis=0), support, labels, query, noise)
        neg = gaussian_head(priors[SIGNS[:, axis] == -1].mean(axis=0), support, labels, query, noise)
        w = edge_evidence(pos, neg, labels) - np.log(SIZE - 1)
        individual.append(matching_posterior(w))
    (p_matrix, p), (q_matrix, q) = individual
    combined["independent"] = (projectors(p_matrix, q_matrix), p, q)
    return np.stack([h[0] for h in heads]), combined


def score(p, q, predicted_p, predicted_q, query_scores, truth):
    prediction = query_scores.argmax(axis=1)
    return {
        "shifted_accuracy": float(np.mean(prediction == truth)),
        "generator_accuracy": float((np.mean(p == predicted_p) + np.mean(q == predicted_q)) / 2),
        "composition_accuracy": float(np.mean(p[q] == predicted_p[predicted_q])),
        "noncommutation": float(np.mean(predicted_p[predicted_q] != predicted_q[predicted_p])),
        "exact_action": float(np.array_equal(p, predicted_p) and np.array_equal(q, predicted_q)),
    }


def run(args, seed, bank):
    rng = np.random.default_rng(seed)
    oracle, action = covariance_basis(rng)
    encode = encoder(rng, args.encoder_samples)
    source = rng.normal(size=(args.source_pairs, WIDTH))
    paired = source @ action
    wrong = rng.normal(size=paired.shape)
    noise_u = rng.uniform(size=paired.shape[:2])
    output = {"seed": seed, "corruption": {}}
    for rate in args.corruption:
        observed = np.where((noise_u < rate)[..., None], wrong, paired)
        learned = discover(source, observed)
        priors = kernels(learned)
        task_rng = np.random.default_rng(seed + 1000)
        shots = {str(n): [] for n in args.shots}
        for _ in range(args.tasks):
            p, q = truth_actions(task_rng)
            support, query = observations(task_rng, p, q, action, max(args.shots), args.queries,
                                          args.sample_noise)
            identity = np.eye(SIZE)
            x_query = encode(query)
            truth = np.repeat(p[q], args.queries)
            for n in args.shots:
                x = encode(support[:, :n].reshape(-1, WIDTH))
                y = np.repeat(identity, n, axis=0) - 1 / SIZE
                heads, decisions = estimates(x, y, x_query, priors, bank, args.label_noise)
                decisions["oracle"] = (projectors(identity[p], identity[q]), p, q)
                iso_pred = gaussian_head(np.eye(WIDTH) / WIDTH, x, y, x_query, args.label_noise)[0]
                row = {"isotropic": {"shifted_accuracy": float(np.mean(iso_pred.argmax(axis=-1) == truth))}}
                for name, (projector, p_hat, q_hat) in decisions.items():
                    mu = np.einsum("qvi,qij->vj", heads, projector)
                    row[name] = score(p, q, p_hat, q_hat, mu, truth)
                shots[str(n)].append(row)
        summary = {
            n: {m: {key: float(np.mean([row[m][key] for row in rows])) for key in metrics}
                for m, metrics in rows[0].items()} for n, rows in shots.items()
        }
        output["corruption"][str(rate)] = {"projector_error": float(np.linalg.norm(learned - oracle)),
                                           "shots": summary}
    print(seed, output["corruption"][str(args.corruption[-1])]["shots"][str(max(args.shots))]["joint"],
          flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results/metrics.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 17)))
    parser.add_argument("--corruption", type=float, nargs="+", default=[0.0, 0.55])
    parser.add_argument("--shots", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--queries", type=int, default=80)
    parser.add_argument("--source-pairs", type=int, default=400)
    parser.add_argument("--encoder-samples", type=int, default=2000)
    parser.add_argument("--sample-noise", type=float, default=0.45)
    parser.add_argument("--label-noise", type=float, default=0.4)
    args = parser.parse_args()
    bank = candidate_bank()
    metrics = {"config": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
               "candidate_count": len(bank[0]), "runs": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    for seed in args.seeds:
        metrics["runs"].append(run(args, seed, bank))
        metrics["elapsed_seconds"] = time.monotonic() - start
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(metrics, indent=2) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
