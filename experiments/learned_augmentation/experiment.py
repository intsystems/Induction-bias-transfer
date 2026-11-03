"""Learn a shift policy from each teacher and transfer it to new pattern tasks."""

import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_pattern_transfer"))

from common import (
    fit, make_model, make_parser, pattern_labels, prepare, record,
    save, shift_candidates, train_teacher, validate,
)


@torch.no_grad()
def teacher_costs(teacher, x, args):
    x = x.to(args.device)
    candidates = shift_candidates(x)
    probabilities = teacher(candidates.flatten(0, 1)).sigmoid().view(len(x), 4)
    return (probabilities - teacher(x).sigmoid()[:, None]).square().mean(dim=0)


def learn_policy(teacher, source, args):
    # Only teacher outputs on source training inputs determine the policy.
    costs = teacher_costs(teacher, source["train"][0], args)
    logits = torch.zeros(4, device=args.device, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=0.1)
    history = []
    for step in range(1, args.policy_steps + 1):
        log_p = logits.log_softmax(dim=0)
        p = log_p.exp()
        entropy = -(p * log_p).sum()
        # Entropy discourages collapse to a single transformation.
        loss = (p * costs).sum() - args.entropy_weight * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append({"step": step, "loss": loss.item()})
    p = logits.softmax(dim=0).detach()
    return p, {
        "probabilities": p.cpu().tolist(),
        "source_train_costs": costs.cpu().tolist(),
        "source_test_costs": teacher_costs(teacher, source["test"][0], args).cpu().tolist(),
        "entropy": -(p * p.clamp_min(1e-12).log()).sum().item(),
        "history": history,
    }


def consistency_penalty(probabilities):
    def penalty(model, x, logits):
        candidates = shift_candidates(x)
        predictions = model(candidates.flatten(0, 1)).sigmoid().view(len(x), 4)
        gaps = (predictions - logits.sigmoid()[:, None]).abs()
        # Exact expectation over four shifts avoids sampling noise.
        return (gaps * probabilities).sum(dim=1).mean()
    return penalty


@torch.no_grad()
def policy_metrics(model, task, probabilities, args):
    model.eval()
    x, y, _ = (tensor.to(args.device) for tensor in task["test"])
    candidates = shift_candidates(x)
    actual_labels = pattern_labels(candidates, task["pattern"])
    predictions = model(candidates.flatten(0, 1)).view(len(x), 4)
    same_label = (actual_labels == y[:, None].bool()).float()
    correct = ((predictions >= 0) == actual_labels).float()
    changed = (candidates != x[:, None]).any(dim=-1).float()
    gaps = (predictions.sigmoid() - model(x).sigmoid()[:, None]).abs()
    return {
        "label_preservation": (same_label * probabilities).sum(dim=1).mean().item(),
        "changed_fraction": (changed * probabilities).sum(dim=1).mean().item(),
        "accuracy_on_policy_shifts": (correct * probabilities).sum(dim=1).mean().item(),
        "policy_invariance_mae": (gaps * probabilities).sum(dim=1).mean().item(),
    }


def main():
    parser = make_parser(__doc__, "results/pattern_augmentation/metrics.json")
    parser.add_argument("--policy-steps", type=int, default=200)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    args = parser.parse_args()
    validate(parser, args)
    if args.policy_steps <= 0:
        parser.error("--policy-steps must be positive")
    for name in ("entropy_weight", "consistency_weight"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")

    source, targets, payload = prepare(args, "learned_pattern_augmentation")
    payload["shift_order"] = ["right_fill_-1", "right_fill_+1", "left_fill_-1", "left_fill_+1"]
    policies = {"uniform": torch.full((4,), 0.25, device=args.device)}
    for kind in args.teachers:
        teacher, info = train_teacher(kind, source, args)
        probabilities, policy_info = learn_policy(teacher, source, args)
        info["policy"] = policy_info
        payload["teachers"].append(info)
        policies[f"learned_{kind}"] = probabilities
    save(args, payload)

    for task_id, task in enumerate(targets):
        for size in args.train_sizes:
            data = tuple(tensor[:size] for tensor in task["train"])
            for method in ["baseline", *policies]:
                student = make_model("mlp", args, args.seed + 200 + task_id)
                probabilities = policies.get(method)
                penalty = consistency_penalty(probabilities) if probabilities is not None else None
                history = fit(
                    student, data, task["test"], args, args.epochs,
                    penalty=penalty, weight=args.consistency_weight,
                )
                extra = {}
                if probabilities is not None:
                    extra["policy_metrics"] = policy_metrics(student, task, probabilities, args)
                record(payload, args, method, task_id, size, history, **extra)
    print(f"Metrics saved to {args.output}")


if __name__ == "__main__":
    main()
