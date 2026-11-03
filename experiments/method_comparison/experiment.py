"""Compare all transfer methods on the same binary pattern detection tasks."""

import copy
import math
from pathlib import Path
import sys
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "_pattern_transfer"))

from common import fit, freeze, make_model, make_parser, pattern_labels, prepare, record, save, train_teacher, validate
from experiments.coordinate_sum.experiment import shuffle_coordinates
from experiments.hidden_prior.experiment import distill_features, parameter_prior
from experiments.learned_augmentation.experiment import consistency_penalty, learn_policy, policy_metrics


def timed(function, args):
    # CUDA work must finish before reading the wall clock.
    if torch.device(args.device).type == "cuda":
        torch.cuda.synchronize(args.device)
    started = time.perf_counter()
    result = function()
    if torch.device(args.device).type == "cuda":
        torch.cuda.synchronize(args.device)
    return result, time.perf_counter() - started


def shuffle_penalty(seed):
    generator = torch.Generator().manual_seed(seed)

    def penalty(model, x, logits):
        # Reuse the original CPU permutation sampler, then move inputs back.
        shuffled_x = shuffle_coordinates(x.cpu(), generator).to(x.device)
        return (logits.sigmoid() - model(shuffled_x).sigmoid()).abs().mean()
    return penalty


@torch.no_grad()
def shuffle_metrics(model, task, args, seed):
    model.eval()
    x, y, _ = task["test"]
    generator = torch.Generator().manual_seed(seed)
    shuffled_x = shuffle_coordinates(x, generator)
    actual_labels = pattern_labels(shuffled_x, task["pattern"])
    changed = (shuffled_x != x).any(dim=1)
    logits = model(shuffled_x.to(args.device)).cpu()
    original_logits = model(x.to(args.device)).cpu()
    return {
        "label_preservation": (actual_labels == y.bool()).float().mean().item(),
        "changed_fraction": changed.float().mean().item(),
        "accuracy_on_policy_shifts": ((logits >= 0) == actual_labels).float().mean().item(),
        "policy_invariance_mae": (logits.sigmoid() - original_logits.sigmoid()).abs().mean().item(),
    }


def method_names(teachers):
    names = ["baseline", "known_shift"]
    names.extend(f"learned_{kind}" for kind in teachers)
    names.extend(f"{kind}_{mode}" for kind in teachers for mode in ("init", "prior"))
    names.append("random_prior")
    names.extend(f"untrained_{kind}_{mode}" for kind in teachers for mode in ("init", "prior"))
    return names + ["shuffle"]


def prepare_transfers(source, args):
    policies = {"known_shift": torch.full((4,), 0.25, device=args.device)}
    encoders = {}
    teachers = []
    for kind in args.teachers:
        (teacher, info), teacher_seconds = timed(lambda: train_teacher(kind, source, args), args)
        (policy, policy_info), policy_seconds = timed(lambda: learn_policy(teacher, source, args), args)
        (encoder, feature_info), feature_seconds = timed(lambda: distill_features(teacher, source, args), args)
        policies[f"learned_{kind}"] = policy
        encoders[kind] = encoder
        # Match the initialization of the trained teacher for this control.
        untrained = freeze(make_model(kind, args, args.seed + 100))
        (encoder, untrained_info), untrained_seconds = timed(
            lambda: distill_features(untrained, source, args), args,
        )
        encoders[f"untrained_{kind}"] = encoder
        info.update({
            "policy": policy_info,
            "feature_distillation": feature_info,
            "untrained_feature_distillation": untrained_info,
            "teacher_seconds": teacher_seconds,
            "policy_seconds": policy_seconds,
            "feature_seconds": feature_seconds,
            "untrained_feature_seconds": untrained_seconds,
        })
        teachers.append(info)
    return policies, encoders, teachers


def compare_task(task, task_id, policies, encoders, payload, args):
    for size in args.train_sizes:
        data = tuple(tensor[:size] for tensor in task["train"])
        for method in payload["methods"]:
            student = make_model("mlp", args, args.seed + 200 + task_id)
            encoder_name = method.rsplit("_", 1)[0]
            if encoder_name in encoders:
                student.encoder.load_state_dict(encoders[encoder_name])
            penalty, weight = None, 0.0
            if method in policies:
                penalty = consistency_penalty(policies[method])
                weight = args.consistency_weight
            elif method == "shuffle":
                penalty = shuffle_penalty(args.seed + 300 + task_id)
                weight = args.consistency_weight
            elif method.endswith("_prior"):
                penalty = parameter_prior(student)
                weight = args.prior_weight
            history, fit_seconds = timed(
                lambda: fit(student, data, task["test"], args, args.epochs, penalty, weight), args,
            )
            diagnostics, diagnostic_seconds = {}, 0.0
            if method in policies:
                diagnostics, diagnostic_seconds = timed(
                    lambda: policy_metrics(student, task, policies[method], args), args,
                )
            elif method == "shuffle":
                diagnostics, diagnostic_seconds = timed(
                    lambda: shuffle_metrics(student, task, args, args.seed + 400 + task_id), args,
                )
            # Give shift and permutation diagnostics the same names in the JSON.
            if diagnostics:
                diagnostics["transformed_accuracy"] = diagnostics.pop("accuracy_on_policy_shifts")
                diagnostics["probability_gap"] = diagnostics.pop("policy_invariance_mae")
            record(
                payload, args, method, task_id, size, history,
                seed=args.seed, target_fit_seconds=fit_seconds,
                diagnostic_seconds=diagnostic_seconds, augmentation_metrics=diagnostics,
            )


def main():
    parser = make_parser(__doc__, "results/method_comparison/metrics.json")
    parser.add_argument("--seeds", nargs="+", type=int, help="Default: seed, seed+1, seed+2")
    parser.add_argument("--feature-epochs", type=int, default=50)
    parser.add_argument("--policy-steps", type=int, default=200)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--prior-weight", type=float, default=1.0)
    args = parser.parse_args()
    validate(parser, args)
    if args.seeds is None:
        args.seeds = [args.seed + offset for offset in range(3)]
    for name in ("seeds", "train_sizes", "teachers"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            parser.error(f"--{name.replace('_', '-')} must contain unique values")
    if args.feature_epochs <= 0 or args.policy_steps <= 0:
        parser.error("--feature-epochs and --policy-steps must be positive")
    for name in ("entropy_weight", "consistency_weight", "prior_weight"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")

    payload = {
        "experiment": "method_comparison", "config": vars(args),
        "methods": method_names(args.teachers),
        "shift_order": ["right_fill_-1", "right_fill_+1", "left_fill_-1", "left_fill_+1"],
        "runs": [], "results": [],
    }
    for seed in args.seeds:
        run_args = copy.copy(args)
        run_args.seed = seed
        (source, targets, info), data_seconds = timed(
            lambda: prepare(run_args, "method_comparison"), run_args,
        )
        policies, encoders, teachers = prepare_transfers(source, run_args)
        payload["runs"].append({
            "seed": seed, "source_pattern": info["source_pattern"],
            "target_patterns": info["target_patterns"],
            "data_seconds": data_seconds, "teachers": teachers,
        })
        save(args, payload)
        for task_id, task in enumerate(targets):
            print(f"seed={seed} task={task_id} pattern={task['pattern']}")
            compare_task(task, task_id, policies, encoders, payload, run_args)
    print(f"Metrics saved to {args.output}")


if __name__ == "__main__":
    main()
