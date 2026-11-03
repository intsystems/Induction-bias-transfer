"""Use teacher hidden states to learn an MLP initialization and parameter prior."""

import copy
import math
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_pattern_transfer"))

from common import (
    batches, fit, freeze, make_model, make_parser, prepare, record,
    save, train_teacher, validate,
)


def distill_features(teacher, source, args):
    student = make_model("mlp", args, args.seed + 200)
    train_x = source["train"][0].to(args.device)
    test_x = source["test"][0].to(args.device)
    with torch.no_grad():
        train_h = teacher.features(train_x)
        test_h = teacher.features(test_x)
        # Equalize the overall target scale across teacher types.
        scale = train_h.square().mean().sqrt().clamp_min(1e-6)
        train_h, test_h = train_h / scale, test_h / scale
    loader = batches((train_x.cpu(), train_h.cpu()), args, args.seed + 20)
    optimizer = torch.optim.Adam(student.encoder.parameters(), lr=args.lr)
    history = []
    for epoch in range(args.feature_epochs + 1):
        with torch.no_grad():
            history.append({
                "epoch": epoch,
                "train_feature_mse": F.mse_loss(student.features(train_x), train_h).item(),
                "test_feature_mse": F.mse_loss(student.features(test_x), test_h).item(),
            })
        if epoch == args.feature_epochs:
            break
        for x, h in loader:
            loss = F.mse_loss(student.features(x.to(args.device)), h.to(args.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return copy.deepcopy(student.encoder.state_dict()), {
        "feature_scale": scale.item(), "history": history,
    }


def parameter_prior(model):
    center = [parameter.detach().clone() for parameter in model.encoder.parameters()]
    count = sum(parameter.numel() for parameter in center)

    def penalty(student, x, logits):
        # Isotropic Gaussian prior on encoder parameters; the head is free.
        return sum(
            (parameter - reference).square().sum()
            for parameter, reference in zip(student.encoder.parameters(), center)
        ) / count
    return penalty


def main():
    parser = make_parser(__doc__, "results/pattern_hidden_prior/metrics.json")
    parser.add_argument("--feature-epochs", type=int, default=50)
    parser.add_argument("--prior-weight", type=float, default=1.0)
    args = parser.parse_args()
    validate(parser, args)
    if args.feature_epochs <= 0:
        parser.error("--feature-epochs must be positive")
    if not math.isfinite(args.prior_weight) or args.prior_weight < 0:
        parser.error("--prior-weight must be finite and nonnegative")

    source, targets, payload = prepare(args, "hidden_state_initialization_and_prior")
    encoders = {}
    for kind in args.teachers:
        teacher, info = train_teacher(kind, source, args)
        encoder, feature_info = distill_features(teacher, source, args)
        encoders[kind] = encoder
        info["feature_distillation"] = feature_info
        # This control separates learned features from architecture alone.
        untrained = freeze(make_model(kind, args, args.seed + 100))
        encoder, random_info = distill_features(untrained, source, args)
        encoders[f"untrained_{kind}"] = encoder
        info["untrained_feature_distillation"] = random_info
        payload["teachers"].append(info)
    save(args, payload)

    methods = [("baseline", None, False), ("random_prior", None, True)]
    for name, encoder in encoders.items():
        methods.extend([(f"{name}_init", encoder, False), (f"{name}_prior", encoder, True)])
    for task_id, task in enumerate(targets):
        for size in args.train_sizes:
            data = tuple(tensor[:size] for tensor in task["train"])
            for method, encoder, use_prior in methods:
                student = make_model("mlp", args, args.seed + 200 + task_id)
                if encoder is not None:
                    student.encoder.load_state_dict(encoder)
                # Every method starts with the same fresh target head.
                penalty = parameter_prior(student) if use_prior else None
                history = fit(
                    student, data, task["test"], args, args.epochs,
                    penalty=penalty, weight=args.prior_weight,
                )
                record(payload, args, method, task_id, size, history)
    print(f"Metrics saved to {args.output}")


if __name__ == "__main__":
    main()
