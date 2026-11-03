"""Train an MLP with a coordinate permutation consistency penalty."""

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_data(size, dim, generator):
    x = torch.randn(size, dim, generator=generator)
    # The target depends on the sum, so any permutation preserves it.
    y = torch.sin(x.sum(dim=1, keepdim=True) / math.sqrt(dim))
    return x, y


def shuffle_coordinates(x, generator):
    # Use an independent permutation for each sample.
    indices = torch.rand(x.shape, generator=generator).argsort(dim=1)
    return x.gather(1, indices)


@torch.no_grad()
def evaluate(model, x, shuffled_x, y):
    model.eval()
    prediction = model(x)
    shuffled_prediction = model(shuffled_x)
    return {
        "test_mse": (prediction - y).square().mean().item(),
        "shuffled_test_mse": (shuffled_prediction - y).square().mean().item(),
        "invariance_mae": (prediction - shuffled_prediction).abs().mean().item(),
    }


def train(args, weight, train_data, test_data):
    # Reset initialization and batch order for a fair comparison.
    torch.manual_seed(args.seed)
    model = nn.Sequential(
        nn.Linear(args.dim, args.hidden),
        nn.ReLU(),
        nn.Linear(args.hidden, args.hidden),
        nn.ReLU(),
        nn.Linear(args.hidden, 1),
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(
        TensorDataset(*train_data),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    augmentation_rng = torch.Generator().manual_seed(args.seed + 2)
    history = [{"epoch": 0, **evaluate(model, *test_data)}]

    for epoch in range(1, args.epochs + 1):
        model.train()
        mse_sum = 0.0
        consistency_sum = 0.0
        for x, y in loader:
            augmented_x = shuffle_coordinates(x, augmentation_rng).to(args.device)
            x, y = x.to(args.device), y.to(args.device)
            prediction = model(x)
            supervised_loss = (prediction - y).square().mean()
            # Both predictions receive gradients from the consistency loss.
            consistency_loss = (prediction - model(augmented_x)).abs().mean()
            loss = supervised_loss + weight * consistency_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mse_sum += supervised_loss.item() * len(x)
            consistency_sum += consistency_loss.item() * len(x)

        metrics = evaluate(model, *test_data)
        history.append({
            "epoch": epoch,
            "train_mse": mse_sum / len(loader.dataset),
            "train_consistency_mae": consistency_sum / len(loader.dataset),
            **metrics,
        })
        print(f"weight={weight:g} epoch={epoch}/{args.epochs} "
              f"test_mse={metrics['test_mse']:.4f} "
              f"invariance_mae={metrics['invariance_mae']:.4f}")

    return {"consistency_weight": weight, "history": history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--test-size", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.1, 0.5, 1.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="results/metrics.json")
    args = parser.parse_args()
    for name in ("train_size", "test_size", "dim", "hidden", "epochs", "batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    if any(not math.isfinite(weight) or weight < 0 for weight in args.weights):
        parser.error("--weights must be finite and nonnegative")

    data_rng = torch.Generator().manual_seed(args.seed)
    train_data = make_data(args.train_size, args.dim, data_rng)
    test_x, test_y = make_data(args.test_size, args.dim, data_rng)
    # Keep evaluation inputs fixed across all epochs and penalty weights.
    shuffled_test_x = shuffle_coordinates(test_x, data_rng)
    test_data = tuple(
        tensor.to(args.device) for tensor in (test_x, shuffled_test_x, test_y)
    )
    payload = {
        "experiment": "sum_permutation_consistency",
        "target": "sin(sum(x) / sqrt(dim))",
        "config": vars(args),
        "results": [],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for weight in args.weights:
        payload["results"].append(train(args, weight, train_data, test_data))
        output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"Metrics saved to {output}")


if __name__ == "__main__":
    main()
