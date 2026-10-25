"""Train an MNIST MLP with an image shift consistency penalty."""

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST


def load_data(root, train, size, generator):
    dataset = MNIST(root, train=train, download=True)
    indices = torch.randperm(len(dataset), generator=generator)[:size]
    x = dataset.data[indices].unsqueeze(1).float() / 255.0
    return x, dataset.targets[indices]


def shift_images(x, max_shift, generator):
    # Shift each image independently; exposed pixels stay black.
    offsets = torch.randint(
        -max_shift, max_shift + 1, (len(x), 2), generator=generator,
    )
    padded = F.pad(x, (max_shift,) * 4)
    height, width = x.shape[-2:]
    images = []
    for image, (dy, dx) in zip(padded, offsets.tolist()):
        top, left = max_shift - dy, max_shift - dx
        images.append(image[:, top:top + height, left:left + width])
    return torch.stack(images)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = shifted_correct = 0
    ce_sum = gap_sum = 0.0
    for x, shifted_x, y in loader:
        x, shifted_x, y = x.to(device), shifted_x.to(device), y.to(device)
        logits = model(x)
        shifted_logits = model(shifted_x)
        correct += (logits.argmax(dim=1) == y).sum().item()
        shifted_correct += (shifted_logits.argmax(dim=1) == y).sum().item()
        ce_sum += F.cross_entropy(logits, y, reduction="sum").item()
        gap_sum += (logits - shifted_logits).abs().mean(dim=1).sum().item()
    size = len(loader.dataset)
    return {
        "test_ce": ce_sum / size,
        "test_accuracy": correct / size,
        "shifted_test_accuracy": shifted_correct / size,
        "invariance_mae": gap_sum / size,
    }


def train(args, weight, train_data, test_loader):
    # Reset initialization and batch order for each penalty weight.
    torch.manual_seed(args.seed)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, args.hidden),
        nn.ReLU(),
        nn.Linear(args.hidden, args.hidden),
        nn.ReLU(),
        nn.Linear(args.hidden, 10),
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(
        TensorDataset(*train_data), batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    augmentation_rng = torch.Generator().manual_seed(args.seed + 2)
    history = [{"epoch": 0, **evaluate(model, test_loader, args.device)}]

    for epoch in range(1, args.epochs + 1):
        model.train()
        ce_sum = consistency_sum = 0.0
        for x, y in loader:
            shifted_x = shift_images(x, args.max_shift, augmentation_rng)
            x, shifted_x, y = x.to(args.device), shifted_x.to(args.device), y.to(args.device)
            logits = model(x)
            supervised_loss = F.cross_entropy(logits, y)
            # Both branches receive gradients; f(x) denotes logits.
            consistency_loss = (logits - model(shifted_x)).abs().mean()
            loss = supervised_loss + weight * consistency_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ce_sum += supervised_loss.item() * len(x)
            consistency_sum += consistency_loss.item() * len(x)

        metrics = evaluate(model, test_loader, args.device)
        history.append({
            "epoch": epoch,
            "train_ce": ce_sum / len(loader.dataset),
            "train_consistency_mae": consistency_sum / len(loader.dataset),
            **metrics,
        })
        print(f"weight={weight:g} epoch={epoch}/{args.epochs} "
              f"accuracy={metrics['test_accuracy']:.4f} "
              f"shifted_accuracy={metrics['shifted_test_accuracy']:.4f} "
              f"invariance_mae={metrics['invariance_mae']:.4f}")

    return {"consistency_weight": weight, "history": history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-shift", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.1, 0.5, 1.0, 3.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="results/image_shift/metrics.json")
    args = parser.parse_args()
    for name in ("train_size", "test_size", "hidden", "epochs", "batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.train_size > 60000 or args.test_size > 10000:
        parser.error("MNIST has 60000 training and 10000 test images")
    if not 1 <= args.max_shift < 28:
        parser.error("--max-shift must be between 1 and 27 pixels")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    if any(not math.isfinite(weight) or weight < 0 for weight in args.weights):
        parser.error("--weights must be finite and nonnegative")

    data_rng = torch.Generator().manual_seed(args.seed)
    train_data = load_data(args.data_dir, True, args.train_size, data_rng)
    test_x, test_y = load_data(args.data_dir, False, args.test_size, data_rng)
    # Fix test shifts across epochs and penalty weights.
    shifted_test_x = shift_images(test_x, args.max_shift, data_rng)
    test_loader = DataLoader(
        TensorDataset(test_x, shifted_test_x, test_y), batch_size=args.batch_size,
        generator=torch.Generator().manual_seed(args.seed + 3),
    )
    payload = {
        "experiment": "mnist_shift_consistency",
        "consistency_output": "logits",
        "config": vars(args),
        "results": [],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for weight in args.weights:
        payload["results"].append(train(args, weight, train_data, test_loader))
        output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"Metrics saved to {output}")


if __name__ == "__main__":
    main()
