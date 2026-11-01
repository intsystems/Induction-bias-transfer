"""Shared data and training code for the two pattern transfer experiments."""

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


def shift_candidates(x):
    # The four one-step shifts used in assets/conv_data_v2.ipynb.
    minus = -torch.ones_like(x[:, :1])
    plus = torch.ones_like(minus)
    return torch.stack([
        torch.cat([minus, x[:, :-1]], dim=1),
        torch.cat([plus, x[:, :-1]], dim=1),
        torch.cat([x[:, 1:], minus], dim=1),
        torch.cat([x[:, 1:], plus], dim=1),
    ], dim=1)


def pattern_labels(x, pattern):
    p = x.new_tensor([2 * int(bit) - 1 for bit in pattern])
    return (x.unfold(-1, len(pattern), 1) == p).all(dim=-1).any(dim=-1)


def generate(pattern_len, seq_len, task_num, size, seed):
    """Port the notebook sampler to CPU without notebook dependencies."""
    rng = random.Random(seed)
    tasks = []
    patterns = set()
    for _ in range(task_num):
        # Ensure that source and target tasks have different patterns.
        pattern = "".join(rng.choices("01", k=pattern_len))
        while pattern in patterns:
            pattern = "".join(rng.choices("01", k=pattern_len))
        patterns.add(pattern)
        strings = ["".join(rng.choices("01", k=seq_len)) for _ in range(2 * size)]
        labels = [int(pattern in text) for text in strings]
        for start in (0, size):
            # As in the notebook, raise the positive count to at least half.
            while sum(labels[start:start + size]) < size // 2:
                index = rng.choice([
                    i for i in range(start, start + size) if labels[i] == 0
                ])
                offset = rng.randrange(seq_len - pattern_len + 1)
                text = strings[index]
                strings[index] = text[:offset] + pattern + text[offset + pattern_len:]
                labels[index] = 1

        x = torch.tensor([[2 * int(bit) - 1 for bit in text] for text in strings]).float()
        y = torch.tensor(labels).float()
        candidates = shift_candidates(x)
        valid = pattern_labels(candidates, pattern) == y[:, None].bool()
        # The paired evaluation shift is conditioned on the true pattern.
        choices = [rng.choice(row.nonzero().flatten().tolist()) for row in valid]
        shifted_x = candidates[torch.arange(len(x)), choices]
        tasks.append({
            "pattern": pattern,
            "train": (x[:size], y[:size]),
            "test": (x[size:], y[size:], shifted_x[size:]),
        })
    return tasks


class MLP(nn.Module):
    def __init__(self, seq_len, hidden):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(seq_len, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, 1)

    def features(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


class CNN(nn.Module):
    def __init__(self, pattern_len, channels):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(1, channels, pattern_len), nn.ReLU())
        self.head = nn.Linear(channels, 1)

    def features(self, x):
        return self.encoder(x[:, None]).flatten(1)

    def forward(self, x):
        # Shared local filters and max pooling encode the locality bias.
        h = self.encoder(x[:, None]).amax(dim=-1)
        return self.head(h).squeeze(-1)


def make_model(kind, args, seed):
    torch.manual_seed(seed)
    if kind == "cnn":
        model = CNN(args.pattern_len, args.channels)
    else:
        model = MLP(args.seq_len, args.channels * (args.seq_len - args.pattern_len + 1))
    return model.to(args.device)


def batches(data, args, seed):
    return DataLoader(
        TensorDataset(*data), batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )


@torch.no_grad()
def evaluate(model, data, args):
    model.eval()
    totals = dict(test_bce=0.0, test_accuracy=0.0, shifted_test_accuracy=0.0, invariance_mae=0.0)
    for x, y, shifted_x in DataLoader(TensorDataset(*data), batch_size=args.batch_size):
        x, y, shifted_x = x.to(args.device), y.to(args.device), shifted_x.to(args.device)
        logits, shifted_logits = model(x), model(shifted_x)
        totals["test_bce"] += F.binary_cross_entropy_with_logits(logits, y, reduction="sum").item()
        totals["test_accuracy"] += ((logits >= 0) == y.bool()).sum().item()
        totals["shifted_test_accuracy"] += ((shifted_logits >= 0) == y.bool()).sum().item()
        totals["invariance_mae"] += (logits.sigmoid() - shifted_logits.sigmoid()).abs().sum().item()
    return {name: value / len(data[0]) for name, value in totals.items()}


def fit(model, data, test, args, epochs, penalty=None, weight=0.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = batches(data, args, args.seed + 10)
    history = [{"epoch": 0, **evaluate(model, test, args)}]
    for epoch in range(1, epochs + 1):
        model.train()
        bce_sum = penalty_sum = 0.0
        for x, y in loader:
            x, y = x.to(args.device), y.to(args.device)
            logits = model(x)
            bce = F.binary_cross_entropy_with_logits(logits, y)
            regularizer = penalty(model, x, logits) if penalty else logits.new_zeros(())
            loss = bce + weight * regularizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bce_sum += bce.item() * len(x)
            penalty_sum += regularizer.item() * len(x)
        history.append({
            "epoch": epoch, "train_bce": bce_sum / len(data[0]),
            "train_penalty": penalty_sum / len(data[0]),
            **evaluate(model, test, args),
        })
    return history


def freeze(model):
    model.eval()
    model.requires_grad_(False)
    return model


def make_parser(description, output):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--pattern-len", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--size", type=int, default=512, help="Train and test size per task")
    parser.add_argument("--target-tasks", type=int, default=3)
    parser.add_argument("--train-sizes", nargs="+", type=int, default=[32, 128, 512])
    parser.add_argument("--teachers", nargs="+", choices=["cnn", "mlp"], default=["cnn", "mlp"])
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--teacher-epochs", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=output)
    return parser


def validate(parser, args):
    for name in ("size", "target_tasks", "channels", "batch_size", "teacher_epochs", "epochs"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 2 <= args.pattern_len < args.seq_len:
        parser.error("Require 2 <= pattern-len < seq-len")
    if args.target_tasks + 1 >= 2 ** args.pattern_len:
        parser.error("Require more possible patterns than source and target tasks")
    if (args.pattern_len + 2 ** (args.seq_len - args.pattern_len)) / (4 * args.size) <= 1:
        parser.error("Increase seq-len or reduce size to satisfy the notebook sampling condition")
    if any(size < 2 or size > args.size for size in args.train_sizes):
        parser.error("Each train size must be between 2 and --size")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")


def prepare(args, name):
    tasks = generate(args.pattern_len, args.seq_len, args.target_tasks + 1, args.size, args.seed)
    payload = {
        "experiment": name, "config": vars(args),
        "source_pattern": tasks[0]["pattern"],
        "target_patterns": [task["pattern"] for task in tasks[1:]],
        "teachers": [], "results": [],
    }
    return tasks[0], tasks[1:], payload


def train_teacher(kind, source, args):
    teacher = make_model(kind, args, args.seed + 100)
    history = fit(teacher, source["train"], source["test"], args, args.teacher_epochs)
    print(f"teacher={kind} source_accuracy={history[-1]['test_accuracy']:.4f}")
    return freeze(teacher), {"kind": kind, "history": history}


def save(args, payload):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def record(payload, args, method, task_id, size, history, **extra):
    payload["results"].append({
        "method": method, "task": task_id, "train_size": size, "history": history, **extra,
    })
    save(args, payload)
    print(f"method={method} task={task_id} size={size} "
          f"accuracy={history[-1]['test_accuracy']:.4f}")
