"""Shared plotting code for the two pattern transfer experiments."""

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = [
    ("test_accuracy", "Original test accuracy", 100),
    ("shifted_test_accuracy", "Label-preserving shift accuracy", 100),
    ("invariance_mae", "Shift gap (probability MAE)", 1),
]


def draw(rows, x_key, output, title):
    figure, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    methods = list(dict.fromkeys(row["method"] for row in rows))
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        xs = sorted({row[x_key] for row in selected})
        for axis, (metric, metric_title, scale) in zip(axes, METRICS):
            values = [[row[metric] * scale for row in selected if row[x_key] == x] for x in xs]
            averages = [mean(group) for group in values]
            deviations = [pstdev(group) for group in values]
            line, = axis.plot(xs, averages, label=method, marker="o" if x_key == "train_size" else None)
            # Shading is the spread across target tasks, not a confidence interval.
            axis.fill_between(
                xs, [m - s for m, s in zip(averages, deviations)],
                [m + s for m, s in zip(averages, deviations)],
                color=line.get_color(), alpha=0.12,
            )
            axis.set_title(metric_title)
            axis.set_xlabel("Training examples" if x_key == "train_size" else "Epoch")
            axis.set_ylabel("Accuracy (%)" if scale == 100 else "Error")
            axis.grid(alpha=0.3)
    axes[-1].legend(fontsize=7, loc="best")
    figure.suptitle(title)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main(default_input):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default=default_input, help="Experiment metrics JSON")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--methods", nargs="+", help="Optional subset of method names")
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text())
    results = payload["results"]
    if args.methods:
        available = {row["method"] for row in results}
        if not set(args.methods) <= available:
            parser.error(f"Available methods: {', '.join(sorted(available))}")
        results = [row for row in results if row["method"] in args.methods]
    if not results:
        parser.error("No completed target runs in the JSON")
    output = Path(args.output_dir) if args.output_dir else Path(args.input).parent
    output.mkdir(parents=True, exist_ok=True)

    final_rows = [
        {**row["history"][-1], "method": row["method"], "train_size": row["train_size"]}
        for row in results
    ]
    draw(final_rows, "train_size", output / "sample_efficiency.png", payload["experiment"])
    for size in sorted({row["train_size"] for row in results}):
        history_rows = [
            {**epoch, "method": row["method"]}
            for row in results if row["train_size"] == size for epoch in row["history"]
        ]
        draw(history_rows, "epoch", output / f"curves_n{size}.png", f"Training examples: {size}")
    print(f"Plots saved to {output}")
