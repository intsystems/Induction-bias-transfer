"""Plot the method comparison JSON without running training."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = [
    ("test_accuracy", "Original test accuracy", "Accuracy (%)", 100),
    ("shifted_test_accuracy", "Label-preserving shift accuracy", "Accuracy (%)", 100),
    ("test_bce", "Original test BCE", "Loss", 1),
    ("invariance_mae", "Shift gap (probability MAE)", "Error", 1),
]


def draw(rows, x_key, methods, output, title):
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"], row[x_key], row["seed"]].append(row)
    xs = sorted({row[x_key] for row in rows})
    seeds = sorted({row["seed"] for row in rows})
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    for index, method in enumerate(methods):
        color = plt.get_cmap("tab20")(index % 20)
        for axis, (metric, metric_title, ylabel, scale) in zip(axes.flat, METRICS):
            averages, deviations = [], []
            for x in xs:
                # First average tasks within a seed, then summarize the seeds.
                seed_means = [
                    mean(row[metric] * scale for row in groups[method, x, seed])
                    for seed in seeds
                ]
                averages.append(mean(seed_means))
                deviations.append(stdev(seed_means) if len(seed_means) > 1 else 0.0)
            axis.plot(
                xs, averages, label=method, color=color,
                marker="o" if x_key == "train_size" else None,
            )
            axis.fill_between(
                xs, [m - s for m, s in zip(averages, deviations)],
                [m + s for m, s in zip(averages, deviations)], color=color, alpha=0.1,
            )
            axis.set_title(metric_title)
            axis.set_xlabel("Training examples" if x_key == "train_size" else "Epoch")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    figure.suptitle(title + "\nTask means; mean ± SD across seeds")
    figure.subplots_adjust(bottom=0.18, top=0.88, hspace=0.35, wspace=0.25)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="results/method_comparison/metrics.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--methods", nargs="+", help="Optional subset of method names")
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text())
    methods = args.methods if args.methods else payload["methods"]
    if len(methods) != len(set(methods)) or not set(methods) <= set(payload["methods"]):
        parser.error(f"Choose unique methods from: {', '.join(payload['methods'])}")
    results = [row for row in payload["results"] if row["method"] in methods]
    config = payload["config"]
    expected = {
        (method, seed, task, size)
        for method in methods for seed in config["seeds"]
        for task in range(config["target_tasks"]) for size in config["train_sizes"]
    }
    actual = [(row["method"], row["seed"], row["task"], row["train_size"]) for row in results]
    if set(actual) != expected or len(actual) != len(expected):
        parser.error("The JSON must contain all seeds, tasks and sizes for the selected methods")
    if any(row["history"][-1]["epoch"] != config["epochs"] for row in results):
        parser.error("Target training histories are incomplete")
    output = Path(args.output_dir) if args.output_dir else Path(args.input).parent
    output.mkdir(parents=True, exist_ok=True)
    final_rows = [
        {**row["history"][-1], "method": row["method"], "seed": row["seed"], "train_size": row["train_size"]}
        for row in results
    ]
    draw(final_rows, "train_size", methods, output / "sample_efficiency.png", "Method comparison")
    for size in sorted(config["train_sizes"]):
        rows = [
            {**epoch, "method": row["method"], "seed": row["seed"]}
            for row in results if row["train_size"] == size for epoch in row["history"]
        ]
        draw(rows, "epoch", methods, output / f"curves_n{size}.png", f"Training examples: {size}")
    print(f"Plots saved to {output}")


if __name__ == "__main__":
    main()
