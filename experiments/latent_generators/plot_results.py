"""Plot the robustness and clean accuracy of generated augmentations."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=HERE / "results/metrics.json")
    parser.add_argument("--output", type=Path, default=HERE / "results/curves.png")
    args = parser.parse_args()
    data = json.loads(args.metrics.read_text())
    budgets = data["config"]["shots"]
    fig, axes = plt.subplots(3, len(budgets), figsize=(5.8 * len(budgets), 8.2),
                             sharex=True, sharey="row", squeeze=False, layout="constrained")
    metrics = (("shifted_accuracy", "Shifted accuracy (%)"),
               ("affine_accuracy", "Rotated/sheared accuracy (%)"),
               ("clean_accuracy", "Clean accuracy (%)"))
    for column, budget in enumerate(budgets):
        evaluations = [run["shots"][str(budget)] for run in data["runs"]]
        axes[0, column].set_title(f"{budget} labels per class")
        for row, (metric, label) in enumerate(metrics):
            ax = axes[row, column]
            for method in evaluations[0]:
                curves = [arm[method]["curve"] for arm in evaluations]
                epochs = [point["epoch"] for point in curves[0]]
                values = 100 * np.array([[point[metric] for point in curve] for curve in curves])
                mean, sd = values.mean(axis=0), values.std(axis=0)
                line, = ax.plot(epochs, mean, label=method.replace("_", " "))
                ax.fill_between(epochs, mean - sd, mean + sd, color=line.get_color(), alpha=0.10)
            ax.grid(alpha=0.2)
            if column == 0:
                ax.set_ylabel(label)
            if row == len(metrics) - 1:
                ax.set_xlabel("Target training epoch")
    axes[-1, -1].legend(fontsize=8, loc="lower right")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)


if __name__ == "__main__":
    main()
