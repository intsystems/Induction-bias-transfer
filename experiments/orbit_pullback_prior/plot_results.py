"""Plot mean learning curves and their variation across paired seeds."""

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
    fig, axes = plt.subplots(2, len(budgets), figsize=(6 * len(budgets), 7),
                             sharex=True, sharey="row", squeeze=False, layout="constrained")
    for column, budget in enumerate(budgets):
        arms = [run["shots"][str(budget)] for run in data["runs"]]
        axes[0, column].set_title(f"{budget} labels per class")
        for row, metric in enumerate(("shifted_accuracy", "clean_accuracy")):
            ax = axes[row, column]
            for method in arms[0]:
                epochs = [point["epoch"] for point in arms[0][method]]
                values = 100 * np.array([[point[metric] for point in arm[method]] for arm in arms])
                mean, spread = values.mean(axis=0), values.std(axis=0)
                line, = ax.plot(epochs, mean, label=method.replace("_", " "))
                ax.fill_between(epochs, mean - spread, mean + spread, color=line.get_color(), alpha=0.09)
            ax.grid(alpha=0.25)
            if column == 0:
                ax.set_ylabel(metric.replace("_", " ").capitalize() + " (%)")
            if row == 1:
                ax.set_xlabel("Target training epoch")
    axes[0, -1].legend(fontsize=8, loc="lower right")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)


if __name__ == "__main__":
    main()
