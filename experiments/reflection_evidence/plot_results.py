"""Plot traffic-sign accuracy on reflected inputs and inferred parity shares."""

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
    shots = data["config"]["shots"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True, sharey="row",
                             layout="constrained")
    for col, family in enumerate(data["target_tasks"]):
        runs = [run["tasks"][family] for run in data["runs"]]
        one_task = next(iter(runs[0].values()))
        colors = {method: f"C{i}" for i, method in enumerate(one_task[str(shots[0])])}
        axes[0, col].set_title(f"{family.capitalize()} sign pairs")
        for row, (metric, label) in enumerate([("mirror_accuracy", "Balanced mirror accuracy (%)"),
                                               ("mean_odd_share", "Posterior odd share (%)")]):
            ax = axes[row, col]
            for method, measurements in one_task[str(shots[0])].items():
                if metric not in measurements:
                    continue
                values = 100 * np.array([
                    [np.mean([task[str(n)][method][metric] for task in run.values()]) for n in shots]
                    for run in runs
                ])
                mean, sd = values.mean(axis=0), values.std(axis=0)
                line, = ax.plot(shots, mean, marker="o", color=colors[method],
                                label=method.replace("_", " "))
                ax.fill_between(shots, mean - sd, mean + sd, color=line.get_color(), alpha=0.11)
            if col == 0:
                ax.set_ylabel(label)
            if row == 1:
                ax.set_xlabel("Labeled sign tracks per class")
            ax.set_xticks(shots)
            ax.grid(alpha=0.2)
    axes[0, 1].legend(fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)


if __name__ == "__main__":
    main()
