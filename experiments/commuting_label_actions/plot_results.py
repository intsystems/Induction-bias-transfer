"""Plot recovery of the held-out composition across support budgets."""

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
    shots, rates = data["config"]["shots"], data["config"]["corruption"]
    fig, axes = plt.subplots(2, len(rates), figsize=(5 * len(rates), 6.7), squeeze=False,
                             sharex=True, sharey="row", layout="constrained")
    for col, rate in enumerate(rates):
        runs = [r["corruption"][str(rate)]["shots"] for r in data["runs"]]
        colors = {method: f"C{i}" for i, method in enumerate(runs[0][str(shots[0])])}
        axes[0, col].set_title(f"Wrong source pairs: {rate:.0%}")
        for row, (metric, label) in enumerate([
                ("shifted_accuracy", "Composed-view accuracy (%)"),
                ("composition_accuracy", "Recovered class action (%)")]):
            ax = axes[row, col]
            for mode, values in runs[0][str(shots[0])].items():
                if metric not in values:
                    continue
                scores = 100 * np.array([[run[str(n)][mode][metric] for n in shots] for run in runs])
                center, spread = scores.mean(axis=0), scores.std(axis=0)
                line, = ax.plot(shots, center, marker="o", color=colors[mode], label=mode)
                ax.fill_between(shots, center - spread, center + spread, color=line.get_color(), alpha=.12)
            ax.grid(alpha=.2)
            ax.set_xticks(shots)
            if col == 0:
                ax.set_ylabel(label)
            if row == 1:
                ax.set_xlabel("Labeled examples per class")
    axes[0, -1].legend(fontsize=8, loc="lower right")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)


if __name__ == "__main__":
    main()
