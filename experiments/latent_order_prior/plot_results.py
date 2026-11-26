"""Plot how marginal-likelihood adaptation behaves under partial symmetry."""

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
    breaking = data["config"]["breaking"]
    corruption = data["config"]["corruption"]
    fig, axes = plt.subplots(2, len(breaking), figsize=(4.7 * len(breaking), 7),
                             squeeze=False, layout="constrained")
    for col, q in enumerate(breaking):
        top, bottom = axes[:, col]
        for ax, rate in ((top, corruption[0]), (bottom, corruption[-1])):
            runs = [run["corruption"][str(rate)]["breaking"][str(q)] for run in data["runs"]]
            for method in runs[0][str(shots[0])]:
                values = np.array([[run[str(n)][method]["shifted_mse"] for n in shots] for run in runs])
                mean, sd = values.mean(axis=0), values.std(axis=0)
                line, = ax.plot(shots, mean, marker="o", label=method.replace("_", " "))
                ax.fill_between(shots, mean - sd, mean + sd, color=line.get_color(), alpha=0.10)
            ax.set_title(f"Target breaking: {q:.0%} · wrong pairs: {rate:.0%}")
            ax.set_xticks(shots)
            ax.grid(alpha=0.20)
            if col == 0:
                ax.set_ylabel("Test MSE at the held-out composition")
        bottom.set_xlabel("Labeled target examples")
    axes[0, -1].legend(fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)


if __name__ == "__main__":
    main()
