"""Plot performance and selected orbit budgets across support sizes."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results/budget_evidence"
METHODS = {"fixed_1": "Isotropic", "fixed_0.25": "Fixed budget 25%",
           "evidence": "Marginal evidence", "centered": "Centered evidence",
           "loo": "Clean leave-one-out", "orientation": "Orientation contrast",
           "orientation_guarded": "Contrast with isotropic fallback", "ordinary_orbit": "Ordinary orbit"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=RESULTS / "metrics.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "curves.png")
    args = parser.parse_args()
    data = json.loads(args.metrics.read_text())
    shots = data["config"]["shots"]
    runs = [[r["shots"][str(n)] for n in shots] for r in data["runs"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for method, label in METHODS.items():
        for i, metric in enumerate(("matched_view_accuracy", "rotated_accuracy")):
            values = np.array([[np.mean([ep["scores"][method][metric] for ep in row]) * 100
                                for row in r] for r in runs])
            axes[i].plot(shots, values.mean(axis=0), marker="o", label=label)
        if method in runs[0][0][0]["chosen"]:
            budgets = np.array([[np.mean([ep["chosen"][method] for ep in row]) for row in r] for r in runs])
            axes[2].plot(shots, budgets.mean(axis=0), marker="o", label=label)
    axes[0].set_ylabel("Original and reflection correct (%)")
    axes[1].set_ylabel("Rotation accuracy (%)")
    axes[2].set_ylabel("Selected residual variance budget")
    for ax in axes:
        ax.set_xlabel("Labels per class")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    axes[2].legend(fontsize=7)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)


if __name__ == "__main__":
    main()
