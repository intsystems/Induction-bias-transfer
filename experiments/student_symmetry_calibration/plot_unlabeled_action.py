"""Plot label-action inference and pair accuracy with unlabelled calibration."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results/unlabeled_action"
METHODS = {"clean": "Clean labels", "pair_only": "Unlabelled pairs", "combined": "Combined evidence",
           "combined_posterior": "Combined · posterior moment", "combined_shuffled": "Shuffled pairs",
           "known_sign": "Known action"}
CENTERED = {"clean": "Clean labels", "combined": "Uncentered pairs", "centered_geometry": "Geometry-centered",
            "centered_random": "Random-head centered", "centered_labels": "Label-permutation centered",
            "centered_labels_shuffled": "Label-centered · shuffled pairs", "known_sign": "Known action"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=RESULTS / "metrics.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "curves.png")
    args = parser.parse_args()
    data = json.loads(args.metrics.read_text())
    shots = data["config"]["shots"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    methods = CENTERED if data["config"].get("centered_null", False) else METHODS
    for key, label in methods.items():
        correct = []
        for ax, family in zip(axes[:2], ("invariant", "antisymmetric"), strict=True):
            rows = [[ep for r in data["runs"] for task in r["tasks"][family].values()
                     for ep in task[str(n)]] for n in shots]
            ax.plot(shots, [100 * np.mean([ep["metrics"][key]["both"] for ep in row]) for row in rows],
                    marker="o", label=label)
            sign = np.array([[ep["same_weights"][key] for ep in row] for row in rows])
            correct.append(sign.mean(axis=1) if family == "invariant" else 1 - sign.mean(axis=1))
        axes[2].plot(shots, (correct[0] + correct[1]) / 2, marker="o", label=label)
    axes[0].set_title("Reflection preserves label")
    axes[1].set_title("Reflection swaps labels")
    axes[2].set_title("Action inference")
    for ax in axes[:2]:
        ax.set_ylabel("Both views correct (%)")
    axes[2].set_ylabel("Mean weight assigned to true action")
    for ax in axes:
        ax.set_xlabel("Labels per class")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)


if __name__ == "__main__":
    main()
