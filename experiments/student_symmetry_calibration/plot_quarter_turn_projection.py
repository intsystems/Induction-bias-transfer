"""Plot held-out CIFAR-10 accuracy after quarter-turn projection."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results" / "quarter_turn_projection"
METHODS = {"inference_only": "Project only for prediction", "learned_training": "Project in training",
           "rotated_training": "Rotated action in training"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=RESULTS / "metrics.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "curves.png")
    args = parser.parse_args()
    data = json.loads(args.metrics.read_text())
    shots = data["config"]["shots"]
    performance = {method: 100 * np.array([
        [run["shots"][str(count)][method]["clean_accuracy"] for count in shots] for run in data["runs"]
    ]) for method in METHODS}
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for method, label in METHODS.items():
        axes[0].plot(shots, performance[method].mean(axis=0), marker="o", label=label)
    for method in ("learned_training", "rotated_training"):
        effect = performance[method] - performance["inference_only"]
        axes[1].plot(shots, effect.mean(axis=0), marker="o", label=METHODS[method])
    axes[1].axhline(0, color=".5", linewidth=1)
    axes[0].set_ylabel("Rotation-invariant balanced accuracy (%)")
    axes[1].set_ylabel("Change from prediction-only projection (pp)")
    for axis in axes:
        axis.set_xlabel("Labels per class")
        axis.set_xticks(shots)
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170)


if __name__ == "__main__":
    main()
