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
APPROXIMATE_METHODS = {"inference_only": "Project only for prediction", "learned_training": "Round all angles",
                       "midpoint_training": "Round angles halfway", "unrounded_training": "Keep fitted angles"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=RESULTS / "metrics.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "curves.png")
    args = parser.parse_args()
    data = json.loads(args.metrics.read_text())
    shots = data["config"]["shots"]
    if data["config"].get("view_budget", False):
        plot_view_budget(data["runs"], shots, args.output)
        return
    methods = APPROXIMATE_METHODS if data["config"].get("approximate_action", False) else METHODS
    performance = {method: 100 * np.array([
        [run["shots"][str(count)][method]["clean_accuracy"] for count in shots] for run in data["runs"]
    ]) for method in methods}
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for method, label in methods.items():
        axes[0].plot(shots, performance[method].mean(axis=0), marker="o", label=label)
    for method in (key for key in methods if key != "inference_only"):
        effect = performance[method] - performance["inference_only"]
        axes[1].plot(shots, effect.mean(axis=0), marker="o", label=methods[method])
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


def plot_view_budget(runs, shots, output):
    names = ("one", "adjacent", "opposite", "four")
    labels = ("One", "Two adjacent", "Two opposite", "Four")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for axis, (metric, title) in zip(axes, (("mean_accuracy", "Mean over input rotations"),
                                           ("all_rotations_accuracy", "All input rotations correct")), strict=True):
        for model, label in (("inference_only", "Projection only at prediction"),
                             ("learned_training", "Projection also in training")):
            accuracy = [[run["shots"][str(min(shots))][model][view][metric] for view in names] for run in runs]
            axis.plot(labels, 100 * np.mean(accuracy, axis=0), marker="o", label=label)
        axis.set_title(title)
        axis.set_xlabel(f"Views averaged at prediction, {min(shots)} labels/class")
        axis.set_ylabel("Balanced accuracy (%)")
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)


if __name__ == "__main__":
    main()
