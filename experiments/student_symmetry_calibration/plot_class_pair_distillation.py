"""Plot target accuracy and the effect of metric orientation by schedule."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results" / "class_pair_distillation"
METHODS = {"uniform": "Uniform throughout", "early_natural": "Natural metric · early",
           "early_rotated": "Rotated metric · early", "late_natural": "Natural metric · late",
           "late_rotated": "Rotated metric · late"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=RESULTS / "metrics.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "curves.png")
    args = parser.parse_args()
    data = json.loads(args.metrics.read_text())
    shots = data["config"]["shots"]
    if data["config"].get("reflection_readouts", False):
        plot_readouts(data["runs"], shots, args.output)
        return
    values = {metric: {key: 100 * np.array([
        [run["shots"][str(count)][key][metric] for count in shots] for run in data["runs"]
    ]) for key in METHODS} for metric in ("mirror_accuracy", "clean_accuracy")}
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for key, label in METHODS.items():
        axes[0].plot(shots, values["mirror_accuracy"][key].mean(axis=0), marker="o", label=label)
    for phase in ("early", "late"):
        for metric, kind, view in (("mirror_accuracy", "-", "Reflected"), ("clean_accuracy", "--", "Original")):
            effects = values[metric][f"{phase}_natural"] - values[metric][f"{phase}_rotated"]
            axes[1].plot(shots, effects.mean(axis=0), linestyle=kind, marker="o",
                         label=f"{view} · {phase}")
    axes[1].axhline(0, color=".5", linewidth=1)
    axes[0].set_ylabel("Reflected balanced accuracy (%)")
    axes[1].set_ylabel("Natural minus rotated accuracy (pp)")
    for axis in axes:
        axis.set_xlabel("Labels per class")
        axis.set_xticks(shots)
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170)


def plot_readouts(runs, shots, output):
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, layout="constrained")
    for axis, (suffix, label) in zip(axes, (("", "All coordinates"), ("_even", "Even coordinates"),
                                           ("_pooled", "Mean of both views")), strict=True):
        for key, method in METHODS.items():
            values = [[100 * run["shots"][str(count)][key + suffix]["matched_view_accuracy"]
                       for count in shots] for run in runs]
            axis.plot(shots, np.mean(values, axis=0), marker="o", label=method)
        axis.set_title(label)
        axis.set_xlabel("Labels per class")
        axis.set_xticks(shots)
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Original and reflection correct (%)")
    axes[-1].legend(fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)


if __name__ == "__main__":
    main()
