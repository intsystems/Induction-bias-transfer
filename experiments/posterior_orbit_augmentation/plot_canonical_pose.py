"""Plot classification and pose recovery for transferred canonicalizers."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results/canonical_pose"
METHODS = {"raw": "Uncorrected", "oracle": "Known inverse", "pose_posterior": "Pose-only posterior",
           "hidden_point": "Teacher loss · point", "hidden_posterior": "Teacher loss · posterior",
           "hidden_shuffled": "Shuffled posterior"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=RESULTS / "metrics.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "curves.png")
    args = parser.parse_args()
    results = json.loads(args.metrics.read_text())
    runs = results["runs"]
    levels = results["config"]["strengths"]
    shots = str(4 if 4 in results["config"]["shots"] else results["config"]["shots"][0])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, family in zip(axes, ("invariant", "antisymmetric")):
        for key, label in METHODS.items():
            accuracy = np.array([[100 * np.mean([
                task[shots]["strengths"][str(level)][key]["reflected"]
                for task in r["tasks"][family].values()]) for level in levels] for r in runs])
            ax.plot(levels, accuracy.mean(axis=0), marker="o", label=label)
        ax.set_title("Reflection preserves the label" if family == "invariant" else "Reflection changes the label")
        ax.set_ylabel(f"Reflected balanced accuracy (%), {shots} labels/class")
    axes[0].legend(fontsize=8)
    for key, label in (("pose", "Pose-only training"), ("hidden", "Teacher loss")):
        mse = np.array([[r["pose_calibration"][str(level)][key]["mse"] for level in levels] for r in runs])
        axes[2].plot(levels, np.sqrt(mse.mean(axis=0)[:, 0]) * 32, marker="o", label=label)
    axes[2].set_title("Inverse rotation")
    axes[2].set_ylabel("Angle RMSE (degrees)")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("Pose distortion scale")
        ax.grid(alpha=.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)


if __name__ == "__main__":
    main()
