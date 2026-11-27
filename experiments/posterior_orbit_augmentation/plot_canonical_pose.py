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
POLICIES = {"raw": "Uncorrected", "pose_point": "Full correction", "source_fixed": "Source fixed",
            "pose_selector": "Pose selector", "teacher_selector": "Teacher selector", "oracle": "Known inverse"}
POSTERIOR_POLICIES = {"raw": "Uncorrected", "pose_point": "Full correction", "point_margin": "Point margin",
                      "conditional_consensus": "Conditional samples", "global_consensus": "Global-noise samples",
                      "oracle": "Known inverse"}
SYMMETRIC_METHODS = {"raw": "Uncorrected", "pose_point": "Single prediction", "reflected_point": "Reflected prediction",
                     "symmetric_point": "Reflection-symmetric mean", "independent_point": "Independent-model mean",
                     "symmetric_posterior": "Reflection-symmetric posterior", "oracle": "Known inverse"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=RESULTS / "metrics.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "curves.png")
    args = parser.parse_args()
    results = json.loads(args.metrics.read_text())
    runs = results["runs"]
    levels = results["config"]["strengths"]
    shots = str(4 if 4 in results["config"]["shots"] else results["config"]["shots"][0])
    selective = results["config"].get("selective", False)
    posterior = results["config"].get("posterior_selection", False)
    symmetric = results["config"].get("symmetrize_pose", False)
    methods = METHODS
    if symmetric:
        methods = SYMMETRIC_METHODS
    elif posterior:
        methods = POSTERIOR_POLICIES
    elif selective:
        methods = POLICIES
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, family in zip(axes, ("invariant", "antisymmetric")):
        for key, label in methods.items():
            accuracy = np.array([[100 * np.mean([
                task[shots]["strengths"][str(level)][key]["reflected"]
                for task in r["tasks"][family].values()]) for level in levels] for r in runs])
            ax.plot(levels, accuracy.mean(axis=0), marker="o", label=label)
        ax.set_title("Reflection preserves the label" if family == "invariant" else "Reflection changes the label")
        ax.set_ylabel(f"Reflected balanced accuracy (%), {shots} labels/class")
    axes[0].legend(fontsize=8)
    if symmetric:
        for key in ("pose_point", "reflected_point", "symmetric_point", "symmetric_precision", "independent_point"):
            mse = [[r["symmetric_pose_mse"][str(level)][key][0] for level in levels] for r in runs]
            axes[2].plot(levels, np.sqrt(np.mean(mse, axis=0)) * 32, marker="o",
                         label=methods.get(key, "Precision-weighted symmetry"))
        axes[2].set_title("Inverse rotation")
        axes[2].set_ylabel("Angle RMSE (degrees)")
    elif posterior:
        for key in ("point_margin", "conditional_consensus", "global_consensus"):
            steps = [[np.mean([task[shots]["strengths"][str(level)][key]["mean_step"]
                               for task in r["tasks"]["antisymmetric"].values()])
                      for level in levels] for r in runs]
            axes[2].plot(levels, np.mean(steps, axis=0), marker="o", label=methods[key])
        axes[2].set_title("Selected correction when reflection changes labels")
        axes[2].set_ylabel("Mean correction fraction")
        axes[2].set_ylim(0, 1)
    elif selective:
        for key in ("pose_selector", "teacher_selector"):
            steps = [[r["policy_statistics"][str(level)][key]["mean_step"] for level in levels] for r in runs]
            axes[2].plot(levels, np.mean(steps, axis=0), marker="o", label=POLICIES[key])
        axes[2].set_title("Selected correction")
        axes[2].set_ylabel("Mean correction fraction")
        axes[2].set_ylim(0, 1)
    else:
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
