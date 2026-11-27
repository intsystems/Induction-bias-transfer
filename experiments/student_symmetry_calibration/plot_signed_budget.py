"""Plot performance and inferred label actions for signed orbit priors."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results/signed_budget"
METHODS = {"teacher_parity": "Teacher parity", "proxy_centered": "Marginal proxy",
           "paired_marginal": "Paired · marginal evidence", "paired_centered": "Paired · centered evidence",
           "paired_fixed": "Paired · fixed strength", "paired_known_sign": "Paired · known action"}
CROSSED = {"teacher_parity": "Teacher head and action", "paired_centered": "Paired head and action",
           "paired_teacher_sign": "Paired head · teacher action", "teacher_paired_sign": "Teacher head · paired action",
           "proxy_teacher_sign": "Marginal proxy · teacher action", "teacher_known_sign": "Teacher head · known action",
           "paired_known_sign": "Paired head · known action"}
BLOCKS = {"teacher_parity": "Teacher parity", "block_fixed_teacher_sign": "Within-block orbit",
          "block_adaptive_teacher_sign": "Adaptive within-block orbit", "teacher_known_sign": "Parity · known action",
          "block_fixed_known_sign": "Within-block orbit · known action",
          "block_rotated_known_sign": "Within-block rotation · known action"}
RELATIVE = {"teacher_known_sign": "Teacher parity", "block_fixed_known_sign": "Absolute orbit error",
            "relative_metric_known_sign": "Compatible metric only", "relative_fixed_known_sign": "Relative orbit error",
            "relative_rotated_known_sign": "Relative spectrum · rotated",
            "relative_adaptive_known_sign": "Relative error · adaptive"}
HYBRID = {"teacher_parity": "Teacher parity", "relative_adaptive_teacher_sign": "Relative error · both actions",
          "hybrid_actions": "Action-specific hybrid", "hybrid_metric": "Hybrid · metric only",
          "hybrid_rotated": "Hybrid · rotated spectrum", "hybrid_reverse": "Reverse hybrid",
          "hybrid_known_sign": "Hybrid · known action"}
CALIBRATION = {"teacher_parity": "Teacher parity", "hybrid_actions": "Uncalibrated hybrid",
               "hybrid_source_scale": "Source RMS matching", "hybrid_cross_fit": "Cross-fitted RMS matching",
               "teacher_scale_only": "Teacher parity with hybrid scale"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=RESULTS / "metrics.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "curves.png")
    args = parser.parse_args()
    data = json.loads(args.metrics.read_text())
    shots = data["config"]["shots"]
    crossed = data["config"].get("crossed", False)
    blocked = data["config"].get("parity_blocks", False)
    relative = data["config"].get("relative_blocks", False)
    hybrid = data["config"].get("hybrid_actions", False)
    calibrated = data["config"].get("calibrate_hybrid", False)
    methods = (CALIBRATION if calibrated else HYBRID if hybrid else RELATIVE if relative
               else BLOCKS if blocked else CROSSED if crossed else METHODS)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for ax, family in zip(axes, ("invariant", "antisymmetric")):
        title = "Reflection preserves label" if family == "invariant" else "Reflection swaps labels"
        ax.set_title(title)
        for key, label in methods.items():
            names = ([f"{key}_{i}" for i in range(data["block_orientation_controls"])]
                     if key == "hybrid_rotated" or key.endswith("_rotated_known_sign") else [key])
            values = [100 * np.mean([ep["metrics"][model]["reflected"]
                                     for r in data["runs"] for task in r["tasks"][family].values()
                                     for ep in task[str(n)] for model in names]) for n in shots]
            ax.plot(shots, values, marker="o", label=label)
        ax.set_ylabel("Reflected balanced accuracy (%)")
    if hybrid:
        selected = tuple(CALIBRATION) if calibrated else (
            "teacher_parity", "hybrid_actions", "relative_adaptive_teacher_sign", "hybrid_rotated")
        for key in selected:
            names = [f"{key}_{i}" for i in range(data["block_orientation_controls"])] if key == "hybrid_rotated" else [key]
            values = [100 * np.mean([ep["metrics"][model]["both"]
                                     for r in data["runs"] for tasks in r["tasks"].values()
                                     for task in tasks.values() for ep in task[str(n)] for model in names]) for n in shots]
            axes[2].plot(shots, values, marker="o", label=methods[key])
        axes[2].set_xlabel("Labels per class")
        axes[2].set_ylabel("Both views correct (%)")
    elif relative:
        for i, label in enumerate(("Even", "Odd")):
            records = [r["achieved_budgets"]["relative_blocks"][i] for r in data["runs"]]
            signal = np.array([r["native_signal"] for r in records])
            error = np.array([r["native_orbit_error"] for r in records])
            values = np.divide(error, signal, out=np.full_like(error, np.nan), where=signal > 0)
            axes[2].plot(data["block_budgets"], np.nanmean(values, axis=0), marker="o", label=label)
        axes[2].set_xlabel("Requested residual variance budget")
        axes[2].set_ylabel("Prior-weighted residual / compatible variance")
    else:
        for key in ("teacher_parity", "proxy_centered", "paired_marginal", "paired_centered"):
            values = []
            for n in shots:
                groups = [[ep["diagnostics"][key]["same_probability"]
                           for r in data["runs"] for task in r["tasks"][family].values() for ep in task[str(n)]]
                          for family in ("invariant", "antisymmetric")]
                values.append((np.mean(groups[0]) + 1 - np.mean(groups[1])) / 2)
            axes[2].plot(shots, values, marker="o", label=METHODS[key])
        axes[2].set_title("Label action from clean support")
        axes[2].set_ylabel("Mean probability assigned to true action")
        axes[2].set_xlabel("Labels per class")
    for ax in axes[:2]:
        ax.set_xlabel("Labels per class")
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    axes[2].legend(fontsize=7)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)


if __name__ == "__main__":
    main()
