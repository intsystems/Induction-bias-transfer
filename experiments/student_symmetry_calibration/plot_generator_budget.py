"""Plot generator and composition accuracy with joint variance constraints."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results/generator_budget"
METHODS = {"isotropic": "Isotropic", "canonical_average": "Canonical edges · average",
           "canonical_balanced": "Canonical edges · balanced", "orbit_average": "Orbit edges · average",
           "orbit_balanced": "Orbit edges · balanced", "orbit_rotated": "Rotated spectrum"}
CONTROL_METHODS = {"orbit_rotation_only": "Quarter-turn only", "orbit_reflection_only": "Reflection only",
                   "orbit_composition_only": "Composition only", "orbit_average": "Shared budget",
                   "orbit_balanced": "Separate budgets", "orbit_rotated": "Rotated spectrum"}
OPERATOR_METHODS = {"split_balanced": "Separate budgets · split pairs",
                    "orbit_rotation_only": "Quarter-turn · all pairs",
                    "composition_plug_in": "Composed operators",
                    "composition_additive": "Additive residuals",
                    "composition_transported": "Transported residuals",
                    "orbit_composition_only": "Direct composition pairs",
                    "transported_rotated": "Rotated transported spectrum"}
PATH_METHODS = {"composition_plug_in": "Quarter-turn, then reflection",
                "composition_reverse_path": "Reflection, then inverse quarter-turn",
                "composition_mean_prediction": "Mean of equivalent predictions",
                "composition_mean_moment": "Mean of equivalent moments",
                "composition_wrong_prediction": "Mean with non-equivalent path",
                "orbit_composition_only": "Direct composition pairs",
                "path_rotated": "Rotated mean-prediction spectrum"}


def plot_teacher_views(rows, shots):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    views = ("horizontal", "vertical")
    gains = {}
    for ax, test_axis in zip(axes[:2], views, strict=True):
        for teacher, color in zip(views, ("tab:blue", "tab:orange"), strict=True):
            selected = [r["teacher_views"][teacher][test_axis] for r in rows]
            methods = {"composition_mean_moment": ("Mean moments", "-"),
                       "split_balanced": ("Separate budgets", "--")}
            curves = {}
            for method, (label, style) in methods.items():
                values = [100 * np.mean([ep[method]["composition"] for r in selected
                                         for ep in r["shots"][str(n)]]) for n in shots]
                curves[method] = np.asarray(values)
                ax.plot(shots, values, marker="o", linestyle=style, color=color, label=f"{teacher.title()} · {label}")
            gains[teacher, test_axis] = curves["composition_mean_moment"] - curves["split_balanced"]
        ax.set_title(f"Quarter-turn then {test_axis} reflection")
        ax.set_ylabel("Correct on clean and composition (%)")
    ax = axes[2]
    ax.plot(shots, (gains["horizontal", "horizontal"] + gains["vertical", "vertical"]) / 2,
            marker="o", label="Matching teacher reflection")
    ax.plot(shots, (gains["vertical", "horizontal"] + gains["horizontal", "vertical"]) / 2,
            marker="o", label="Other teacher reflection")
    ax.set_title("Effect of averaging the two path moments")
    ax.set_ylabel("Gain over separate budgets (pp)")
    for ax in axes:
        ax.set_xlabel("Labels per class")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    axes[2].legend(fontsize=7)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=RESULTS / "metrics.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "curves.png")
    parser.add_argument("--teacher-state", choices=("trained", "initialized"), default="trained")
    args = parser.parse_args()
    data = json.loads(args.metrics.read_text())
    rows = data["runs"]
    shots = data["config"]["shots"]
    if data["config"].get("compare_teacher_views", False):
        fig = plot_teacher_views(rows, shots)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=170)
        return
    teacher_comparison = data["config"].get("compare_teacher_training", False)
    if args.teacher_state == "initialized":
        if not teacher_comparison:
            parser.error("initialized teacher results are unavailable in these metrics")
        rows = [r["initialized_teacher"] for r in rows]
    metrics = [("rotation", "Clean and quarter-turn"), ("composition", "Clean and composed transform"),
               ("all_views", "All eight views")]
    methods = METHODS
    if data["config"].get("path_composition", False):
        methods = PATH_METHODS
        metrics.insert(0, ("reflection", "Clean and reflection"))
    elif data["config"].get("operator_composition", False):
        methods = OPERATOR_METHODS
        metrics.insert(0, ("reflection", "Clean and reflection"))
    elif data["config"].get("generator_controls", False):
        methods = CONTROL_METHODS
        metrics.insert(0, ("reflection", "Clean and reflection"))
    fig, axes = plt.subplots(1, len(metrics), figsize=(17 if len(metrics) == 4 else 13, 4), layout="constrained")
    if "source_domain" in data["config"]:
        label = {"cifar": "CIFAR-10", "gtsrb": "GTSRB"}[data["config"]["source_domain"]]
        title = f"Source {label} · Target CIFAR-10"
        if teacher_comparison:
            title += f" · {args.teacher_state.capitalize()} teacher"
        fig.suptitle(title)
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        for key, label in methods.items():
            names = [f"{key}_{i}" for i in range(4)] if key.endswith("_rotated") else [key]
            values = [100 * np.mean([ep[name][metric] for r in rows for ep in r["shots"][str(n)]
                                     for name in names]) for n in shots]
            ax.plot(shots, values, marker="o", label=label)
        ax.set_title(title)
        ax.set_xlabel("Labels per class")
        ax.set_ylabel("Correct on every indicated view (%)")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)


if __name__ == "__main__":
    main()
