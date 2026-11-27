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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=RESULTS / "metrics.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "curves.png")
    args = parser.parse_args()
    data = json.loads(args.metrics.read_text())
    shots = data["config"]["shots"]
    metrics = [("rotation", "Clean and quarter-turn"), ("composition", "Clean and composed transform"),
               ("all_views", "All eight views")]
    methods = METHODS
    if data["config"].get("operator_composition", False):
        methods = OPERATOR_METHODS
        metrics.insert(0, ("reflection", "Clean and reflection"))
    elif data["config"].get("generator_controls", False):
        methods = CONTROL_METHODS
        metrics.insert(0, ("reflection", "Clean and reflection"))
    fig, axes = plt.subplots(1, len(metrics), figsize=(17 if len(metrics) == 4 else 13, 4), layout="constrained")
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        for key, label in methods.items():
            names = [f"{key}_{i}" for i in range(4)] if key.endswith("_rotated") else [key]
            values = [100 * np.mean([ep[name][metric] for r in data["runs"] for ep in r["shots"][str(n)]
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
