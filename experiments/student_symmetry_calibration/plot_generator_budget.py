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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=RESULTS / "metrics.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "curves.png")
    args = parser.parse_args()
    data = json.loads(args.metrics.read_text())
    shots = data["config"]["shots"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, (metric, title) in zip(axes, (("rotation", "Clean and quarter-turn"),
                                         ("composition", "Clean and composed transform"),
                                         ("all_views", "All eight views")), strict=True):
        for key, label in METHODS.items():
            names = [f"{key}_{i}" for i in range(4)] if key == "orbit_rotated" else [key]
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
