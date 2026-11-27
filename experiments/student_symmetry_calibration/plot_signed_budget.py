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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=RESULTS / "metrics.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "curves.png")
    args = parser.parse_args()
    data = json.loads(args.metrics.read_text())
    shots = data["config"]["shots"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for ax, family in zip(axes, ("invariant", "antisymmetric")):
        title = "Reflection preserves label" if family == "invariant" else "Reflection swaps labels"
        ax.set_title(title)
        for key, label in METHODS.items():
            values = [100 * np.mean([ep["metrics"][key]["reflected"]
                                     for r in data["runs"] for task in r["tasks"][family].values()
                                     for ep in task[str(n)]]) for n in shots]
            ax.plot(shots, values, marker="o", label=label)
        ax.set_ylabel("Reflected balanced accuracy (%)")
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
    for ax in axes:
        ax.set_xlabel("Labels per class")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    axes[2].legend(fontsize=7)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)


if __name__ == "__main__":
    main()
