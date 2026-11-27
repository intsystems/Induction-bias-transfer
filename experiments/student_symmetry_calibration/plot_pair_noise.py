"""Plot target performance and source covariance error under pair corruption."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.metrics.read_text())
    fractions = data["config"]["fractions"]
    shots = str(20 if 20 in data["config"]["shots"] else data["config"]["shots"][0])
    arms = ("naive", "confidence", "trusted_only", "oracle", "clean_full", "confidence_rotated")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for arm in arms:
        rows = [[run["measurements"][str(f)] for f in fractions] for run in data["runs"]]
        accuracy = np.array([[100 * row["scores"][shots][arm]["matched_view_accuracy"]
                              for row in run] for run in rows])
        line, = axes[0].plot(fractions, accuracy.mean(axis=0), marker="o", label=arm)
        axes[0].fill_between(fractions, accuracy.mean(axis=0) - accuracy.std(axis=0),
                             accuracy.mean(axis=0) + accuracy.std(axis=0), alpha=.12,
                             color=line.get_color())
        if arm != "confidence_rotated":
            error = np.array([[row["relative_orbit_error"][arm] for row in run] for run in rows])
            axes[1].plot(fractions, error.mean(axis=0), marker="o", color=line.get_color())
    axes[0].set_ylabel(f"Both views correct (%) · {shots} labels/class")
    axes[1].set_ylabel("Relative source covariance error")
    for ax in axes:
        ax.set_xlabel("Fraction of mismatched pairs")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)


if __name__ == "__main__":
    main()
