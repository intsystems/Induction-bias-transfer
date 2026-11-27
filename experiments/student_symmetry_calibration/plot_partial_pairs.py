"""Plot student accuracy, covariance error and recovery under partial overlap."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results/partial_pairs_independent"
METHODS = {"forced": "Forced assignment", "posthoc": "Post-hoc rejection",
           "nearest": "Nearest with rejection", "optional": "Joint rejection",
           "trusted_only": "Trusted anchors", "oracle": "Known pairs"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=RESULTS / "metrics.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "curves.png")
    args = parser.parse_args()
    data = json.loads(args.metrics.read_text())
    overlaps = sorted(data["config"]["overlaps"])
    shots = str(20 if 20 in data["config"]["shots"] else data["config"]["shots"][0])
    runs = [[r["measurements"][str(level)] for level in overlaps] for r in data["runs"]]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    accuracy, error, precision, recall = axes.flat

    def add(ax, get_value, label, color=None):
        values = np.array([[get_value(row) for row in run] for run in runs], dtype=float)
        counts = np.sum(np.isfinite(values), axis=0)
        means = np.divide(np.nansum(values, axis=0), counts,
                          out=np.full(len(overlaps), np.nan), where=counts > 0)
        line, = ax.plot(overlaps, means, marker="o", color=color, label=label)
        return line.get_color()

    for key, label in METHODS.items():
        color = add(accuracy, lambda row: 100 * row["scores"][shots][key]["matched_view_accuracy"], label)
        add(error, lambda row: row["relative_orbit_error"][key], label, color)
        if key in runs[0][0]["correspondence"]:
            for axis, metric in ((precision, "precision"), (recall, "recall")):
                add(axis, lambda row: row["correspondence"][key][metric], label, color)
    accuracy.set_ylabel(f"Both views correct (%) · {shots} labels/class")
    error.set_ylabel("Relative covariance error")
    precision.set_ylabel("Pair precision")
    recall.set_ylabel("Pair recall")
    for ax in axes.flat:
        ax.set_xlabel("Fraction with a true counterpart")
        ax.grid(alpha=.2)
    accuracy.legend(fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)


if __name__ == "__main__":
    main()
