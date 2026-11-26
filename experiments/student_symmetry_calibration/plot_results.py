"""Plot CIFAR-10 one-view prior calibration and the two-view control."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=HERE / "results/metrics.json")
    parser.add_argument("--output", type=Path, default=HERE / "results/curves.png")
    args = parser.parse_args()
    data = json.loads(args.metrics.read_text())
    shots = data["config"]["shots"]
    runs = [run["shots"] for run in data["runs"]]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.3), layout="constrained")
    metrics = (("clean_accuracy", "Clean accuracy (%)"),
               ("mirror_accuracy", "Mirror accuracy (%)"),
               ("matched_view_accuracy", "Both views correct (%)"))
    for ax, (metric, label) in zip(axes, metrics, strict=True):
        for method in runs[0][str(shots[0])]:
            values = 100 * np.array([[run[str(n)][method][metric] for n in shots] for run in runs])
            center, spread = values.mean(axis=0), values.std(axis=0)
            line, = ax.plot(shots, center, marker="o", label=method.replace("_", " "))
            ax.fill_between(shots, center - spread, center + spread, color=line.get_color(), alpha=0.10)
        ax.set_xticks(shots)
        ax.set_ylabel(label)
        ax.set_xlabel("Labeled images per class")
        ax.grid(alpha=0.2)
    axes[-1].legend(fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)


if __name__ == "__main__":
    main()
