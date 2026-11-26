"""Plot CIFAR-10 one-view prior calibration and the two-view control."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent


def plot_precision_sweep(data):
    strengths = data["config"]["precision_strengths"]
    runs = [run["shots"] for run in data["runs"]]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.4), layout="constrained")
    fields = [
        ("matched_view_accuracy", "Both views correct"),
        ("clean_accuracy", "Clean accuracy"),
        ("agreement", "Paired predictions agree"),
    ]
    for ax, (metric, label) in zip(axes, fields, strict=True):
        for budget in data["config"]["shots"]:
            paired = []
            for run in runs:
                row = run[str(budget)]
                paired.append([100 * (row[f"orbit_strength_{s:g}"][metric] -
                                      row[f"rotated_strength_{s:g}"][metric]) for s in strengths])
            deltas = np.array(paired)
            center, spread = deltas.mean(axis=0), deltas.std(axis=0)
            line, = ax.plot(strengths, center, marker="o", label=f"{budget} labels per class")
            ax.fill_between(strengths, center - spread, center + spread,
                            color=line.get_color(), alpha=0.12)
        ax.axhline(0, color="black", alpha=0.35, linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(strengths, [f"{s:g}" for s in strengths])
        ax.set_ylabel(f"{label} · orbit minus rotated (pp)")
        ax.set_xlabel("Prior precision multiplier")
        ax.grid(alpha=0.18)
    axes[-1].legend(fontsize=8)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=HERE / "results/metrics.json")
    parser.add_argument("--output", type=Path, default=HERE / "results/curves.png")
    args = parser.parse_args()
    data = json.loads(args.metrics.read_text())
    if data["config"].get("precision_strengths"):
        fig = plot_precision_sweep(data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=170)
        return
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
