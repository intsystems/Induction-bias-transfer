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


def plot_coordinate_check(data):
    fields = (("matched_view_accuracy", "Both views correct (%)"),
              ("clean_accuracy", "Clean accuracy (%)"))
    shots = data["config"]["shots"]
    runs = [run["coordinate_comparison"] for run in data["runs"]]
    modes = ("native", "mixed")
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.3), sharex=True, sharey="row",
                             layout="constrained")
    for col, mode in enumerate(modes):
        axes[0, col].set_title(f"{mode.capitalize()} latent coordinates")
        for row, (metric, label) in enumerate(fields):
            ax = axes[row, col]
            for method in runs[0][mode][str(shots[0])]:
                values = 100 * np.asarray([
                    [run[mode][str(n)][method][metric] for n in shots] for run in runs
                ])
                center, spread = values.mean(axis=0), values.std(axis=0)
                line, = ax.plot(shots, center, marker="o", label=method.replace("_", " "))
                ax.fill_between(shots, center - spread, center + spread,
                                color=line.get_color(), alpha=0.10)
            ax.set_xticks(shots)
            ax.grid(alpha=0.2)
            if col == 0:
                ax.set_ylabel(label)
            if row == 1:
                ax.set_xlabel("Labeled images per class")
    axes[0, -1].legend(fontsize=8)
    return fig


def plot_variance_budgets(data):
    choices = sorted(data["config"]["variance_budgets"], reverse=True)
    shots = data["config"]["shots"]
    runs = [run["coordinate_comparison"]["native"] for run in data["runs"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), layout="constrained")
    for ax, (metric, label) in zip(axes, [
        ("matched_view_accuracy", "Both views correct"),
        ("clean_accuracy", "Clean accuracy"),
    ], strict=True):
        for n in shots:
            values = np.array([
                [100 * (run[str(n)][f"natural_budget_{b:g}"][metric] -
                        run[str(n)][f"natural_budget_rotated_{b:g}"][metric]) for b in choices]
                for run in runs
            ])
            center, spread = values.mean(axis=0), values.std(axis=0)
            line, = ax.plot(100 * np.array(choices), center, marker="o", label=f"{n} labels per class")
            ax.fill_between(100 * np.array(choices), center - spread, center + spread,
                            alpha=0.12, color=line.get_color())
        ax.axhline(0, color="black", alpha=0.35, linewidth=1)
        ax.set_xticks(100 * np.array(choices))
        ax.invert_xaxis()
        ax.set_xlabel("Remaining source orbit variance (%)")
        ax.set_ylabel(f"{label} · orbit minus rotated (pp)")
        ax.grid(alpha=0.2)
    axes[-1].legend(fontsize=8)
    return fig


def plot_pair_counts(data):
    config = data["config"]
    full_count = 4 * config["probe_examples"]
    counts = sorted(config["calibration_pairs"]) + [full_count]
    budget = min(config["variance_budgets"])
    shots = config["shots"]
    seeds = data["runs"]
    chosen = f"natural_budget_{budget:g}"
    rotated = f"natural_budget_rotated_{budget:g}"
    fig, axes = plt.subplots(1, 2, figsize=(10.9, 4.2), layout="constrained")
    for ax, (name, label) in zip(axes, [
        ("matched_view_accuracy", "Both views correct"),
        ("clean_accuracy", "Clean accuracy"),
    ], strict=True):
        for n in shots:
            values = []
            for run in seeds:
                complete = run["coordinate_comparison"]["native"][str(n)]
                subsamples = [run["calibration_pair_results"][str(count)]["coordinate_comparison"]
                              ["native"][str(n)] for count in counts[:-1]]
                values.append([100 * (metrics[chosen][name] - metrics[rotated][name])
                               for metrics in (*subsamples, complete)])
            delta = np.array(values)
            center, spread = delta.mean(axis=0), delta.std(axis=0)
            line, = ax.plot(counts, center, marker="o", label=f"{n} labels per class")
            ax.fill_between(counts, center - spread, center + spread,
                            color=line.get_color(), alpha=0.12)
        ax.axhline(0, color="black", alpha=0.35, linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(counts, [str(n) for n in counts])
        ax.set_xlabel("Source calibration pairs")
        ax.set_ylabel(f"{label} · orbit minus rotated (pp)")
        ax.grid(alpha=0.2)
    axes[-1].legend(fontsize=8)
    return fig


def plot_source_domains(data):
    shots = data["config"]["shots"]
    budget = min(data["config"]["variance_budgets"])
    methods = ("ordinary_iso", "ordinary_orbit", "natural_orbit",
               f"natural_budget_{budget:g}", f"natural_budget_rotated_{budget:g}")
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.4), sharex=True, sharey="row",
                             layout="constrained")
    for col, domain in enumerate(("CIFAR-10", "GTSRB")):
        if data["config"].get("match_source_ranks"):
            key = ("cifar", "traffic")[col]
            runs = [r["rank_matched"][key] for r in data["runs"]]
            axes[0, col].set_title(f"Rank-matched teacher · {domain}")
        else:
            runs = [r if col == 0 else r["traffic_source_result"] for r in data["runs"]]
            axes[0, col].set_title(f"Source teacher · {domain}")
        runs = [r["coordinate_comparison"]["native"] for r in runs]
        for row, (metric, label) in enumerate([
                ("matched_view_accuracy", "Both views correct (%)"),
                ("clean_accuracy", "Clean accuracy (%)")]):
            ax = axes[row, col]
            for method in methods:
                values = 100 * np.array([[r[str(n)][method][metric] for n in shots] for r in runs])
                center, spread = values.mean(axis=0), values.std(axis=0)
                line, = ax.plot(shots, center, marker="o", label=method.replace("_", " "))
                ax.fill_between(shots, center - spread, center + spread,
                                color=line.get_color(), alpha=0.10)
            ax.set_xticks(shots)
            ax.grid(alpha=0.2)
            if col == 0:
                ax.set_ylabel(label)
            if row == 1:
                ax.set_xlabel("Labeled target images per class")
    axes[0, -1].legend(fontsize=8)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=HERE / "results/metrics.json")
    parser.add_argument("--output", type=Path, default=HERE / "results/curves.png")
    args = parser.parse_args()
    data = json.loads(args.metrics.read_text())
    if data["config"].get("gauge_check"):
        if data["config"].get("compare_source_domains"):
            fig = plot_source_domains(data)
        elif data["config"].get("calibration_pairs"):
            fig = plot_pair_counts(data)
        elif data["config"].get("variance_budgets"):
            fig = plot_variance_budgets(data)
        else:
            fig = plot_coordinate_check(data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=170)
        return
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
