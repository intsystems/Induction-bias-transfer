"""Plot few-shot accuracy and exact matching recovery across reflection scenarios."""

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
    scenarios = list(data["scenarios"])
    fig, axes = plt.subplots(2, len(scenarios), figsize=(5 * len(scenarios), 7),
                             sharex=True, sharey="row", squeeze=False, layout="constrained")
    for col, scenario in enumerate(scenarios):
        runs = [run["scenarios"][scenario] for run in data["runs"]]
        methods = dict.fromkeys(method for n in shots for method in runs[0][str(n)])
        palette = plt.get_cmap("tab20").colors
        colors = {method: color for method, color in zip(methods, (*palette[::2], *palette[1::2]))}
        axes[0, col].set_title(scenario.replace("_", " ").capitalize())
        for row, (metric, label) in enumerate(
            [("mirror_accuracy", "Balanced mirror accuracy (%)"),
             ("class_mapping_accuracy", "Correct class mappings (%)")]
        ):
            ax = axes[row, col]
            for method in methods:
                available = [n for n in shots if metric in runs[0][str(n)].get(method, {})]
                if not available:
                    continue
                values = 100 * np.array(
                    [[run[str(n)][method][metric] for n in available] for run in runs]
                )
                mean, sd = values.mean(axis=0), values.std(axis=0)
                line, = ax.plot(available, mean, marker="o", color=colors[method],
                                label=method.replace("_", " "))
                ax.fill_between(available, mean - sd, mean + sd, color=line.get_color(), alpha=0.09)
            ax.set_xticks(shots)
            ax.grid(alpha=0.2)
            if col == 0:
                ax.set_ylabel(label)
            if row == 1:
                ax.set_xlabel("Labeled sign tracks per class")
    axes[0, -1].legend(fontsize=7, loc="lower right", ncols=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)


if __name__ == "__main__":
    main()
