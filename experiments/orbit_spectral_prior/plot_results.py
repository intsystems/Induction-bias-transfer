"""Plot seed-paired learning curves for the MNIST transfer experiment."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
LABELS = {
    "scratch": "Scratch",
    "init": "Feature initialization",
    "isotropic": "Isotropic prior",
    "orbit_prior": "Orbit prior",
    "reversed_prior": "Reversed prior",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=HERE / "results" / "metrics.json")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "curves.png")
    args = parser.parse_args()
    data = json.loads(args.metrics.read_text())
    shots = data["config"]["shots"]
    fig, axes = plt.subplots(2, len(shots), figsize=(5.4 * len(shots), 6), squeeze=False,
                             sharex=True, sharey="row", layout="constrained")
    for column, count in enumerate(shots):
        axes[0, column].set_title(f"{count} labeled samples per class")
        for row, (metric, label) in enumerate(
            [("shifted_accuracy", "Shifted accuracy (%)"), ("clean_accuracy", "Clean accuracy (%)")]
        ):
            axis = axes[row, column]
            if column == 0:
                axis.set_ylabel(label)
            for method, name in LABELS.items():
                curves = [run["shots"][str(count)][method] for run in data["runs"]]
                epochs = [point["epoch"] for point in curves[0]]
                values = 100 * np.array([[point[metric] for point in curve] for curve in curves])
                mean, spread = values.mean(axis=0), values.std(axis=0)
                line, = axis.plot(epochs, mean, label=name)
                axis.fill_between(epochs, mean - spread, mean + spread, color=line.get_color(), alpha=0.12)
            axis.grid(alpha=0.2)
            if row == 1:
                axis.set_xlabel("Target training epoch")
    axes[0, -1].legend(fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)


if __name__ == "__main__":
    main()
