"""Plot held-out shifts by precision and by teacher training condition."""

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
    teachers = list(data["runs"][0]["encoders"])
    fig, axes = plt.subplots(len(teachers), len(shots), figsize=(5.2 * len(shots), 6.8),
                             squeeze=False, sharex=True, sharey="row", layout="constrained")
    for row, teacher in enumerate(teachers):
        for col, shot in enumerate(shots):
            ax = axes[row, col]
            arms = [run["encoders"][teacher]["shots"][str(shot)] for run in data["runs"]]
            for name in arms[0]:
                epochs = [point["epoch"] for point in arms[0][name]]
                values = 100 * np.array([[point["shifted_accuracy"] for point in arm[name]]
                                         for arm in arms])
                mean, sd = values.mean(axis=0), values.std(axis=0)
                line, = ax.plot(epochs, mean, label=name.replace("_", " "))
                ax.fill_between(epochs, mean - sd, mean + sd, color=line.get_color(), alpha=0.1)
            ax.set_title(f"{teacher.capitalize()} teacher · {shot} labels per class")
            ax.grid(alpha=0.20)
            if col == 0:
                ax.set_ylabel("Far-shift accuracy (%)")
            if row == len(teachers) - 1:
                ax.set_xlabel("Target training epoch")
    axes[0, -1].legend(fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)


if __name__ == "__main__":
    main()
