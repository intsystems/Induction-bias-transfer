"""Plot metrics saved by experiment.py without running training."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="results/metrics.json")
    parser.add_argument("--output", default="results/curves.png")
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text())

    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    metrics = [
        ("test_mse", "Original test MSE"),
        ("shuffled_test_mse", "Shuffled test MSE"),
        ("invariance_mae", "Permutation gap (MAE)"),
    ]
    for result in payload["results"]:
        history = result["history"]
        weight = result["consistency_weight"]
        label = "Baseline" if weight == 0 else f"Consistency weight = {weight:g}"
        for axis, (metric, title) in zip(axes, metrics):
            axis.plot(
                [row["epoch"] for row in history],
                [row[metric] for row in history],
                label=label,
            )
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Error")
            axis.grid(alpha=0.3)
    axes[0].legend()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"Plot saved to {output}")


if __name__ == "__main__":
    main()
