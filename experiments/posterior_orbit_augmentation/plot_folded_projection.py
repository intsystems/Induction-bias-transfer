"""Plot projected-student latency before and after folding the weights."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results" / "track_distillation_projection_folded"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=RESULTS / "metrics.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "curves.png")
    args = parser.parse_args()
    data = json.loads(args.metrics.read_text())
    runs = [run["folded_projection"]["latency_ms"] for run in data["runs"]]
    sizes = sorted(map(int, runs[0]))
    methods = {"two_view": "Batched two-view projection", "folded": "Folded weights"}
    latency = {key: np.array([[run[str(size)][key] for size in sizes] for run in runs]) for key in methods}
    figure, (times, ratios) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for method, label in methods.items():
        low, center, high = np.percentile(latency[method], [25, 50, 75], axis=0)
        times.plot(sizes, center, marker="o", label=label)
        times.fill_between(sizes, low, high, alpha=.15)
    ratio = latency["folded"] / latency["two_view"]
    low, center, high = np.percentile(ratio, [25, 50, 75], axis=0)
    ratios.plot(sizes, center, marker="o")
    ratios.fill_between(sizes, low, high, alpha=.15)
    ratios.axhline(1, color=".5", linewidth=1)
    times.set_ylabel("Latency per batch (ms)")
    ratios.set_ylabel("Folded latency ÷ two-view latency")
    times.legend(fontsize=8)
    for axis in (times, ratios):
        axis.set_xlabel("Images per batch")
        axis.set_xticks(sizes)
        axis.grid(alpha=.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170)


if __name__ == "__main__":
    main()
