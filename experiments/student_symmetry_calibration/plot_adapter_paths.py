"""Plot test accuracy after adapting the student with different source priors."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results/adapter_paths"
METHODS = {"frozen": "Frozen adapter", "initialization": "Initialization only",
           "isotropic": "Isotropic prior", "split": "Separate generator budgets",
           "rotated": "Rotated path spectrum", "path": "Path prior"}
MATCHED_METHODS = {"isotropic": "Isotropic reference", "rotated": "Rotated spectrum",
                   "path": "Path prior", "rotated_matched": "Rotated · matched displacement",
                   "path_matched": "Path · matched displacement", "split_matched": "Separate · matched displacement"}
ROW_METHODS = {"path_matched": "Path · uniform rows", "split_matched": "Separate · uniform rows",
               "teacher_rows": "Path · teacher rows", "rotated_rows": "Path · rotated rows",
               "reversed_rows": "Path · reversed rows"}
ACTION_METHODS = {"path_matched": "Input path prior", "split_matched": "Separate generator budgets",
                  "teacher_action": "Teacher action", "rotated_action": "Rotated teacher action",
                  "identity_action": "Identity output action"}
PROBE_METHODS = {"path_matched": "GTSRB paths", "rotated_matched": "GTSRB rotated",
                 "auxiliary_path_matched": "CIFAR-10 paths", "auxiliary_rotated_matched": "CIFAR-10 rotated",
                 "split_matched": "GTSRB generators", "auxiliary_split_matched": "CIFAR-10 generators"}
HYBRID_METHODS = {"split_matched": "Separate generators", "hybrid_quarter_matched": "25% path precision",
                  "hybrid_half_matched": "50% path precision", "hybrid_three_quarters_matched": "75% path precision",
                  "path_matched": "Path prior", "hybrid_covariance_matched": "Covariance mixture",
                  "hybrid_rotated_matched": "Rotated 50% mixture"}
SPECTRAL_METHODS = {"path_matched": "Path prior", "global_quarter_split_matched": "25% separate precision",
                    "block_high_matched": "High-precision block", "block_high_rotated_matched": "High block · rotated",
                    "block_low_matched": "Low-precision block", "block_low_rotated_matched": "Low block · rotated",
                    "split_matched": "Separate generators"}
ORIENTATION_METHODS = {"path_matched": "Path prior", "global_quarter_split_matched": "25% separate precision",
                       "block_high_matched": "High block · aligned", "block_high_signed_matched": "High block · sign flips",
                       "block_high_diagonal_matched": "High block · diagonal",
                       "block_high_rotated_matched": "High block · rotated"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=RESULTS / "metrics.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "curves.png")
    args = parser.parse_args()
    data = json.loads(args.metrics.read_text())
    shots = data["config"]["shots"]
    methods = MATCHED_METHODS if data["config"].get("match_update_rms", False) else METHODS
    if data["config"].get("teacher_row_prior", False):
        methods = ROW_METHODS
    elif data["config"].get("intertwining_prior", False):
        methods = ACTION_METHODS
    if data["config"].get("compare_probe_domain", False):
        methods = PROBE_METHODS
    if data["config"].get("hybrid_prior", False):
        methods = HYBRID_METHODS
    if data["config"].get("spectral_blocks", False):
        methods = SPECTRAL_METHODS
    if data["config"].get("spectral_orientation", False):
        methods = ORIENTATION_METHODS
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    if source := data["config"].get("source_domain"):
        label = {"cifar": "CIFAR-10", "gtsrb": "GTSRB"}[source]
        fig.suptitle(f"Source {label} · Target CIFAR-10")
    for ax, (metric, title) in zip(axes, (("clean", "Clean"),
                                         ("composition", "Clean and composition"),
                                         ("all_views", "All eight views")), strict=True):
        for name, label in methods.items():
            values = [100 * np.mean([episode[name][metric] for run in data["runs"] for episode in run["shots"][str(n)]])
                      for n in shots]
            ax.plot(shots, values, marker="o", label=label)
        ax.set_title(title)
        ax.set_xlabel("Labels per class")
        ax.set_ylabel("Correct on every indicated view (%)")
        ax.set_xticks(shots)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)


if __name__ == "__main__":
    main()
