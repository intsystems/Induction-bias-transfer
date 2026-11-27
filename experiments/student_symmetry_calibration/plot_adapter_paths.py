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
TARGET_ROW_METHODS = {"path_matched": "Path · uniform output", "rotated_matched": "Rotated path · uniform output",
                      "target_rows_path_aligned": "Path · target centroids",
                      "target_rows_path_rotated": "Path · rotated centroids",
                      "target_rows_rotated_aligned": "Rotated path · target centroids",
                      "target_rows_rotated_rotated": "Rotated path · rotated centroids"}
TARGET_LABEL_METHODS = {"path_matched": "Uniform output", "target_rows_path_aligned": "True centroids",
                        "target_rows_path_permuted_labels": "Permuted-label centroids",
                        "target_rows_path_unlabeled_pca": "Unlabeled PCA",
                        "target_rows_path_rotated": "Rotated centroids"}
TARGET_SCATTER_METHODS = {"path_matched": "Uniform output", "target_rows_path_aligned": "Target centroids",
                          "target_rows_path_within_scatter": "Within-class scatter adjustment",
                          "target_rows_path_total_scatter": "Total scatter adjustment",
                          "target_rows_path_rotated": "Rotated centroids"}
COUPLING_METHODS = {"path_matched": "Uniform output",
                    "target_rows_path_aligned_product": "Product · target centroids",
                    "target_rows_path_aligned_additive": "Sum · target centroids",
                    "target_rows_path_rotated_product": "Product · rotated centroids",
                    "target_rows_path_rotated_additive": "Sum · rotated centroids"}
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
AXIS_METHODS = {"axis_horizontal_matched": "Horizontal-reflection path",
                "axis_vertical_matched": "Vertical-reflection path",
                "axis_rotated_matched": "Rotated basis"}
AXIS_MIXTURE_METHODS = {"axis_horizontal_matched": "Horizontal path", "axis_vertical_matched": "Vertical path",
                        "axis_arithmetic_matched": "Arithmetic mean", "axis_log_mean_matched": "Log-space mean",
                        "axis_arithmetic_rotated_matched": "Rotated arithmetic mean",
                        "axis_log_mean_rotated_matched": "Rotated log-space mean"}
AXIS_ENSEMBLE_METHODS = {"axis_arithmetic_matched": "Single fit · mean precision",
                         "axis_ensemble_logits": "Separate fits · mean logits",
                         "axis_ensemble_weights": "Separate fits · mean weights",
                         "axis_rotated_ensemble_logits": "Rotated fits · mean logits",
                         "axis_rotated_ensemble_weights": "Rotated fits · mean weights"}
AXIS_OUTPUT_METHODS = {"axis_arithmetic_matched": "Mean precision · uniform output",
                       "target_rows_axis_arithmetic_aligned": "Mean precision · target centroids",
                       "target_rows_axis_arithmetic_rotated": "Mean precision · rotated centroids",
                       "target_rows_axis_horizontal_aligned": "Horizontal path · target centroids",
                       "target_rows_axis_vertical_aligned": "Vertical path · target centroids"}
AXIS_DIRECT_METHODS = {"target_rows_axis_arithmetic_aligned": "Quarter-rotation path",
                       "target_rows_axis_arithmetic_rotated_aligned": "Rotated path basis",
                       "target_rows_axis_direct_45_aligned": "Direct source pairs · 45°",
                       "target_rows_axis_direct_135_aligned": "Direct source pairs · 135°"}
ANGLE_MIXTURE_METHODS = {"target_rows_axis_arithmetic_aligned": "Quarter-rotation path",
                         "target_rows_axis_angle_average_aligned": "Three-angle mean",
                         "target_rows_axis_angle_matched_aligned": "Three-angle basis · path spectrum",
                         "target_rows_axis_angle_rotated_aligned": "Three-angle mean · rotated"}


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
    elif data["config"].get("target_row_prior", False):
        methods = TARGET_ROW_METHODS
    elif data["config"].get("target_row_labels", False):
        methods = TARGET_LABEL_METHODS
    elif data["config"].get("target_row_scatter", False):
        methods = TARGET_SCATTER_METHODS
    elif data["config"].get("target_row_coupling", False):
        methods = COUPLING_METHODS
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
    panels = (("clean", "Clean"), ("composition", "Clean and composition"), ("all_views", "All eight views"))
    if data["config"].get("path_axes", False):
        methods = AXIS_METHODS
        panels = (("composition", "Clean and horizontal-reflection composition"),
                  ("vertical_composition", "Clean and vertical-reflection composition"),
                  ("all_views", "All eight views"))
    mixture = data["config"].get("path_axis_mixture", False)
    ensemble = data["config"].get("path_axis_ensemble", False)
    axis_output = data["config"].get("path_axis_output", False)
    if mixture or ensemble or axis_output:
        methods = AXIS_MIXTURE_METHODS
        if ensemble:
            methods = AXIS_ENSEMBLE_METHODS
        if axis_output:
            methods = AXIS_OUTPUT_METHODS
        panels = (("composition", "Horizontal-reflection composition"),
                  ("vertical_composition", "Vertical-reflection composition"),
                  ("both_compositions", "Clean and both compositions"))
    if data["config"].get("path_axis_interpolation", False):
        panels = (("both_compositions_45", "Both compositions · 45°"),
                  ("both_compositions", "Both compositions · 90°"),
                  ("both_compositions_135", "Both compositions · 135°"))
    if data["config"].get("path_axis_direct", False):
        methods = AXIS_DIRECT_METHODS
    if data["config"].get("path_angle_mixture", False):
        methods = ANGLE_MIXTURE_METHODS
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    if source := data["config"].get("source_domain"):
        label = {"cifar": "CIFAR-10", "gtsrb": "GTSRB"}[source]
        fig.suptitle(f"Source {label} · Target CIFAR-10")
    for ax, (metric, title) in zip(axes, panels, strict=True):
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
