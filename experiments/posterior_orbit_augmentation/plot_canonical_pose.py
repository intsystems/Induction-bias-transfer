"""Plot classification and pose recovery for transferred canonicalizers."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results/canonical_pose"
METHODS = {"raw": "Uncorrected", "oracle": "Known inverse", "pose_posterior": "Pose-only posterior",
           "hidden_point": "Teacher loss · point", "hidden_posterior": "Teacher loss · posterior",
           "hidden_shuffled": "Shuffled posterior"}
POLICIES = {"raw": "Uncorrected", "pose_point": "Full correction", "source_fixed": "Source fixed",
            "pose_selector": "Pose selector", "teacher_selector": "Teacher selector", "oracle": "Known inverse"}
POSTERIOR_POLICIES = {"raw": "Uncorrected", "pose_point": "Full correction", "point_margin": "Point margin",
                      "conditional_consensus": "Conditional samples", "global_consensus": "Global-noise samples",
                      "oracle": "Known inverse"}
SYMMETRIC_METHODS = {"raw": "Uncorrected", "pose_point": "Single prediction", "reflected_point": "Reflected prediction",
                     "symmetric_point": "Reflection-symmetric mean", "independent_point": "Independent-model mean",
                     "symmetric_posterior": "Reflection-symmetric posterior", "oracle": "Known inverse"}
IDENTITY_METHODS = {"raw": "Uncorrected", "gaussian_point": "Gaussian point", "slab_point": "Continuous component",
                    "fixed_prior": "Fixed identity mass", "identity_mean": "Conditional identity mass",
                    "identity_posterior": "Marginalized mixture", "oracle": "Known inverse"}
REFINEMENT_METHODS = {"raw": "Uncorrected", "pose_point": "One pass", "iterative": "Two passes",
                      "damped": "Damped second pass", "calibrated_first": "Calibrated first pass",
                      "calibrated_refinement": "Calibrated refinement", "oracle": "Known inverse"}
RESIDUAL_METHODS = {"raw": "Uncorrected", "pose_point": "One pass", "damped": "Damped second pass",
                    "continued_point": "Extra training · one pass", "continued_two": "Extra training · two passes",
                    "residual_trained": "Trained residual", "oracle": "Known inverse"}
RELATIVE_MODELS = {"pose": "Absolute loss", "relative": "Same-image relative loss",
                   "relative_shuffled": "Cross-image relative loss"}
RELATIVE_METHODS = {"raw": "Uncorrected",
                    **{name + "_point": label for name, label in RELATIVE_MODELS.items()},
                    "relative_posterior": "Same-image posterior", "oracle": "Known inverse"}
ANGLE_MODELS = {"pose": "Absolute loss", "angle_same": "Same-image mean error",
                "angle_shuffled": "Cross-image mean error", "angle_point": "Individual squared errors"}
ANGLE_METHODS = {"raw": "Uncorrected", **{name + "_point": label for name, label in ANGLE_MODELS.items()},
                 "oracle": "Known inverse"}
ANGLE_GATE_MODELS = {"pose": "Absolute angle", "angle_same": "Paired angle",
                     "gate_fixed": "Fixed mixture", "gate_adaptive": "Adaptive mixture",
                     "gate_shuffled": "Shuffled mixture weights"}
ANGLE_GATE_METHODS = {"raw": "Uncorrected",
                      **{name + "_point": label for name, label in ANGLE_GATE_MODELS.items()},
                      "oracle": "Known inverse"}
TRACK_METHODS = {"raw": "Uncorrected", "pose_point": "Absolute pose", "angle_same_point": "Paired angle",
                 "paired_full": "Paired pose", "angle_shuffled": "Shuffled paired angle"}
TRACK_TRAINING_METHODS = {"raw": "Uncorrected", "pose_point": "Synthetic pose loss",
                          "same_track": "Same track", "same_class": "Different tracks, same class"}
TRACK_TEACHER_METHODS = {"raw": "Uncorrected", "pose_point": "Synthetic pose loss",
                         "same_track": "Aligned teacher basis", "rotated_teacher": "Rotated teacher basis"}
TRACK_TEACHER_PARITY_METHODS = {"raw": "Uncorrected", "pose_point": "Synthetic pose loss",
                                "same_track": "Even and odd teacher blocks", "even_teacher": "Even block",
                                "odd_teacher": "Odd block"}
TRACK_PARITY_POLICY_METHODS = {"raw": "Uncorrected", "same_track": "Both blocks",
                               "parity_fixed": "Fixed mixture", "parity_soft": "Posterior mixture",
                               "parity_hard": "Posterior hard choice", "parity_oracle": "Known label action"}
TRACK_PARITY_PRIOR_ROUTE_METHODS = {"raw": "Uncorrected", "same_track": "Both blocks",
                                    "parity_fixed": "Fixed mixture", "parity_soft": "Uniform-block routing",
                                    "parity_prior_soft": "Track-prior routing", "parity_oracle": "Known label action"}
TRACK_HEAD_PRIOR_METHODS = {"raw": "Uniform parity blocks", "head_same_track": "Same-track basis",
                            "head_same_class": "Same-class basis", "head_rotated": "Rotated basis"}
TRACK_HEAD_BETWEEN_METHODS = {**TRACK_HEAD_PRIOR_METHODS, "head_track_means": "Track centroids",
                              "head_split_means": "Cross-moment of split centroids"}
TRACK_HEAD_STUDENT_METHODS = {"raw": "Uniform parity blocks", "head_same_track": "Teacher · same track",
                              "head_same_class": "Teacher · same class", "head_student_track": "Student · same track",
                              "head_student_class": "Student · same class", "head_rotated": "Rotated basis"}
TRACK_HEAD_EVIDENCE_METHODS = {"raw": "Uniform parity blocks", "head_same_class": "Same-class basis",
                               "head_equal_mix": "Equal geometry mixture", "head_evidence": "Posterior geometry mixture"}
TRACK_HEAD_BUDGET_METHODS = {"raw": "Uniform parity blocks", "head_same_class": "All pairs",
                             "head_pairs_2": "2 pairs per class", "head_pairs_8": "8 pairs per class",
                             "head_pairs_32": "32 pairs per class"}
TRACK_HEAD_BLOCK_METHODS = {"raw": "Uniform parity blocks", "head_same_class": "Both bases learned",
                            "head_learned_even": "Learned even basis", "head_learned_odd": "Learned odd basis",
                            "head_rotated": "Both bases rotated"}
TRACK_HEAD_PAIR_METHODS = {"raw": "Uniform parity blocks", "head_same_class": "Same class, different tracks",
                           "head_cross_class": "Different source classes", "head_random_pairs": "Shuffled pairings",
                           "head_same_track": "Same track", "head_rotated": "Rotated basis"}
TRACK_HEAD_DISTANCE_METHODS = {"raw": "Uniform parity blocks", "head_same_class": "All same-class pairs",
                               "head_near": "Small differences", "head_far": "Large differences",
                               "head_random_half": "Random half"}
TRACK_HEAD_NEIGHBOR_METHODS = {"raw": "Uniform parity blocks", "head_same_class": "Source class labels",
                               "head_neighbor_even": "Nearest track · even features",
                               "head_neighbor_full": "Nearest track · all features", "head_rotated": "Rotated basis"}
TRACK_HEAD_SPECTRUM_METHODS = {"raw": "Uniform parity blocks",
                               "head_same_track": "Same-track basis and spectrum",
                               "head_same_class": "Same-class basis · within-track spectrum",
                               "head_track_cross_spectrum": "Same-track basis · cross-track spectrum",
                               "head_class_cross_spectrum": "Same-class basis and spectrum",
                               "head_rotated_cross_spectrum": "Rotated basis · cross-track spectrum"}
TRACK_HEAD_PREDICTED_METHODS = {"raw": "Uniform parity blocks", "head_same_class": "Known-class pairs",
                                "head_neighbor_even": "Unrestricted nearest tracks",
                                "head_predicted_class": "Nearest · predicted class",
                                "head_known_class_nearest": "Nearest · known class"}
TRACK_HEAD_SOURCE_SPLIT_METHODS = {"raw": "Uniform parity blocks", "head_same_class": "Full training-track basis",
                                   "head_matched_train": "Matched training tracks",
                                   "head_matched_probe": "Matched held-out tracks", "head_rotated": "Rotated basis"}
TRACK_HEAD_SYNTHETIC_METHODS = {"raw": "Uniform parity blocks", "head_same_class": "Natural track pairs",
                                 "head_synthetic_pairs": "Synthetic pose pairs",
                                 "head_natural_synthetic": "Natural and synthetic mixture",
                                 "head_rotated": "Rotated basis"}
TRACK_HEAD_CROSS_PARITY_METHODS = {"raw": "Uniform parity blocks", "head_same_class": "Independent parity blocks",
                                    "head_cross_aligned": "Negative residual correlation",
                                    "head_cross_reversed": "Positive residual correlation"}
TRACK_DISTILLATION_METHODS = {"raw": "Uniform distillation loss", "track_distilled": "Natural-pair metric",
                              "rotated_distilled": "Rotated metric"}
TRACK_DISTILLATION_BUDGET_METHODS = {"raw": "All frames · uniform", "track_distilled": "All frames · natural pairs",
                                    "rotated_distilled": "All frames · rotated", "budget_uniform": "32 frames · uniform",
                                    "budget_track": "32 frames · natural pairs", "budget_rotated": "32 frames · rotated"}
TRACK_DISTILLATION_JOINT_METHODS = {"raw": "Uniform loss and head", "head_same_class": "Uniform loss · natural head",
                                   "track_distilled": "Natural loss · uniform head",
                                   "track_distilled_track_head": "Natural loss and head",
                                   "track_distilled_rotated_head": "Natural loss · rotated head",
                                   "rotated_distilled_rotated_head": "Rotated loss and head"}
TRACK_DISTILLATION_BLOCK_METHODS = {"raw": "Uniform loss", "track_distilled": "Both bases learned",
                                    "even_distilled": "Learned even basis", "odd_distilled": "Learned odd basis",
                                    "rotated_distilled": "Both bases rotated"}
TRACK_DISTILLATION_SCHEDULE_METHODS = {"raw": "Uniform throughout", "early_track": "Natural metric · early",
                                       "late_track": "Natural metric · late", "early_rotated": "Rotated metric · early",
                                       "late_rotated": "Rotated metric · late"}
TRACK_CONSISTENCY_METHODS = {"raw": "Uncorrected", "same_track": "Full correction",
                             "matched_weight": "Matched mean weight", "shuffled_weight": "Shuffled weights",
                             "consistency_weight": "Track consistency weights"}
TRACK_POOLING_METHODS = {"raw": "Raw feature mean", "feature_corrected": "Corrected feature mean",
                         "pixel_raw": "Raw pixel mean", "pixel_corrected": "Corrected pixel mean",
                         "pixel_shuffled": "Pixel mean · shuffled poses"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=RESULTS / "metrics.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "curves.png")
    args = parser.parse_args()
    results = json.loads(args.metrics.read_text())
    runs = results["runs"]
    levels = results["config"]["strengths"]
    available_shots = results["config"]["shots"]
    natural_track_pose = results["config"].get("natural_track_pose", False)
    track_pooling = results["config"].get("track_pooling", False)
    shots = str(max(available_shots) if natural_track_pose else (4 if 4 in available_shots else available_shots[0]))
    selective = results["config"].get("selective", False)
    posterior = results["config"].get("posterior_selection", False)
    symmetric = results["config"].get("symmetrize_pose", False)
    identity = results["config"].get("identity_mixture", False)
    refinement = results["config"].get("pose_refinement", False)
    trained_residual = results["config"].get("train_residual", False)
    relative_pose = results["config"].get("relative_pose", False)
    paired_angle_bias = results["config"].get("paired_angle_bias", False)
    angle_gate = results["config"].get("angle_gate", False)
    methods = METHODS
    if results["config"].get("track_distillation_schedule", False):
        methods = TRACK_DISTILLATION_SCHEDULE_METHODS
    elif results["config"].get("track_distillation_blocks", False):
        methods = TRACK_DISTILLATION_BLOCK_METHODS
    elif results["config"].get("track_distillation_joint", False):
        methods = TRACK_DISTILLATION_JOINT_METHODS
    elif results["config"].get("track_distillation_budget", False):
        methods = TRACK_DISTILLATION_BUDGET_METHODS
    elif results["config"].get("track_distillation", False):
        methods = TRACK_DISTILLATION_METHODS
    elif results["config"].get("track_parity_prior_route", False):
        methods = TRACK_PARITY_PRIOR_ROUTE_METHODS
    elif results["config"].get("track_head_cross_parity", False):
        methods = TRACK_HEAD_CROSS_PARITY_METHODS
    elif results["config"].get("track_head_synthetic", False):
        methods = TRACK_HEAD_SYNTHETIC_METHODS
    elif results["config"].get("track_head_source_split", False):
        methods = TRACK_HEAD_SOURCE_SPLIT_METHODS
    elif results["config"].get("track_head_predicted", False):
        methods = TRACK_HEAD_PREDICTED_METHODS
    elif results["config"].get("track_head_spectrum", False):
        methods = TRACK_HEAD_SPECTRUM_METHODS
    elif results["config"].get("track_head_neighbors", False):
        methods = TRACK_HEAD_NEIGHBOR_METHODS
    elif results["config"].get("track_head_distance", False):
        methods = TRACK_HEAD_DISTANCE_METHODS
    elif results["config"].get("track_head_pair_control", False):
        methods = TRACK_HEAD_PAIR_METHODS
    elif results["config"].get("track_head_blocks", False):
        methods = TRACK_HEAD_BLOCK_METHODS
    elif results["config"].get("track_head_budget", False):
        methods = TRACK_HEAD_BUDGET_METHODS
    elif results["config"].get("track_head_evidence", False):
        methods = TRACK_HEAD_EVIDENCE_METHODS
    elif results["config"].get("track_head_student", False):
        methods = TRACK_HEAD_STUDENT_METHODS
    elif results["config"].get("track_head_between", False):
        methods = TRACK_HEAD_BETWEEN_METHODS
    elif results["config"].get("track_head_prior", False):
        methods = TRACK_HEAD_PRIOR_METHODS
    elif track_pooling:
        methods = TRACK_POOLING_METHODS
    elif results["config"].get("track_parity_policy", False):
        methods = TRACK_PARITY_POLICY_METHODS
    elif results["config"].get("track_teacher_parity", False):
        methods = TRACK_TEACHER_PARITY_METHODS
    elif results["config"].get("track_teacher_basis", False):
        methods = TRACK_TEACHER_METHODS
    elif results["config"].get("track_consistency", False):
        methods = TRACK_CONSISTENCY_METHODS
    elif results["config"].get("train_track_canonicalizer", False):
        methods = TRACK_TRAINING_METHODS
    elif natural_track_pose:
        methods = TRACK_METHODS
    elif angle_gate:
        methods = ANGLE_GATE_METHODS
    elif paired_angle_bias:
        methods = ANGLE_METHODS
    elif relative_pose:
        methods = RELATIVE_METHODS
    elif trained_residual:
        methods = RESIDUAL_METHODS
    elif refinement:
        methods = REFINEMENT_METHODS
    elif identity:
        methods = IDENTITY_METHODS
    elif symmetric:
        methods = SYMMETRIC_METHODS
    elif posterior:
        methods = POSTERIOR_POLICIES
    elif selective:
        methods = POLICIES
    panels = 2 if natural_track_pose else 3
    fig, axes = plt.subplots(1, panels, figsize=(9 if natural_track_pose else 13, 4), layout="constrained")
    for ax, family in zip(axes, ("invariant", "antisymmetric")):
        for key, label in methods.items():
            accuracy = np.array([[100 * np.mean([
                task[shots]["strengths"][str(level)][key]["reflected"]
                for task in r["tasks"][family].values()]) for level in levels] for r in runs])
            ax.plot(levels, accuracy.mean(axis=0), marker="o", label=label)
        ax.set_title("Reflection preserves the label" if family == "invariant" else "Reflection changes the label")
        ax.set_ylabel(f"Reflected balanced accuracy (%), {shots} labels/class")
    axes[0].legend(fontsize=8)
    if refinement or trained_residual:
        for key in methods:
            if key in {"raw", "oracle"}:
                continue
            mse = [[r["refinement_pose_mse"][str(level)][key][0] for level in levels] for r in runs]
            axes[2].plot(levels, np.sqrt(np.mean(mse, axis=0)) * 32, marker="o", label=methods[key])
        axes[2].set_title("Inverse rotation")
        axes[2].set_ylabel("Angle RMSE (degrees)")
    elif identity:
        for key, label in (("hurdle_p_zero", "Mixture model"), ("gaussian_p_zero", "Gaussian control")):
            probs = [[r["identity_calibration"][str(level)][key] for level in levels] for r in runs]
            axes[2].plot(levels, np.mean(probs, axis=0), marker="o", label=label)
        axes[2].set_title("Predicted identity mass")
        axes[2].set_ylabel("Probability of zero distortion")
        axes[2].set_ylim(0, 1)
    elif symmetric:
        for key in ("pose_point", "reflected_point", "symmetric_point", "symmetric_precision", "independent_point"):
            mse = [[r["symmetric_pose_mse"][str(level)][key][0] for level in levels] for r in runs]
            axes[2].plot(levels, np.sqrt(np.mean(mse, axis=0)) * 32, marker="o",
                         label=methods.get(key, "Precision-weighted symmetry"))
        axes[2].set_title("Inverse rotation")
        axes[2].set_ylabel("Angle RMSE (degrees)")
    elif posterior:
        for key in ("point_margin", "conditional_consensus", "global_consensus"):
            steps = [[np.mean([task[shots]["strengths"][str(level)][key]["mean_step"]
                               for task in r["tasks"]["antisymmetric"].values()])
                      for level in levels] for r in runs]
            axes[2].plot(levels, np.mean(steps, axis=0), marker="o", label=methods[key])
        axes[2].set_title("Selected correction when reflection changes labels")
        axes[2].set_ylabel("Mean correction fraction")
        axes[2].set_ylim(0, 1)
    elif selective:
        for key in ("pose_selector", "teacher_selector"):
            steps = [[r["policy_statistics"][str(level)][key]["mean_step"] for level in levels] for r in runs]
            axes[2].plot(levels, np.mean(steps, axis=0), marker="o", label=POLICIES[key])
        axes[2].set_title("Selected correction")
        axes[2].set_ylabel("Mean correction fraction")
        axes[2].set_ylim(0, 1)
    elif not natural_track_pose:
        pose_models = {"pose": "Pose-only training", "hidden": "Teacher loss"}
        if angle_gate:
            pose_models = ANGLE_GATE_MODELS
        elif paired_angle_bias:
            pose_models = ANGLE_MODELS
        elif relative_pose:
            pose_models = RELATIVE_MODELS
        for key, label in pose_models.items():
            mse = np.array([[r["pose_calibration"][str(level)][key]["mse"] for level in levels] for r in runs])
            axes[2].plot(levels, np.sqrt(mse.mean(axis=0)[:, 0]) * 32, marker="o", label=label)
        axes[2].set_title("Inverse rotation")
        axes[2].set_ylabel("Angle RMSE (degrees)")
    if not natural_track_pose:
        axes[2].legend(fontsize=8)
    label = "Relative frame position" if natural_track_pose else "Pose distortion scale"
    if track_pooling:
        label = "Frames per track"
    for ax in axes:
        ax.set_xlabel(label)
        ax.grid(alpha=.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)


if __name__ == "__main__":
    main()
