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
TRACK_DISTILLATION_WIDTH_METHODS = {"raw": "Width 192 · uniform", "track_distilled": "Width 192 · natural metric",
                                   "rotated_distilled": "Width 192 · rotated", "narrow_uniform": "Width 48 · uniform",
                                   "narrow_track": "Width 48 · natural metric", "narrow_rotated": "Width 48 · rotated"}
TRACK_DISTILLATION_SYNTHETIC_METHODS = {"raw": "Uniform loss", "track_distilled": "Natural pairs",
                                       "synthetic_distilled": "Synthetic pairs", "mixed_distilled": "Mixed pairs",
                                       "rotated_distilled": "Rotated directions"}
TRACK_DISTILLATION_LATE_HEAD_METHODS = {"raw": "All layers · uniform", "late_track": "All layers · natural",
                                       "late_rotated": "All layers · rotated", "head_uniform": "Output only · uniform",
                                       "head_track": "Output only · natural", "head_rotated": "Output only · rotated"}
TRACK_DISTILLATION_PROJECTION_METHODS = {"raw": "Uniform loss · one view", "raw_aligned": "Uniform loss · teacher action",
                                        "raw_rotated": "Uniform loss · rotated action", "late_track": "Natural loss · one view",
                                        "late_track_aligned": "Natural loss · teacher action",
                                        "late_track_rotated": "Natural loss · rotated action"}
TRACK_DISTILLATION_PROJECTION_TRAIN_METHODS = {"raw_aligned": "Uniform loss · project for prediction",
                                              "late_track_aligned": "Natural loss · project for prediction",
                                              "trained_uniform_aligned": "Uniform loss · project in training",
                                              "trained_late_aligned": "Natural loss · project in training"}
TRACK_DISTILLATION_PROJECTION_CALIBRATED_METHODS = {"raw_aligned": "Uniform loss · teacher action",
                                                   "raw_rotated": "Uniform loss · rotated action",
                                                   "raw_calibrated": "Uniform loss · student action",
                                                   "late_track_aligned": "Natural loss · teacher action",
                                                   "late_track_rotated": "Natural loss · rotated action",
                                                   "late_track_calibrated": "Natural loss · student action"}
TRACK_DISTILLATION_PROJECTION_SHRINK_METHODS = {"raw_aligned": "Uniform loss · teacher action",
                                               "raw_sparse": "Uniform loss · sparse calibration",
                                               "raw_shrunk": "Uniform loss · teacher + sparse calibration",
                                               "late_track_aligned": "Natural loss · teacher action",
                                               "late_track_sparse": "Natural loss · sparse calibration",
                                               "late_track_shrunk": "Natural loss · teacher + sparse calibration"}
TRACK_DISTILLATION_PROJECTION_SCHEDULE_METHODS = {"late_track_aligned": "Prediction only",
                                                 "project_early_aligned": "Early training",
                                                 "project_late_aligned": "Late training",
                                                 "project_all_aligned": "Throughout training"}
TRACK_DISTILLATION_PROJECTION_COMPRESS_METHODS = {"late_track_aligned": "Standard first stage · two views",
                                                 "late_track_compressed": "Standard first stage · one view",
                                                 "trained_late_aligned": "Projected first stage · two views",
                                                 "trained_late_compressed": "Projected first stage · one view"}
TRACK_DISTILLATION_PROJECTION_TIMING_WEIGHTS_METHODS = {
    "raw_aligned": "Uniform loss · prediction only", "uniform_early_aligned": "Uniform loss · early",
    "uniform_late_aligned": "Uniform loss · late", "late_track_aligned": "Natural loss · prediction only",
    "natural_early_aligned": "Natural loss · early", "natural_late_aligned": "Natural loss · late",
}
TRACK_DISTILLATION_PROJECTION_TERMINAL_METHODS = {
    "late_track_aligned": "Prediction only", "terminal_none_aligned": "Early block only",
    "terminal_one_aligned": "Early block + final epoch", "terminal_part_aligned": "Early and late blocks",
    "terminal_all_aligned": "Late block only",
}
TRACK_DISTILLATION_PROJECTION_CANONICAL_METHODS = {
    "trained_late_aligned": "No pose correction",
    "canonical_aligned": "Independent pose correction",
    "canonical_symmetric_aligned": "Reflection-consistent pose correction",
}
TRACK_DISTILLATION_PROJECTION_JOINT_HEAD_METHODS = {
    "late_track_aligned": "Prediction-only projection · isotropic head",
    "late_track_natural_head_aligned": "Prediction-only projection · natural head",
    "late_track_rotated_head_aligned": "Prediction-only projection · rotated head",
    "trained_late_aligned": "Training projection · isotropic head",
    "trained_late_natural_head_aligned": "Training projection · natural head",
    "trained_late_rotated_head_aligned": "Training projection · rotated head",
}
TRACK_DISTILLATION_PROJECTION_CANONICAL_TRAIN_METHODS = {
    "trained_late_aligned": "Original training · original prediction",
    "trained_late_canonical_aligned": "Original training · corrected prediction",
    "canonical_trained_aligned": "Corrected training · original prediction",
    "canonical_trained_canonical_aligned": "Corrected training · corrected prediction",
}
TRACK_DISTILLATION_PROJECTION_SINGLE_VIEW_PENALTY_METHODS = {
    "late_track": "Ordinary loss · one view", "penalty_free": "No penalty · one view",
    "even_penalty": "Even penalty · one view", "odd_penalty": "Odd penalty · one view",
    "even_penalty_aligned": "Even penalty · two views", "odd_penalty_aligned": "Odd penalty · two views",
}
TRACK_DISTILLATION_PROJECTION_STUDENT_PRIOR_METHODS = {
    "late_track_natural_head_aligned": "Prediction-only projection · teacher prior",
    "late_track_student_head_aligned": "Prediction-only projection · student prior",
    "late_track_rotated_head_aligned": "Prediction-only projection · rotated prior",
    "trained_late_natural_head_aligned": "Training projection · teacher prior",
    "trained_late_student_head_aligned": "Training projection · student prior",
    "trained_late_rotated_head_aligned": "Training projection · rotated prior",
}
TRACK_DISTILLATION_PROJECTION_HEAD_POLICY_METHODS = {
    "trained_late_natural_head_aligned": "Natural prior · evidence weights",
    "trained_late_natural_head_aligned_equal": "Natural prior · equal mixture",
    "trained_late_natural_head_aligned_hard": "Natural prior · hard choice",
    "trained_late_rotated_head_aligned": "Rotated prior · evidence weights",
    "trained_late_rotated_head_aligned_equal": "Rotated prior · equal mixture",
    "trained_late_rotated_head_aligned_hard": "Rotated prior · hard choice",
}
TRACK_DISTILLATION_PROJECTION_HEAD_VALIDATION_METHODS = {
    "trained_late_natural_head_aligned": "Natural prior · centered evidence",
    "trained_late_natural_head_aligned_equal": "Natural prior · equal weights",
    "trained_late_natural_head_aligned_validation": "Natural prior · paired validation",
    "trained_late_rotated_head_aligned": "Rotated prior · centered evidence",
    "trained_late_rotated_head_aligned_equal": "Rotated prior · equal weights",
    "trained_late_rotated_head_aligned_validation": "Rotated prior · paired validation",
}
TRACK_DISTILLATION_PROJECTION_RELATIVE_AUGMENTATION_METHODS = {
    "trained_late_aligned": "No pose augmentation",
    "relative_augmented_aligned": "Relative pose from paired frame",
    "relative_shuffled_aligned": "Same relative poses shuffled within class",
}
TRACK_DISTILLATION_PROJECTION_APPROX_TEACHER_METHODS = {
    "raw_aligned": "Estimated action · prediction only",
    "estimated_training_aligned": "Estimated action in training",
    "rotated_training_aligned": "Rotated action in training",
}
TRACK_DISTILLATION_PROJECTION_CONFIDENT_TEACHER_METHODS = {
    "raw_aligned": "Ordinary training · full projection",
    "estimated_training_aligned": "Full training projection · full projection",
    "confident_training_aligned": "Weighted training projection · full projection",
    "raw_confident": "Ordinary training · weighted projection",
    "estimated_training_confident": "Full training projection · weighted projection",
    "confident_training_confident": "Weighted training and prediction projection",
}
TRACK_DISTILLATION_PROJECTION_TEACHER_CONFIDENCE_HEAD_METHODS = {
    f"{stage}{head}_aligned": f"{training} · {prior}"
    for stage, training in (("raw", "Ordinary training"), ("estimated_training", "Projected training"))
    for head, prior in (("", "Isotropic"), ("_confidence_head", "Teacher confidence"),
                        ("_shuffled_head", "Shuffled confidence"), ("_parity_head", "Parity means"))
}
TRACK_DISTILLATION_PROJECTION_APPROX_TEACHER_TRACK_PRIOR_METHODS = {
    "raw_aligned": "Prediction only · uniform loss",
    "prediction_natural_aligned": "Prediction only · natural loss",
    "prediction_rotated_aligned": "Prediction only · rotated loss",
    "estimated_training_aligned": "Training projection · uniform loss",
    "projected_natural_aligned": "Training projection · natural loss",
    "projected_rotated_aligned": "Training projection · rotated loss",
}
TRACK_DISTILLATION_PROJECTION_TEACHER_RESIDUAL_METHODS = {
    "raw_aligned": "Ordinary training",
    "estimated_training_aligned": "Projected loss",
    "suppress_residual_aligned": "Projected loss · suppress residual",
    "match_residual_aligned": "Projected loss · match teacher residual",
}
TRACK_DISTILLATION_PROJECTION_PROJECTED_TEACHER_PRIOR_METHODS = {
    "estimated_training_aligned": "Uniform loss",
    "raw_teacher_prior_aligned": "Prior from raw teacher features",
    "projected_teacher_prior_aligned": "Prior from projected teacher features",
    "rotated_teacher_prior_aligned": "Rotated prior",
}
TRACK_DISTILLATION_PROJECTION_TRACK_TARGETS_METHODS = {
    "estimated_training_aligned": "Same-frame targets",
    "same_track_targets_aligned": "Same-track targets",
    "other_track_targets_aligned": "Other-track targets within class",
}
TRACK_DISTILLATION_PROJECTION_BLENDED_TARGETS_METHODS = {
    **TRACK_DISTILLATION_PROJECTION_TRACK_TARGETS_METHODS,
    "same_track_targets_blended_aligned": "Blend with same-track targets",
    "other_track_targets_blended_aligned": "Blend with other-track targets",
}
TRACK_DISTILLATION_PROJECTION_CLASS_TARGETS_METHODS = {
    **TRACK_DISTILLATION_PROJECTION_TRACK_TARGETS_METHODS,
    "class_prototypes_aligned": "Class mean targets",
    "scaled_class_prototypes_aligned": "Class mean targets with matched variance",
}
TRACK_DISTILLATION_PROJECTION_CLASS_GEOMETRY_METHODS = {
    **TRACK_DISTILLATION_PROJECTION_CLASS_TARGETS_METHODS,
    "rotated_class_prototypes_aligned": "Rotated class correspondences with matched covariance",
}
TRACK_DISTILLATION_PROJECTION_GAUSSIAN_TARGETS_METHODS = {
    **TRACK_DISTILLATION_PROJECTION_TRACK_TARGETS_METHODS,
    "class_prototypes_aligned": "Class mean targets",
    "gaussian_targets_aligned": "Class Gaussian targets",
    "rotated_gaussian_targets_aligned": "Class Gaussian · rotated covariance",
}
TRACK_DISTILLATION_PROJECTION_PARITY_STUDENT_METHODS = {
    "raw_aligned": "Ordinary training · two views",
    "estimated_training_aligned": "Projected training · two views",
    "folded_parameterization": "Folded GELU · one view",
    "gated_parameterization": "Gated parity · one view",
}
TRACK_DISTILLATION_PROJECTION_PARITY_SUPERVISION_METHODS = {
    "estimated_training_aligned": "Dense · two teacher views",
    "one_view_dense_aligned": "Dense · one teacher view",
    "folded_parameterization": "Folded GELU · two teacher views",
    "one_view_folded": "Folded GELU · one teacher view",
}
TRACK_DISTILLATION_PROJECTION_MIXED_PRIOR_METHODS = {
    "trained_late_natural_head_aligned": "Training projection · teacher prior",
    "trained_late_student_head_aligned": "Training projection · student prior",
    "trained_late_mean_head_aligned": "Training projection · average with fixed spectrum",
    "trained_late_evidence_head_aligned": "Training projection · evidence mixture",
    "late_track_natural_head_aligned": "Prediction-only projection · teacher prior",
    "late_track_evidence_head_aligned": "Prediction-only projection · evidence mixture",
}
TRACK_CONSISTENCY_METHODS = {"raw": "Uncorrected", "same_track": "Full correction",
                             "matched_weight": "Matched mean weight", "shuffled_weight": "Shuffled weights",
                             "consistency_weight": "Track consistency weights"}
TRACK_POOLING_METHODS = {"raw": "Raw feature mean", "feature_corrected": "Corrected feature mean",
                         "pixel_raw": "Raw pixel mean", "pixel_corrected": "Corrected pixel mean",
                         "pixel_shuffled": "Pixel mean · shuffled poses"}


def plot_head_routing(results, output):
    shots = results["config"]["shots"]
    families = ("invariant", "antisymmetric")
    methods = {
        "late_track_natural_head_aligned": "Prediction-only projection · natural prior",
        "late_track_rotated_head_aligned": "Prediction-only projection · rotated prior",
        "trained_late_natural_head_aligned": "Training projection · natural prior",
        "trained_late_rotated_head_aligned": "Training projection · rotated prior",
    }
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for method, label in methods.items():
        route, accuracy = [], []
        for count in map(str, shots):
            episodes = [task[count] for run in results["runs"]
                        for family in families for task in run["tasks"][family].values()]
            route.append(np.mean([episode["routing"][method]["accuracy"] for episode in episodes]))
            accuracy.append(np.mean([prediction[method]["reflected"] for episode in episodes
                                     for prediction in episode["strengths"].values()]))
        axes[0].plot(shots, 100 * np.array(route), marker="o", label=label)
        axes[1].plot(shots, 100 * np.array(accuracy), marker="o", label=label)
    axes[0].set_ylabel("Correct reflection action (%)")
    axes[1].set_ylabel("Reflected balanced accuracy (%)")
    for axis in axes:
        axis.set_xlabel("Labels per class")
        axis.set_xticks(shots)
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=RESULTS / "metrics.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "curves.png")
    args = parser.parse_args()
    results = json.loads(args.metrics.read_text())
    if results["config"].get("track_distillation_projection_head_routing", False):
        plot_head_routing(results, args.output)
        return
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
    if results["config"].get("track_distillation_width", False):
        methods = TRACK_DISTILLATION_WIDTH_METHODS
    elif results["config"].get("track_distillation_synthetic", False):
        methods = TRACK_DISTILLATION_SYNTHETIC_METHODS
    elif results["config"].get("track_distillation_tail", False):
        methods = {"raw": "Uniform throughout"}
        for phase, epochs in results["distillation_tail_epochs"].items():
            methods[f"{phase}_track"] = f"Final {epochs} · natural"
            methods[f"{phase}_rotated"] = f"Final {epochs} · rotated"
    elif results["config"].get("track_distillation_late_head", False):
        methods = TRACK_DISTILLATION_LATE_HEAD_METHODS
    elif results["config"].get("track_distillation_projection_train", False):
        methods = TRACK_DISTILLATION_PROJECTION_TRAIN_METHODS
    elif results["config"].get("track_distillation_projection_shrink", False):
        methods = TRACK_DISTILLATION_PROJECTION_SHRINK_METHODS
    elif results["config"].get("track_distillation_projection_calibrated", False):
        methods = TRACK_DISTILLATION_PROJECTION_CALIBRATED_METHODS
    elif results["config"].get("track_distillation_projection_penalty", False):
        methods = {f"{name}_aligned": f"Complement penalty {penalty:g}"
                   for name, penalty in sorted(results["projection_penalties"].items(), key=lambda item: item[1])}
    elif results["config"].get("track_distillation_projection_schedule", False):
        methods = TRACK_DISTILLATION_PROJECTION_SCHEDULE_METHODS
    elif results["config"].get("track_distillation_projection_compress", False):
        methods = TRACK_DISTILLATION_PROJECTION_COMPRESS_METHODS
    elif results["config"].get("track_distillation_projection_timing_weights", False):
        methods = TRACK_DISTILLATION_PROJECTION_TIMING_WEIGHTS_METHODS
    elif results["config"].get("track_distillation_projection_terminal", False):
        methods = TRACK_DISTILLATION_PROJECTION_TERMINAL_METHODS
    elif results["config"].get("track_distillation_projection_canonical", False):
        methods = TRACK_DISTILLATION_PROJECTION_CANONICAL_METHODS
    elif results["config"].get("track_distillation_projection_mixed_prior", False):
        methods = TRACK_DISTILLATION_PROJECTION_MIXED_PRIOR_METHODS
    elif results["config"].get("track_distillation_projection_student_prior", False):
        methods = TRACK_DISTILLATION_PROJECTION_STUDENT_PRIOR_METHODS
    elif results["config"].get("track_distillation_projection_head_validation", False):
        methods = TRACK_DISTILLATION_PROJECTION_HEAD_VALIDATION_METHODS
    elif results["config"].get("track_distillation_projection_head_policy", False):
        methods = TRACK_DISTILLATION_PROJECTION_HEAD_POLICY_METHODS
    elif results["config"].get("track_distillation_projection_relative_augmentation", False):
        methods = TRACK_DISTILLATION_PROJECTION_RELATIVE_AUGMENTATION_METHODS
    elif results["config"].get("track_distillation_projection_confident_teacher", False):
        methods = TRACK_DISTILLATION_PROJECTION_CONFIDENT_TEACHER_METHODS
    elif results["config"].get("track_distillation_projection_teacher_confidence_head", False):
        methods = TRACK_DISTILLATION_PROJECTION_TEACHER_CONFIDENCE_HEAD_METHODS
    elif results["config"].get("track_distillation_projection_approx_teacher_track_prior", False):
        methods = TRACK_DISTILLATION_PROJECTION_APPROX_TEACHER_TRACK_PRIOR_METHODS
    elif results["config"].get("track_distillation_projection_teacher_residual", False):
        methods = TRACK_DISTILLATION_PROJECTION_TEACHER_RESIDUAL_METHODS
    elif results["config"].get("track_distillation_projection_mirrored_targets", False):
        methods = {f"{name}_aligned": f"Mirrored target weight {weight:g}"
                   for name, weight in sorted(results["mirrored_target_weights"].items(), key=lambda item: item[1])}
    elif results["config"].get("track_distillation_projection_projected_teacher_prior", False):
        methods = TRACK_DISTILLATION_PROJECTION_PROJECTED_TEACHER_PRIOR_METHODS
    elif results["config"].get("track_distillation_projection_blended_targets", False):
        methods = TRACK_DISTILLATION_PROJECTION_BLENDED_TARGETS_METHODS
    elif results["config"].get("track_distillation_projection_class_geometry", False):
        methods = TRACK_DISTILLATION_PROJECTION_CLASS_GEOMETRY_METHODS
    elif results["config"].get("track_distillation_projection_class_targets", False):
        methods = TRACK_DISTILLATION_PROJECTION_CLASS_TARGETS_METHODS
    elif results["config"].get("track_distillation_projection_gaussian_targets", False):
        methods = TRACK_DISTILLATION_PROJECTION_GAUSSIAN_TARGETS_METHODS
    elif results["config"].get("track_distillation_projection_track_targets", False):
        methods = TRACK_DISTILLATION_PROJECTION_TRACK_TARGETS_METHODS
    elif results["config"].get("track_distillation_projection_parity_supervision", False):
        methods = TRACK_DISTILLATION_PROJECTION_PARITY_SUPERVISION_METHODS
    elif results["config"].get("track_distillation_projection_parity_student", False):
        methods = TRACK_DISTILLATION_PROJECTION_PARITY_STUDENT_METHODS
    elif results["config"].get("track_distillation_projection_approx_teacher", False):
        methods = TRACK_DISTILLATION_PROJECTION_APPROX_TEACHER_METHODS
    elif results["config"].get("track_distillation_projection_joint_head", False):
        methods = TRACK_DISTILLATION_PROJECTION_JOINT_HEAD_METHODS
    elif results["config"].get("track_distillation_projection_canonical_train", False):
        methods = TRACK_DISTILLATION_PROJECTION_CANONICAL_TRAIN_METHODS
    elif results["config"].get("track_distillation_projection_single_view_penalty", False):
        methods = TRACK_DISTILLATION_PROJECTION_SINGLE_VIEW_PENALTY_METHODS
    elif results["config"].get("track_distillation_projection", False):
        methods = TRACK_DISTILLATION_PROJECTION_METHODS
    elif results["config"].get("track_distillation_schedule", False):
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
