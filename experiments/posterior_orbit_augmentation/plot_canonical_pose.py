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
    if track_pooling:
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
