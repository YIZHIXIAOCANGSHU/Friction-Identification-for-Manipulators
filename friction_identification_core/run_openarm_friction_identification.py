#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from friction_identification_core.config import DEFAULT_FRICTION_CONFIG, FrictionIdentificationConfig
from friction_identification_core.estimator import fit_multijoint_friction
from friction_identification_core.models import FrictionSampleBatch, TrackingEvaluationResult
from friction_identification_core.mujoco_driver import MujocoFrictionCollectionConfig, MujocoFrictionCollector
from friction_identification_core.rerun_reporter import FrictionRerunReporter


def parse_args(default_config: FrictionIdentificationConfig) -> argparse.Namespace:
    collection = default_config.collection
    parser = argparse.ArgumentParser(description="Run reusable MuJoCo friction identification for AM-D02.")
    parser.add_argument("--duration", type=float, default=collection.duration, help="Excitation duration in seconds.")
    parser.add_argument("--sample-rate", type=float, default=collection.sample_rate, help="Logging sample rate in Hz.")
    parser.add_argument(
        "--base-frequency",
        type=float,
        default=collection.base_frequency,
        help="Base excitation frequency in Hz.",
    )
    parser.add_argument(
        "--amplitude-scale",
        type=float,
        default=collection.amplitude_scale,
        help="Trajectory amplitude scale.",
    )
    parser.add_argument(
        "--feedback-scale",
        type=float,
        default=collection.feedback_scale,
        help="PD feedback mix used during tracking.",
    )
    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=collection.render,
        help="Open the MuJoCo viewer during collection (default: enabled).",
    )
    parser.add_argument(
        "--spawn-rerun",
        action=argparse.BooleanOptionalAction,
        default=collection.spawn_rerun,
        help="Spawn a local Rerun viewer (default: enabled).",
    )
    return parser.parse_args()


def log_info(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def build_tracking_evaluation(
    *,
    label: str,
    batch: FrictionSampleBatch,
    controller_coulomb: np.ndarray,
    controller_viscous: np.ndarray,
) -> TrackingEvaluationResult:
    joint_error = batch.q - batch.q_cmd
    ee_error = batch.ee_pos - batch.ee_pos_cmd
    joint_rmse = np.sqrt(np.mean(joint_error ** 2, axis=0))
    joint_max_abs_error = np.max(np.abs(joint_error), axis=0)
    ee_rmse_xyz = np.sqrt(np.mean(ee_error ** 2, axis=0))
    ee_error_norm = np.linalg.norm(ee_error, axis=1)
    return TrackingEvaluationResult(
        label=label,
        batch=batch,
        controller_coulomb=np.asarray(controller_coulomb, dtype=np.float64).copy(),
        controller_viscous=np.asarray(controller_viscous, dtype=np.float64).copy(),
        joint_rmse=joint_rmse,
        joint_max_abs_error=joint_max_abs_error,
        ee_rmse_xyz=ee_rmse_xyz,
        mean_joint_rmse=float(np.mean(joint_rmse)),
        ee_position_rmse=float(np.sqrt(np.mean(ee_error_norm ** 2))),
        ee_max_error=float(np.max(ee_error_norm)),
    )


def serialize_tracking_evaluation(result: TrackingEvaluationResult) -> dict[str, object]:
    return {
        "label": result.label,
        "controller_coulomb": result.controller_coulomb.tolist(),
        "controller_viscous": result.controller_viscous.tolist(),
        "mean_joint_rmse_rad": result.mean_joint_rmse,
        "joint_rmse_rad": result.joint_rmse.tolist(),
        "joint_max_abs_error_rad": result.joint_max_abs_error.tolist(),
        "ee_rmse_xyz_m": result.ee_rmse_xyz.tolist(),
        "ee_position_rmse_m": result.ee_position_rmse,
        "ee_max_error_m": result.ee_max_error,
    }


def choose_tracking_winner(
    true_result: TrackingEvaluationResult,
    identified_result: TrackingEvaluationResult,
) -> str:
    tolerance = 1e-9
    if identified_result.ee_position_rmse + tolerance < true_result.ee_position_rmse:
        return identified_result.label
    if true_result.ee_position_rmse + tolerance < identified_result.ee_position_rmse:
        return true_result.label

    if identified_result.mean_joint_rmse + tolerance < true_result.mean_joint_rmse:
        return identified_result.label
    if true_result.mean_joint_rmse + tolerance < identified_result.mean_joint_rmse:
        return true_result.label
    return "tie"


def main() -> None:
    base_config = DEFAULT_FRICTION_CONFIG
    args = parse_args(base_config)
    app_config = base_config.with_collection_overrides(
        duration=args.duration,
        sample_rate=args.sample_rate,
        base_frequency=args.base_frequency,
        amplitude_scale=args.amplitude_scale,
        feedback_scale=args.feedback_scale,
        render=args.render,
        spawn_rerun=args.spawn_rerun,
    )
    model_config = app_config.model
    collection_config = app_config.collection
    sample_filter_config = app_config.sample_filter
    fit_config = app_config.fit

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_info(
        "Friction identification config: "
        f"duration={collection_config.duration:.1f}s, sample_rate={collection_config.sample_rate:.1f}Hz, "
        f"base_frequency={collection_config.base_frequency:.3f}Hz, "
        f"amplitude_scale={collection_config.amplitude_scale:.3f}, "
        f"feedback_scale={collection_config.feedback_scale:.3f}, "
        f"render={collection_config.render}, rerun={collection_config.spawn_rerun}"
    )

    collector = MujocoFrictionCollector(
        model_path=str(model_config.urdf_path),
        joint_names=list(model_config.joint_names),
        actuator_names=None,
        timestep=collection_config.timestep,
        render=collection_config.render,
        home_qpos=model_config.home_qpos,
        end_effector_body=model_config.end_effector_body,
        tcp_offset=model_config.tcp_offset,
        joint_limit_overrides=model_config.joint_limits,
        friction_loss=model_config.friction_loss,
        damping=model_config.damping,
        inverse_friction_loss=np.zeros_like(model_config.friction_loss),
        inverse_damping=np.zeros_like(model_config.damping),
    )

    config = MujocoFrictionCollectionConfig(
        duration=collection_config.duration,
        sample_rate=collection_config.sample_rate,
        timestep=collection_config.timestep,
        base_frequency=collection_config.base_frequency,
        amplitude_scale=collection_config.amplitude_scale,
        render=collection_config.render,
        realtime=collection_config.realtime,
        feedback_scale=collection_config.feedback_scale,
    )

    reporter = FrictionRerunReporter(
        app_name=app_config.app_name,
        spawn=collection_config.spawn_rerun,
    )

    try:
        raw_batch = collector.run_openarm_collection(config)
        log_info(f"Raw collection completed with {raw_batch.time.shape[0]} samples")

        log_info("Filtering clean samples away from joint limits and constraint spikes")
        clean_mask = collector.build_clean_sample_mask(
            raw_batch,
            limit_margin=sample_filter_config.limit_margin,
            constraint_tolerance=sample_filter_config.constraint_tolerance,
        )
        batch = raw_batch.subset(clean_mask)
        retained_samples = int(batch.time.shape[0])
        raw_samples = int(raw_batch.time.shape[0])
        retained_ratio = 100.0 * retained_samples / max(raw_samples, 1)
        log_info(
            f"Sample filtering retained {retained_samples}/{raw_samples} samples "
            f"({retained_ratio:.1f}%)"
        )
        true_coulomb, true_viscous = collector.get_true_friction_parameters()

        validation_mask = sample_filter_config.build_validation_mask(batch.time.shape[0])
        validation_count = int(np.count_nonzero(validation_mask))
        train_count = int(validation_mask.shape[0] - validation_count)
        log_info(
            f"Starting robust friction fitting for {len(model_config.joint_names)} joints "
            f"(train={train_count}, validation={validation_count})"
        )

        result = fit_multijoint_friction(
            velocity=batch.qd,
            torque=batch.tau_friction,
            joint_names=model_config.joint_names,
            validation_mask=validation_mask,
            velocity_scale=fit_config.velocity_scale,
            regularization=fit_config.regularization,
            max_iterations=fit_config.max_iterations,
            huber_delta=fit_config.huber_delta,
            min_velocity_threshold=fit_config.min_velocity_threshold,
            true_coulomb=true_coulomb,
            true_viscous=true_viscous,
            progress_callback=lambda current, total, joint_name: log_info(
                f"Fitting joint {current}/{total}: {joint_name}"
            ),
        )
        log_info("Friction parameter fitting complete")

        estimated_coulomb = np.array([params.coulomb for params in result.parameters], dtype=np.float64)
        estimated_viscous = np.array([params.viscous for params in result.parameters], dtype=np.float64)

        zero_coulomb = np.zeros_like(estimated_coulomb)
        zero_viscous = np.zeros_like(estimated_viscous)
        tracking_zero = build_tracking_evaluation(
            label="zero_friction_controller",
            batch=raw_batch,
            controller_coulomb=zero_coulomb,
            controller_viscous=zero_viscous,
        )

        log_info("Running same-curve tracking comparison after writing identified friction back to controller")
        collector.set_inverse_friction_parameters(
            friction_loss=estimated_coulomb,
            damping=estimated_viscous,
        )
        tracking_identified_batch = collector.run_reference_trajectory(
            q_cmd=raw_batch.q_cmd,
            qd_cmd=raw_batch.qd_cmd,
            qdd_cmd=raw_batch.qdd_cmd,
            sample_rate=collection_config.sample_rate,
            feedback_scale=collection_config.feedback_scale,
            realtime=collection_config.render,
            time_reference=raw_batch.time,
            ee_pos_cmd=raw_batch.ee_pos_cmd,
            ee_quat_cmd=raw_batch.ee_quat_cmd,
        )
        tracking_identified = build_tracking_evaluation(
            label="identified_parameters",
            batch=tracking_identified_batch,
            controller_coulomb=estimated_coulomb,
            controller_viscous=estimated_viscous,
        )

        tracking_winner = choose_tracking_winner(tracking_zero, tracking_identified)
        log_info(
            "Before/after tracking comparison finished: "
            f"zero_ee_rmse={tracking_zero.ee_position_rmse:.6f} m, "
            f"identified_ee_rmse={tracking_identified.ee_position_rmse:.6f} m, "
            f"winner={tracking_winner}"
        )

        log_info("Sending plots and summary to Rerun")
        reporter.init()
        reporter.log(
            raw_batch=raw_batch,
            fit_batch=batch,
            result=result,
            tracking_results=[tracking_zero, tracking_identified],
            output_dir=output_dir,
        )

        summary = {
            "joint_names": result.joint_names,
            "joint_limits_rad": model_config.joint_limits.tolist(),
            "true_coulomb": true_coulomb.tolist(),
            "estimated_coulomb": estimated_coulomb.tolist(),
            "true_viscous": true_viscous.tolist(),
            "estimated_viscous": estimated_viscous.tolist(),
            "estimated_offset": [params.offset for params in result.parameters],
            "train_rmse": result.train_rmse.tolist(),
            "validation_rmse": result.validation_rmse.tolist(),
            "train_r2": result.train_r2.tolist(),
            "validation_r2": result.validation_r2.tolist(),
            "mean_validation_rmse": float(np.nanmean(result.validation_rmse)),
            "mean_validation_r2": float(np.nanmean(result.validation_r2)),
            "raw_samples": int(raw_batch.time.shape[0]),
            "retained_samples": int(batch.time.shape[0]),
            "tracking_comparison": {
                "reference_curve": "same_excitation_curve_used_for_identification",
                "primary_metric": "ee_position_rmse_m",
                "winner": tracking_winner,
                "zero_friction_controller": serialize_tracking_evaluation(tracking_zero),
                "identified_parameters": serialize_tracking_evaluation(tracking_identified),
            },
        }

        log_info("Saving summary JSON and NumPy result bundle")
        with open(output_dir / "friction_identification_summary.json", "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

        np.savez(
            output_dir / "friction_identification_data.npz",
            time_raw=raw_batch.time,
            q_raw=raw_batch.q,
            qd_raw=raw_batch.qd,
            q_cmd_raw=raw_batch.q_cmd,
            ee_pos_raw=raw_batch.ee_pos,
            ee_pos_cmd_raw=raw_batch.ee_pos_cmd,
            ee_quat_raw=raw_batch.ee_quat,
            ee_quat_cmd_raw=raw_batch.ee_quat_cmd,
            tau_passive_raw=raw_batch.tau_passive,
            tau_constraint_raw=raw_batch.tau_constraint,
            tau_friction_raw=raw_batch.tau_friction,
            clean_mask=clean_mask,
            time=batch.time,
            q=batch.q,
            qd=batch.qd,
            ee_pos=batch.ee_pos,
            ee_pos_cmd=batch.ee_pos_cmd,
            tau_passive=batch.tau_passive,
            tau_constraint=batch.tau_constraint,
            tau_friction=batch.tau_friction,
            tau_pred=result.predicted_torque,
            train_mask=result.train_mask,
            validation_mask=result.validation_mask,
            tracking_time=tracking_zero.batch.time,
            tracking_q_cmd=tracking_zero.batch.q_cmd,
            tracking_q_zero=tracking_zero.batch.q,
            tracking_q_identified=tracking_identified.batch.q,
            tracking_qd_zero=tracking_zero.batch.qd,
            tracking_qd_identified=tracking_identified.batch.qd,
            tracking_ee_pos_cmd=tracking_zero.batch.ee_pos_cmd,
            tracking_ee_pos_zero=tracking_zero.batch.ee_pos,
            tracking_ee_pos_identified=tracking_identified.batch.ee_pos,
            tracking_tau_ctrl_zero=tracking_zero.batch.tau_ctrl,
            tracking_tau_ctrl_identified=tracking_identified.batch.tau_ctrl,
            tracking_joint_rmse_zero=tracking_zero.joint_rmse,
            tracking_joint_rmse_identified=tracking_identified.joint_rmse,
            tracking_ee_rmse_xyz_zero=tracking_zero.ee_rmse_xyz,
            tracking_ee_rmse_xyz_identified=tracking_identified.ee_rmse_xyz,
            tracking_controller_coulomb_zero=tracking_zero.controller_coulomb,
            tracking_controller_coulomb_identified=tracking_identified.controller_coulomb,
            tracking_controller_viscous_zero=tracking_zero.controller_viscous,
            tracking_controller_viscous_identified=tracking_identified.controller_viscous,
        )

        log_info("Result files written successfully")
        print("Friction identification finished.")
        print(f"Results saved to: {output_dir}")
        print(f"Mean validation RMSE: {np.nanmean(result.validation_rmse):.6f} Nm")
        print(f"Mean validation R2: {np.nanmean(result.validation_r2):.6f}")
        print(
            "Before/after tracking EE RMSE: "
            f"zero={tracking_zero.ee_position_rmse:.6f} m, "
            f"identified={tracking_identified.ee_position_rmse:.6f} m, "
            f"winner={tracking_winner}"
        )
        print(
            "Before/after mean joint RMSE: "
            f"zero={tracking_zero.mean_joint_rmse:.6f} rad, "
            f"identified={tracking_identified.mean_joint_rmse:.6f} rad"
        )
        for joint_name, params, fc_true, fv_true, rmse in zip(
            result.joint_names,
            result.parameters,
            true_coulomb,
            true_viscous,
            result.validation_rmse,
        ):
            print(
                f"{joint_name}: "
                f"fc={params.coulomb:.4f} (true {fc_true:.4f}), "
                f"fv={params.viscous:.4f} (true {fv_true:.4f}), "
                f"val_rmse={rmse:.6f}"
            )
    finally:
        reporter.close()
        collector.close()


if __name__ == "__main__":
    main()
