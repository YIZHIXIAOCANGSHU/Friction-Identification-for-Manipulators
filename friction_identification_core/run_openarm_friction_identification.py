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

from friction_identification_core.estimator import fit_multijoint_friction
from friction_identification_core.mujoco_driver import MujocoFrictionCollectionConfig, MujocoFrictionCollector
from friction_identification_core.rerun_reporter import FrictionRerunReporter


OPENARM_LEFT_JOINTS = [
    "openarm_left_joint1",
    "openarm_left_joint2",
    "openarm_left_joint3",
    "openarm_left_joint4",
    "openarm_left_joint5",
    "openarm_left_joint6",
    "openarm_left_joint7",
]

OPENARM_LEFT_ACTUATORS = [
    "left_joint1_ctrl",
    "left_joint2_ctrl",
    "left_joint3_ctrl",
    "left_joint4_ctrl",
    "left_joint5_ctrl",
    "left_joint6_ctrl",
    "left_joint7_ctrl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reusable MuJoCo friction identification for OpenArm.")
    parser.add_argument("--duration", type=float, default=18.0, help="Excitation duration in seconds.")
    parser.add_argument("--sample-rate", type=float, default=400.0, help="Logging sample rate in Hz.")
    parser.add_argument("--base-frequency", type=float, default=0.12, help="Base excitation frequency in Hz.")
    parser.add_argument("--amplitude-scale", type=float, default=0.22, help="Trajectory amplitude scale.")
    parser.add_argument("--feedback-scale", type=float, default=0.2, help="PD feedback mix used during tracking.")
    parser.add_argument("--render", action="store_true", help="Open the MuJoCo viewer during collection.")
    parser.add_argument("--spawn-rerun", action="store_true", help="Spawn a local Rerun viewer.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = PROJECT_ROOT / "openarm_mujoco" / "scene_with_target_gripper.xml"
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    collector = MujocoFrictionCollector(
        model_path=str(model_path),
        joint_names=OPENARM_LEFT_JOINTS,
        actuator_names=OPENARM_LEFT_ACTUATORS,
        timestep=0.0005,
        render=args.render,
    )

    config = MujocoFrictionCollectionConfig(
        duration=args.duration,
        sample_rate=args.sample_rate,
        timestep=0.0005,
        base_frequency=args.base_frequency,
        amplitude_scale=args.amplitude_scale,
        render=args.render,
        realtime=False,
        feedback_scale=args.feedback_scale,
    )

    reporter = FrictionRerunReporter(
        app_name="OpenArm Friction Identification",
        spawn=args.spawn_rerun,
    )

    try:
        batch = collector.run_openarm_collection(config)
        clean_mask = collector.build_clean_sample_mask(batch, limit_margin=0.05, constraint_tolerance=0.35)
        batch = batch.subset(clean_mask)
        true_coulomb, true_viscous = collector.get_true_friction_parameters()

        validation_mask = np.zeros(batch.time.shape[0], dtype=bool)
        validation_mask[::5] = True
        validation_mask[: min(20, validation_mask.shape[0])] = False

        result = fit_multijoint_friction(
            velocity=batch.qd,
            torque=batch.tau_friction,
            joint_names=OPENARM_LEFT_JOINTS,
            validation_mask=validation_mask,
            velocity_scale=0.03,
            regularization=1e-8,
            max_iterations=16,
            huber_delta=1.35,
            min_velocity_threshold=0.01,
            true_coulomb=true_coulomb,
            true_viscous=true_viscous,
        )

        reporter.init()
        reporter.log(batch, result, output_dir)

        summary = {
            "joint_names": result.joint_names,
            "true_coulomb": true_coulomb.tolist(),
            "estimated_coulomb": [params.coulomb for params in result.parameters],
            "true_viscous": true_viscous.tolist(),
            "estimated_viscous": [params.viscous for params in result.parameters],
            "estimated_offset": [params.offset for params in result.parameters],
            "train_rmse": result.train_rmse.tolist(),
            "validation_rmse": result.validation_rmse.tolist(),
            "train_r2": result.train_r2.tolist(),
            "validation_r2": result.validation_r2.tolist(),
            "mean_validation_rmse": float(np.nanmean(result.validation_rmse)),
            "mean_validation_r2": float(np.nanmean(result.validation_r2)),
            "retained_samples": int(batch.time.shape[0]),
        }

        with open(output_dir / "friction_identification_summary.json", "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

        np.savez(
            output_dir / "friction_identification_data.npz",
            time=batch.time,
            q=batch.q,
            qd=batch.qd,
            tau_passive=batch.tau_passive,
            tau_constraint=batch.tau_constraint,
            tau_friction=batch.tau_friction,
            tau_pred=result.predicted_torque,
            train_mask=result.train_mask,
            validation_mask=result.validation_mask,
        )

        print("Friction identification finished.")
        print(f"Results saved to: {output_dir}")
        print(f"Mean validation RMSE: {np.nanmean(result.validation_rmse):.6f} Nm")
        print(f"Mean validation R2: {np.nanmean(result.validation_r2):.6f}")
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
