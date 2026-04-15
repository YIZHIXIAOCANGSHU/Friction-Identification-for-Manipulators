from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from friction_identification_core.config import Config
from friction_identification_core.core.controller import FrictionIdentificationController
from friction_identification_core.core.estimator import fit_multijoint_friction
from friction_identification_core.core.models import FrictionIdentificationResult, FrictionSampleBatch
from friction_identification_core.core.safety import SafetyGuard
from friction_identification_core.core.trajectory import build_startup_pose
from friction_identification_core.simulation.mujoco_env import MujocoEnvironment
from friction_identification_core.utils.logging import ensure_directory, log_info, write_json
from friction_identification_core.utils.visualization import build_simulation_reporter


@dataclass(frozen=True)
class SimulationRunResult:
    raw_batch: FrictionSampleBatch
    clean_batch: FrictionSampleBatch
    fit_result: FrictionIdentificationResult
    npz_path: Path
    json_path: Path


def _build_validation_mask(num_samples: int) -> np.ndarray:
    mask = np.zeros(num_samples, dtype=bool)
    if num_samples > 0:
        mask[::5] = True
        mask[: min(20, num_samples)] = False
        if np.all(mask):
            mask[-1] = False
    return mask


def run_simulation(config: Config) -> SimulationRunResult:
    results_dir = ensure_directory(config.results_dir)
    prefix = f"{config.output.simulation_prefix}_joint_{config.target_joint + 1}"
    npz_path = results_dir / f"{prefix}.npz"
    json_path = results_dir / f"{prefix}.json"

    env = MujocoEnvironment(config)
    safety = SafetyGuard(config, active_joint_mask=config.target_joint_mask)
    controller = FrictionIdentificationController(config, env, safety=safety)
    reporter = build_simulation_reporter(config)

    try:
        reference = env.build_excitation_reference()
        startup_target = build_startup_pose(config, reference)
        startup_reference = env.build_startup_reference(config.robot.home_qpos, startup_target)
        log_info(
            "开始仿真辨识: "
            f"target=J{config.target_joint + 1}({config.target_joint_name}), "
            f"duration={config.identification.excitation.duration:.1f}s, "
            f"sample_rate={config.sampling.rate:.1f}Hz"
        )
        raw_batch = env.run_reference_trajectory(
            reference,
            controller,
            safety,
            startup_reference=startup_reference,
            realtime=config.visualization.render,
        )
        clean_mask = env.build_clean_sample_mask(raw_batch)
        clean_batch = raw_batch.subset(clean_mask)
        log_info(
            f"仿真采样完成: raw={raw_batch.time.shape[0]}, clean={clean_batch.time.shape[0]}"
        )
        if clean_batch.time.shape[0] < 16:
            raise RuntimeError("筛样后有效样本过少，无法继续拟合。")

        target_mask = config.target_joint_mask
        true_coulomb, true_viscous = env.get_true_friction_parameters()
        fit_result = fit_multijoint_friction(
            velocity=clean_batch.qd[:, target_mask],
            torque=clean_batch.tau_friction[:, target_mask],
            joint_names=[config.target_joint_name],
            validation_mask=_build_validation_mask(clean_batch.time.shape[0]),
            velocity_scale=config.fitting.velocity_scale,
            regularization=config.fitting.regularization,
            max_iterations=config.fitting.max_iterations,
            huber_delta=config.fitting.huber_delta,
            min_velocity_threshold=config.fitting.min_velocity_threshold,
            true_coulomb=true_coulomb[target_mask],
            true_viscous=true_viscous[target_mask],
        )

        estimated_coulomb = np.zeros(config.joint_count, dtype=np.float64)
        estimated_viscous = np.zeros(config.joint_count, dtype=np.float64)
        estimated_offset = np.zeros(config.joint_count, dtype=np.float64)
        estimated_coulomb[target_mask] = [fit_result.parameters[0].coulomb]
        estimated_viscous[target_mask] = [fit_result.parameters[0].viscous]
        estimated_offset[target_mask] = [fit_result.parameters[0].offset]
        predicted_full = np.zeros_like(clean_batch.tau_friction)
        predicted_full[:, target_mask] = fit_result.predicted_torque

        np.savez(
            npz_path,
            time=raw_batch.time,
            q=raw_batch.q,
            qd=raw_batch.qd,
            qdd=raw_batch.qdd,
            q_cmd=raw_batch.q_cmd,
            qd_cmd=raw_batch.qd_cmd,
            qdd_cmd=raw_batch.qdd_cmd,
            tau_ctrl=raw_batch.tau_ctrl,
            tau_passive=raw_batch.tau_passive,
            tau_constraint=raw_batch.tau_constraint,
            tau_friction=raw_batch.tau_friction,
            clean_mask=clean_mask,
            clean_time=clean_batch.time,
            clean_q=clean_batch.q,
            clean_qd=clean_batch.qd,
            clean_tau_friction=clean_batch.tau_friction,
            tau_pred=fit_result.predicted_torque,
            tau_pred_full=predicted_full,
        )

        summary = {
            "mode": "simulation",
            "config_path": str(config.config_path),
            "target_joint": int(config.target_joint),
            "target_joint_name": config.target_joint_name,
            "sample_count": int(raw_batch.time.shape[0]),
            "clean_sample_count": int(clean_batch.time.shape[0]),
            "estimated_coulomb": estimated_coulomb.tolist(),
            "estimated_viscous": estimated_viscous.tolist(),
            "estimated_offset": estimated_offset.tolist(),
            "true_coulomb": true_coulomb.tolist(),
            "true_viscous": true_viscous.tolist(),
            "validation_rmse": fit_result.validation_rmse.tolist(),
            "validation_r2": fit_result.validation_r2.tolist(),
            "velocity_scale": float(config.fitting.velocity_scale),
        }
        write_json(json_path, summary)

        if reporter is not None:
            reporter.log(
                raw_batch=raw_batch,
                fit_batch=clean_batch,
                result=fit_result,
                tracking_results=[],
                output_dir=results_dir,
                fit_joint_indices=[config.target_joint],
            )
            reporter.close()

        log_info(f"仿真结果已保存: {npz_path}")
        return SimulationRunResult(
            raw_batch=raw_batch,
            clean_batch=clean_batch,
            fit_result=fit_result,
            npz_path=npz_path,
            json_path=json_path,
        )
    finally:
        env.close()
