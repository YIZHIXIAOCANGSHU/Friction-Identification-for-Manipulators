from __future__ import annotations

from typing import Callable

import numpy as np

from friction_identification_core.config import Config
from friction_identification_core.controller import (
    FrictionIdentificationController,
    SafetyGuard,
    has_compensation_results,
    load_compensation_parameters,
    predict_compensation_torque,
)
from friction_identification_core.models import CollectedData, FrictionIdentificationResult, IdentificationInputs
from friction_identification_core.mujoco_env import MujocoEnvironment
from friction_identification_core.trajectory import ReferenceTrajectory, build_startup_pose
from friction_identification_core.runtime import log_info
from friction_identification_core.visualization import build_simulation_reporter


TorqueCommandBuilder = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]


class SimulationSource:
    source_name = "simulation"

    def __init__(self, config: Config) -> None:
        self.config = config
        self.env = MujocoEnvironment(config)
        self.reporter = build_simulation_reporter(config)
        self.inverse_dynamics_backend = self.env

    def build_reference(self) -> ReferenceTrajectory:
        return self.env.build_excitation_reference()

    def supports_identification(self, mode: str) -> bool:
        return mode in {"collect", "full_feedforward"}

    def collect(
        self,
        *,
        mode: str,
        reference: ReferenceTrajectory | None,
        controller: FrictionIdentificationController,
        safety: SafetyGuard,
    ) -> CollectedData:
        if mode not in {"collect", "full_feedforward"}:
            raise ValueError("mode must be 'collect' or 'full_feedforward'.")
        if reference is None:
            reference = self.build_reference()

        torque_callback = self._build_torque_callback(controller, safety, mode=mode)
        startup_target = build_startup_pose(self.config, reference)
        startup_reference = self.env.build_startup_reference(self.config.robot.home_qpos, startup_target)
        log_info(
            "开始仿真辨识: "
            f"mode={mode}, "
            f"target=J{self.config.target_joint + 1}({self.config.target_joint_name}), "
            f"duration={self.config.identification.excitation.duration:.1f}s, "
            f"sample_rate={self.config.sampling.rate:.1f}Hz"
        )

        raw_batch = self.env.run_reference_trajectory(
            reference,
            controller,
            safety,
            startup_reference=startup_reference,
            torque_callback=torque_callback,
            realtime=self.config.visualization.render,
        )
        clean_mask = self.env.build_clean_sample_mask(raw_batch)
        clean_batch = raw_batch.subset(clean_mask)
        log_info(
            f"仿真采样完成: raw={raw_batch.time.shape[0]}, clean={clean_batch.time.shape[0]}"
        )
        if clean_batch.time.shape[0] < 16:
            raise RuntimeError("筛样后有效样本过少，无法继续拟合。")

        true_coulomb, true_viscous = self.env.get_true_friction_parameters()
        return CollectedData(
            source=self.source_name,
            mode=mode,
            time=raw_batch.time.copy(),
            q=raw_batch.q.copy(),
            qd=raw_batch.qd.copy(),
            q_cmd=raw_batch.q_cmd.copy(),
            qd_cmd=raw_batch.qd_cmd.copy(),
            tau_command=raw_batch.tau_ctrl.copy(),
            tau_measured=raw_batch.tau_friction.copy(),
            qdd=raw_batch.qdd.copy(),
            qdd_cmd=raw_batch.qdd_cmd.copy(),
            tau_passive=raw_batch.tau_passive.copy(),
            tau_constraint=raw_batch.tau_constraint.copy(),
            tau_friction=raw_batch.tau_friction.copy(),
            clean_mask=clean_mask.copy(),
            ee_pos=raw_batch.ee_pos.copy(),
            ee_quat=raw_batch.ee_quat.copy(),
            ee_pos_cmd=raw_batch.ee_pos_cmd.copy(),
            ee_quat_cmd=raw_batch.ee_quat_cmd.copy(),
            metadata={
                "raw_batch": raw_batch,
                "clean_batch": clean_batch,
                "fit_joint_indices": [self.config.target_joint],
                "true_coulomb": true_coulomb.copy(),
                "true_viscous": true_viscous.copy(),
            },
        )

    def prepare_identification(self, data: CollectedData) -> IdentificationInputs:
        if data.clean_mask is None or data.tau_friction is None:
            raise ValueError("simulation data is missing clean-mask or friction torque.")

        target_mask = self.config.target_joint_mask
        clean_mask = np.asarray(data.clean_mask, dtype=bool)
        return IdentificationInputs(
            velocity=data.qd[clean_mask][:, target_mask],
            torque=data.tau_friction[clean_mask][:, target_mask],
            joint_names=[self.config.target_joint_name],
            clean_mask=clean_mask,
            true_coulomb=np.asarray(data.metadata["true_coulomb"], dtype=np.float64)[target_mask],
            true_viscous=np.asarray(data.metadata["true_viscous"], dtype=np.float64)[target_mask],
        )

    def finalize(
        self,
        data: CollectedData | None,
        result: FrictionIdentificationResult | None,
    ) -> None:
        try:
            if self.reporter is not None and data is not None and result is not None:
                self.reporter.log(
                    raw_batch=data.metadata["raw_batch"],
                    fit_batch=data.metadata["clean_batch"],
                    result=result,
                    tracking_results=[],
                    output_dir=self.config.results_dir,
                    fit_joint_indices=data.metadata.get("fit_joint_indices", [self.config.target_joint]),
                )
        finally:
            if self.reporter is not None:
                self.reporter.close()
                self.reporter = None
            self.env.close()

    def _build_torque_callback(
        self,
        controller: FrictionIdentificationController,
        safety: SafetyGuard,
        *,
        mode: str,
    ) -> TorqueCommandBuilder:
        if mode == "collect":
            return controller.compute_torque
        if mode != "full_feedforward":
            raise ValueError("mode must be 'collect' or 'full_feedforward'.")

        parameters = load_compensation_parameters(self.config.summary_path, self.config.joint_count)
        if not has_compensation_results(self.config.summary_path):
            log_info(
                "未找到辨识汇总文件，仿真 full_feedforward 将仅使用刚体前馈 + 反馈；"
                f"如需摩擦补偿，请先生成 {self.config.summary_path.name}。"
            )

        def _torque_callback(
            q_cmd: np.ndarray,
            qd_cmd: np.ndarray,
            qdd_cmd: np.ndarray,
            q_curr: np.ndarray,
            qd_curr: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            tau_ff, tau_fb, _ = controller.compute_torque(
                q_cmd=q_cmd,
                qd_cmd=qd_cmd,
                qdd_cmd=qdd_cmd,
                q_curr=q_curr,
                qd_curr=qd_curr,
            )
            tau_friction = predict_compensation_torque(
                qd_cmd,
                parameters,
                torque_limits=self.config.robot.torque_limits,
            )
            tau_ff_total = tau_ff + tau_friction
            tau_command = safety.soften_torque_near_joint_limits(
                q_curr,
                safety.clamp_torque(tau_ff_total + controller.feedback_scale * tau_fb),
            )
            return tau_ff_total, tau_fb, tau_command

        return _torque_callback
