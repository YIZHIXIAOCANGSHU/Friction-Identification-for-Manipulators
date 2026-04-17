from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np

from friction_identification_core.config import Config
from friction_identification_core.models import (
    FrictionIdentificationResult,
    FrictionSampleBatch,
    TrackingEvaluationResult,
)
from friction_identification_core.mujoco_support import build_am_d02_model
from friction_identification_core.runtime import log_info


class SimulationRerunReporter:
    """Offline Rerun reporting for simulated friction identification experiments."""

    def __init__(self, *, app_name: str, spawn: bool = True):
        self.app_name = app_name
        self.spawn = spawn
        self._rr = None

    def init(self) -> None:
        import rerun as rr
        import rerun.blueprint as rrb

        self._rr = rr
        rr.init(self.app_name, spawn=self.spawn)

        for axis_name, color in zip(("x", "y", "z"), ([230, 90, 70], [80, 190, 90], [70, 120, 230])):
            rr.log(
                f"ee_tracking/{axis_name}",
                rr.SeriesLines(
                    colors=[color, [120, 120, 120]],
                    names=["actual", "expected"],
                    widths=[2.0, 1.5],
                ),
                static=True,
            )

        blueprint = rrb.Blueprint(
            rrb.Tabs(
                rrb.Vertical(
                    rrb.TextDocumentView(name="Summary", origin="/summary/report"),
                    rrb.BarChartView(name="Validation RMSE", origin="/summary/validation_rmse"),
                    name="Summary",
                ),
                rrb.Grid(
                    *[
                        rrb.TimeSeriesView(name=f"Joint {joint_idx + 1}", origin=f"/friction/joint_{joint_idx + 1}")
                        for joint_idx in range(7)
                    ],
                    name="Per Joint",
                ),
                rrb.Vertical(
                    rrb.Spatial3DView(name="EE Trajectory", origin="/trajectory_3d"),
                    rrb.Horizontal(
                        rrb.TimeSeriesView(name="EE X", origin="/ee_tracking/x"),
                        rrb.TimeSeriesView(name="EE Y", origin="/ee_tracking/y"),
                        rrb.TimeSeriesView(name="EE Z", origin="/ee_tracking/z"),
                    ),
                    name="EE Path",
                ),
            )
        )
        rr.send_blueprint(blueprint)

    def log(
        self,
        *,
        raw_batch: FrictionSampleBatch,
        fit_batch: FrictionSampleBatch,
        result: FrictionIdentificationResult,
        tracking_results: Sequence[TrackingEvaluationResult],
        output_dir: Path,
        fit_joint_indices: Sequence[int] | None = None,
    ) -> None:
        if self._rr is None:
            return
        rr = self._rr

        if fit_joint_indices is None:
            fit_joint_indices = list(range(len(result.joint_names)))
        else:
            fit_joint_indices = list(fit_joint_indices)
            if len(fit_joint_indices) != len(result.joint_names):
                raise ValueError("fit_joint_indices must align with result.joint_names.")

        rr.log(
            "trajectory_3d/expected_path",
            rr.LineStrips3D([raw_batch.ee_pos_cmd], colors=[[80, 220, 120]], radii=[0.002]),
            static=True,
        )
        rr.log(
            "trajectory_3d/actual_path",
            rr.LineStrips3D([raw_batch.ee_pos], colors=[[230, 110, 70]], radii=[0.0015]),
            static=True,
        )

        for sample_idx, time_s in enumerate(raw_batch.time):
            rr.set_time_seconds("sim_time", float(time_s))
            rr.log("ee_tracking/x/actual", rr.Scalars(raw_batch.ee_pos[sample_idx, 0]))
            rr.log("ee_tracking/x/expected", rr.Scalars(raw_batch.ee_pos_cmd[sample_idx, 0]))
            rr.log("ee_tracking/y/actual", rr.Scalars(raw_batch.ee_pos[sample_idx, 1]))
            rr.log("ee_tracking/y/expected", rr.Scalars(raw_batch.ee_pos_cmd[sample_idx, 1]))
            rr.log("ee_tracking/z/actual", rr.Scalars(raw_batch.ee_pos[sample_idx, 2]))
            rr.log("ee_tracking/z/expected", rr.Scalars(raw_batch.ee_pos_cmd[sample_idx, 2]))

        for sample_idx, time_s in enumerate(fit_batch.time):
            rr.set_time_seconds("sim_time", float(time_s))
            for result_joint_idx, joint_idx in enumerate(fit_joint_indices):
                rr.log(
                    f"friction/joint_{joint_idx + 1}/measured",
                    rr.Scalars(fit_batch.tau_friction[sample_idx, joint_idx]),
                )
                rr.log(
                    f"friction/joint_{joint_idx + 1}/predicted",
                    rr.Scalars(result.predicted_torque[sample_idx, result_joint_idx]),
                )

        rr.log("summary/validation_rmse", rr.BarChart(result.validation_rmse))

        lines = [
            "# Friction Identification Summary",
            "",
            f"- Raw samples: {raw_batch.time.shape[0]}",
            f"- Fit samples: {fit_batch.time.shape[0]}",
            f"- Output: `{output_dir}`",
            "",
            "| Joint | fc_true | fc_est | fv_true | fv_est | Val RMSE | Val R2 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for joint_idx, params in enumerate(result.parameters):
            fc_true = float(result.true_coulomb[joint_idx]) if result.true_coulomb is not None else float("nan")
            fv_true = float(result.true_viscous[joint_idx]) if result.true_viscous is not None else float("nan")
            lines.append(
                "| "
                f"{result.joint_names[joint_idx]} | "
                f"{fc_true:.4f} | {params.coulomb:.4f} | "
                f"{fv_true:.4f} | {params.viscous:.4f} | "
                f"{result.validation_rmse[joint_idx]:.6f} | {result.validation_r2[joint_idx]:.4f} |"
            )
        rr.log("summary/report", rr.TextDocument("\n".join(lines), media_type="text/markdown"))

    def close(self) -> None:
        if self._rr is not None:
            self._rr.disconnect()
            self._rr = None


class HardwareRerunReporter:
    """Realtime Rerun dashboards for live UART collection and compensation runs."""

    def __init__(self, *, app_name: str, joint_names: Sequence[str], spawn: bool = True) -> None:
        self.app_name = app_name
        self.spawn = spawn
        self.joint_names = list(joint_names)
        self._rr = None

    def init(self) -> None:
        import rerun as rr
        import rerun.blueprint as rrb

        self._rr = rr
        rr.init(self.app_name, spawn=self.spawn)

        joint_colors = [
            [230, 50, 50],
            [230, 140, 30],
            [210, 200, 30],
            [50, 200, 50],
            [50, 200, 200],
            [50, 80, 230],
            [150, 50, 230],
        ]

        for idx, color in enumerate(joint_colors, start=1):
            rr.log(f"joint_state/q/J{idx}", rr.SeriesLines(colors=[color], names=[f"J{idx} q"], widths=[2.0]), static=True)
            rr.log(f"joint_state/qd/J{idx}", rr.SeriesLines(colors=[color], names=[f"J{idx} qd"], widths=[2.0]), static=True)
            rr.log(f"torque/measured/J{idx}", rr.SeriesLines(colors=[color], names=[f"J{idx} measured"], widths=[2.0]), static=True)
            rr.log(f"torque/command/J{idx}", rr.SeriesLines(colors=[color], names=[f"J{idx} command"], widths=[2.0]), static=True)

        rr.log("performance/uart_cycle_hz", rr.SeriesLines(colors=[[230, 120, 40]], names=["UART cycle rate"], widths=[2.0]), static=True)
        rr.log("performance/uart_latency_ms", rr.SeriesLines(colors=[[230, 170, 80]], names=["UART cycle period"], widths=[2.0]), static=True)
        rr.log("performance/uart_transfer_kbps", rr.SeriesLines(colors=[[100, 180, 230]], names=["UART throughput"], widths=[2.0]), static=True)

        blueprint = rrb.Blueprint(
            rrb.Tabs(
                rrb.Grid(
                    *[
                        rrb.TimeSeriesView(name=f"J{idx + 1} Position", origin=f"/joint_state/q/J{idx + 1}")
                        for idx in range(len(self.joint_names))
                    ],
                    name="Joint Position",
                ),
                rrb.Grid(
                    *[
                        rrb.TimeSeriesView(name=f"J{idx + 1} Command", origin=f"/torque/command/J{idx + 1}")
                        for idx in range(len(self.joint_names))
                    ],
                    name="Command Torque",
                ),
                rrb.Vertical(
                    rrb.TimeSeriesView(name="UART Frequency", origin="/performance/uart_cycle_hz"),
                    rrb.TimeSeriesView(name="UART Period", origin="/performance/uart_latency_ms"),
                    rrb.TimeSeriesView(name="UART Throughput", origin="/performance/uart_transfer_kbps"),
                    rrb.Spatial3DView(name="EE Pose", origin="/trajectory_3d"),
                    name="Runtime",
                ),
            ),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint)

    def log_step(
        self,
        *,
        elapsed_s: float,
        step_index: int,
        q: np.ndarray,
        qd: np.ndarray,
        tau_measured: np.ndarray,
        tau_command: np.ndarray,
        mos_temperature: np.ndarray,
        coil_temperature: np.ndarray,
        uart_cycle_hz: float,
        uart_latency_ms: float,
        uart_transfer_kbps: float,
        ee_pos: np.ndarray | None,
        ee_quat: np.ndarray | None,
        rx_text: str | None,
        tx_text: str | None,
    ) -> None:
        if self._rr is None:
            return
        rr = self._rr
        rr.set_time_seconds("time", float(elapsed_s))
        rr.set_time_sequence("step", int(step_index))

        for idx in range(q.shape[0]):
            rr.log(f"joint_state/q/J{idx + 1}", rr.Scalars(float(q[idx])))
            rr.log(f"joint_state/qd/J{idx + 1}", rr.Scalars(float(qd[idx])))
            rr.log(f"torque/measured/J{idx + 1}", rr.Scalars(float(tau_measured[idx])))
            rr.log(f"torque/command/J{idx + 1}", rr.Scalars(float(tau_command[idx])))

        rr.log("performance/uart_cycle_hz", rr.Scalars(float(uart_cycle_hz)))
        rr.log("performance/uart_latency_ms", rr.Scalars(float(uart_latency_ms)))
        rr.log("performance/uart_transfer_kbps", rr.Scalars(float(uart_transfer_kbps)))

        if ee_pos is not None:
            rr.log(
                "trajectory_3d/ee",
                rr.Points3D(
                    [np.asarray(ee_pos, dtype=np.float64)],
                    colors=[[230, 100, 50]],
                    radii=[0.012],
                    labels=["ee"],
                ),
            )

    def close(self) -> None:
        if self._rr is not None:
            self._rr.disconnect()
            self._rr = None


class PoseEstimator:
    """Forward-kinematics helper for visualizing real robot joint states in MuJoCo."""

    def __init__(
        self,
        *,
        model_path: str,
        joint_names: list[str],
        end_effector_body: str,
        tcp_offset: np.ndarray,
        render: bool = True,
        viewer_fps: float = 30.0,
    ) -> None:
        import mujoco
        import mujoco.viewer

        self._mujoco = mujoco
        self._viewer_module = mujoco.viewer
        self.model = build_am_d02_model(model_path, np.asarray(tcp_offset, dtype=np.float64))
        self.data = mujoco.MjData(self.model)
        self.qpos_addrs = []
        self.viewer = None
        self._viewer_period_s = 1.0 / max(float(viewer_fps), 1.0)
        self._last_viewer_sync = 0.0

        for name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"找不到关节: {name}")
            self.qpos_addrs.append(self.model.jnt_qposadr[joint_id])

        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, end_effector_body)
        if self.ee_body_id < 0:
            raise ValueError(f"找不到末端 body: {end_effector_body}")

        if render:
            self.viewer = self._viewer_module.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
            )
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -20
            self.viewer.cam.distance = 1.8
            self.viewer.cam.lookat[:] = [0.0, 0.0, 1.1]

    def update(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.size != len(self.qpos_addrs):
            raise ValueError("q size does not match the configured robot joints.")

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        for addr, value in zip(self.qpos_addrs, q):
            self.data.qpos[addr] = value
        self._mujoco.mj_forward(self.model, self.data)

        if self.viewer is not None:
            if not self.viewer.is_running():
                self.viewer.close()
                self.viewer = None
            else:
                now = time.perf_counter()
                if now - self._last_viewer_sync >= self._viewer_period_s:
                    self.viewer.sync()
                    self._last_viewer_sync = now

        return self.data.xpos[self.ee_body_id].copy(), self.data.xquat[self.ee_body_id].copy()

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def build_simulation_reporter(config: Config):
    if not config.visualization.spawn_rerun:
        return None

    try:
        reporter = SimulationRerunReporter(
            app_name=f"AM-D02 Simulation J{config.target_joint + 1}",
            spawn=True,
        )
        reporter.init()
        return reporter
    except Exception as exc:
        log_info(f"Rerun 仿真可视化初始化失败，将继续运行: {exc}")
        return None


def build_hardware_reporter(config: Config):
    if not config.visualization.spawn_rerun:
        return None

    try:
        reporter = HardwareRerunReporter(
            app_name=f"AM-D02 Hardware J{config.target_joint + 1}",
            joint_names=config.robot.joint_names,
            spawn=True,
        )
        reporter.init()
        return reporter
    except Exception as exc:
        log_info(f"Rerun 真机可视化初始化失败，将继续运行: {exc}")
        return None


def build_pose_estimator(config: Config):
    if not config.visualization.render:
        return None

    try:
        return PoseEstimator(
            model_path=str(config.robot.urdf_path),
            joint_names=list(config.robot.joint_names),
            end_effector_body=config.robot.end_effector_body,
            tcp_offset=config.robot.tcp_offset,
            render=True,
            viewer_fps=config.visualization.viewer_fps,
        )
    except Exception as exc:
        log_info(f"MuJoCo 真机姿态可视化初始化失败，将继续运行: {exc}")
        return None


__all__ = [
    "HardwareRerunReporter",
    "PoseEstimator",
    "SimulationRerunReporter",
    "build_hardware_reporter",
    "build_pose_estimator",
    "build_simulation_reporter",
]
