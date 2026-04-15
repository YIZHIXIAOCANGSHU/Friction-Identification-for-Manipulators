from __future__ import annotations

from pathlib import Path
from typing import Sequence

import rerun as rr
import rerun.blueprint as rrb

from .models import FrictionIdentificationResult, FrictionSampleBatch, TrackingEvaluationResult


class FrictionRerunReporter:
    def __init__(self, *, app_name: str = "Friction Identification", spawn: bool = True):
        self.app_name = app_name
        self.spawn = spawn

    def init(self) -> None:
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
            rr.log(
                f"tracking_compare/{axis_name}",
                rr.SeriesLines(
                    colors=[[120, 120, 120], [230, 110, 70], [70, 120, 230]],
                    names=["expected", "zero_friction_controller", "identified_parameters"],
                    widths=[1.5, 2.0, 2.0],
                ),
                static=True,
            )

        blueprint = rrb.Blueprint(
            rrb.Tabs(
                rrb.Vertical(
                    rrb.TextDocumentView(name="Summary", origin="/summary/report"),
                    rrb.BarChartView(name="Validation RMSE", origin="/summary/validation_rmse"),
                    rrb.BarChartView(name="Tracking EE RMSE", origin="/summary/tracking_ee_rmse"),
                    name="Summary",
                ),
                rrb.Grid(
                    *[
                        rrb.TimeSeriesView(name=f"Joint {joint_idx + 1}", origin=f"/friction/joint_{joint_idx + 1}")
                        for joint_idx in range(7)
                    ],
                    name="Per Joint",
                ),
                rrb.Grid(
                    *[
                        rrb.TimeSeriesView(name=f"Velocity {joint_idx + 1}", origin=f"/velocity/joint_{joint_idx + 1}")
                        for joint_idx in range(7)
                    ],
                    name="Velocity",
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
                rrb.Vertical(
                    rrb.Spatial3DView(name="Tracking Compare 3D", origin="/tracking_3d"),
                    rrb.Horizontal(
                        rrb.TimeSeriesView(name="Track X", origin="/tracking_compare/x"),
                        rrb.TimeSeriesView(name="Track Y", origin="/tracking_compare/y"),
                        rrb.TimeSeriesView(name="Track Z", origin="/tracking_compare/z"),
                    ),
                    name="Tracking Compare",
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
    ) -> None:
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
        rr.log(
            "trajectory_3d/start_end",
            rr.Points3D(
                [raw_batch.ee_pos_cmd[0], raw_batch.ee_pos_cmd[-1]],
                colors=[[60, 210, 60], [30, 120, 30]],
                radii=[0.01, 0.012],
                labels=["expected_start", "expected_end"],
            ),
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

        if tracking_results:
            reference_batch = tracking_results[0].batch
            rr.log(
                "tracking_3d/expected_path",
                rr.LineStrips3D([reference_batch.ee_pos_cmd], colors=[[120, 120, 120]], radii=[0.002]),
                static=True,
            )
            for tracking_result, color in zip(
                tracking_results,
                ([230, 110, 70], [70, 120, 230]),
            ):
                rr.log(
                    f"tracking_3d/{tracking_result.label}_path",
                    rr.LineStrips3D([tracking_result.batch.ee_pos], colors=[color], radii=[0.0015]),
                    static=True,
                )

            for sample_idx, time_s in enumerate(reference_batch.time):
                rr.set_time_seconds("sim_time", float(time_s))
                rr.log("tracking_compare/x/expected", rr.Scalars(reference_batch.ee_pos_cmd[sample_idx, 0]))
                rr.log("tracking_compare/y/expected", rr.Scalars(reference_batch.ee_pos_cmd[sample_idx, 1]))
                rr.log("tracking_compare/z/expected", rr.Scalars(reference_batch.ee_pos_cmd[sample_idx, 2]))
                for tracking_result in tracking_results:
                    rr.log(
                        f"tracking_compare/x/{tracking_result.label}",
                        rr.Scalars(tracking_result.batch.ee_pos[sample_idx, 0]),
                    )
                    rr.log(
                        f"tracking_compare/y/{tracking_result.label}",
                        rr.Scalars(tracking_result.batch.ee_pos[sample_idx, 1]),
                    )
                    rr.log(
                        f"tracking_compare/z/{tracking_result.label}",
                        rr.Scalars(tracking_result.batch.ee_pos[sample_idx, 2]),
                    )

        for sample_idx, time_s in enumerate(fit_batch.time):
            rr.set_time_seconds("sim_time", float(time_s))
            for joint_idx, joint_name in enumerate(result.joint_names):
                rr.log(
                    f"friction/joint_{joint_idx + 1}/measured",
                    rr.Scalars(fit_batch.tau_friction[sample_idx, joint_idx]),
                )
                rr.log(
                    f"friction/joint_{joint_idx + 1}/predicted",
                    rr.Scalars(result.predicted_torque[sample_idx, joint_idx]),
                )
                rr.log(
                    f"velocity/joint_{joint_idx + 1}/{joint_name}",
                    rr.Scalars(fit_batch.qd[sample_idx, joint_idx]),
                )

        rr.log("summary/validation_rmse", rr.BarChart(result.validation_rmse))
        if tracking_results:
            rr.log(
                "summary/tracking_ee_rmse",
                rr.BarChart([tracking_result.ee_position_rmse for tracking_result in tracking_results]),
            )

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
        if tracking_results:
            lines.extend(
                [
                    "",
                    "## Before/After Tracking Comparison",
                    "",
                    "| Controller | Mean Joint RMSE (rad) | EE RMSE (m) | EE Max Error (m) |",
                    "|---|---:|---:|---:|",
                ]
            )
            for tracking_result in tracking_results:
                lines.append(
                    "| "
                    f"{tracking_result.label} | "
                    f"{tracking_result.mean_joint_rmse:.6f} | "
                    f"{tracking_result.ee_position_rmse:.6f} | "
                    f"{tracking_result.ee_max_error:.6f} |"
                )
        rr.log("summary/report", rr.TextDocument("\n".join(lines), media_type="text/markdown"))

    def close(self) -> None:
        rr.disconnect()
