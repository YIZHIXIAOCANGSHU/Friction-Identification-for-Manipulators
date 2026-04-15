from __future__ import annotations

from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from .models import FrictionIdentificationResult, FrictionSampleBatch


class FrictionRerunReporter:
    def __init__(self, *, app_name: str = "Friction Identification", spawn: bool = True):
        self.app_name = app_name
        self.spawn = spawn

    def init(self) -> None:
        rr.init(self.app_name, spawn=self.spawn)
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
                rrb.Grid(
                    *[
                        rrb.TimeSeriesView(name=f"Velocity {joint_idx + 1}", origin=f"/velocity/joint_{joint_idx + 1}")
                        for joint_idx in range(7)
                    ],
                    name="Velocity",
                ),
            )
        )
        rr.send_blueprint(blueprint)

    def log(self, batch: FrictionSampleBatch, result: FrictionIdentificationResult, output_dir: Path) -> None:
        for sample_idx, time_s in enumerate(batch.time):
            rr.set_time_seconds("sim_time", float(time_s))
            for joint_idx, joint_name in enumerate(result.joint_names):
                rr.log(
                    f"friction/joint_{joint_idx + 1}/measured",
                    rr.Scalars(batch.tau_friction[sample_idx, joint_idx]),
                )
                rr.log(
                    f"friction/joint_{joint_idx + 1}/predicted",
                    rr.Scalars(result.predicted_torque[sample_idx, joint_idx]),
                )
                rr.log(
                    f"velocity/joint_{joint_idx + 1}/{joint_name}",
                    rr.Scalars(batch.qd[sample_idx, joint_idx]),
                )

        rr.log("summary/validation_rmse", rr.BarChart(result.validation_rmse))

        lines = [
            "# Friction Identification Summary",
            "",
            f"- Samples: {batch.time.shape[0]}",
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
        rr.disconnect()
