#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

from friction_model import (
    MODEL_COEFFICIENT,
    build_prediction_rows,
    fit_all_joints,
    load_samples_from_csv,
    load_seed_map,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit OpenArm-style tanh friction parameters from offline joint data."
    )
    parser.add_argument("--csv", required=True, help="Input CSV with joint samples.")
    parser.add_argument(
        "--seed-json",
        help="Optional JSON file with existing OpenArm friction parameters.",
    )
    parser.add_argument(
        "--joint",
        action="append",
        help="Limit fitting to one or more joint names. Can be repeated.",
    )
    parser.add_argument(
        "--default-joint",
        default="joint1",
        help="Joint name to use when the CSV has no joint column.",
    )
    parser.add_argument(
        "--coriolis-scale",
        type=float,
        default=0.0,
        help="Scale used when deriving friction_torque from measured_torque - gravity - scale*coriolis.",
    )
    parser.add_argument(
        "--seed-regularization",
        type=float,
        default=0.02,
        help="Pull the fitted parameters toward the provided seed values. Use 0 to disable.",
    )
    parser.add_argument("--k-min", type=float, default=0.5)
    parser.add_argument("--k-max", type=float, default=400.0)
    parser.add_argument("--coarse-candidates", type=int, default=160)
    parser.add_argument("--refinement-rounds", type=int, default=4)
    parser.add_argument("--refinement-candidates", type=int, default=80)
    parser.add_argument(
        "--output",
        default="fitted_friction_params.json",
        help="Where to write the fitted parameter JSON.",
    )
    parser.add_argument(
        "--predictions-csv",
        help="Optional CSV path for per-sample fitted predictions.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    sample_map = load_samples_from_csv(
        args.csv,
        joint_filter=args.joint,
        default_joint=args.default_joint,
        coriolis_scale=args.coriolis_scale,
    )
    seed_map = load_seed_map(args.seed_json) if args.seed_json else {}

    fit_results = fit_all_joints(
        sample_map,
        seed_map=seed_map,
        seed_regularization=args.seed_regularization,
        k_bounds=(args.k_min, args.k_max),
        coarse_candidates=args.coarse_candidates,
        refinement_rounds=args.refinement_rounds,
        refinement_candidates=args.refinement_candidates,
        model_coefficient=MODEL_COEFFICIENT,
    )

    output_payload = {
        "model": "openarm_tanh_friction",
        "model_coefficient": MODEL_COEFFICIENT,
        "source_csv": str(Path(args.csv).resolve()),
        "seed_json": str(Path(args.seed_json).resolve()) if args.seed_json else None,
        "results": {
            joint_name: {
                "parameters": result.parameters.to_dict(),
                "metrics": result.metrics,
                "sample_count": result.sample_count,
            }
            for joint_name, result in fit_results.items()
        },
    }

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2, sort_keys=True)

    if args.predictions_csv:
        rows = build_prediction_rows(sample_map, fit_results)
        _write_prediction_csv(args.predictions_csv, rows)

    print(f"Saved fitted parameters to {Path(args.output).resolve()}")
    for joint_name, result in fit_results.items():
        params = result.parameters
        metrics = result.metrics
        print(
            f"{joint_name}: fc={params.fc:.6f}, k={params.k:.6f}, "
            f"fv={params.fv:.6f}, fo={params.fo:.6f}, rmse={metrics['rmse']:.6e}"
        )

    if args.predictions_csv:
        print(f"Saved prediction residuals to {Path(args.predictions_csv).resolve()}")

    return 0


def _write_prediction_csv(path: str, rows: list[Dict[str, float]]) -> None:
    fieldnames = [
        "joint",
        "velocity",
        "friction_torque",
        "predicted_friction",
        "residual",
        "weight",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
