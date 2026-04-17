from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from friction_identification_core.config import DEFAULT_CONFIG_PATH, Config, load_config


def _default_config_argument() -> str:
    return str(DEFAULT_CONFIG_PATH.relative_to(DEFAULT_CONFIG_PATH.parents[1]))


def _apply_overrides(config: Config, *, joint: int | None, output: str | None) -> Config:
    updated = config
    if joint is not None:
        joint_index = int(joint)
        if 1 <= joint_index <= config.joint_count:
            joint_index -= 1
        if not 0 <= joint_index < config.joint_count:
            raise ValueError(f"--joint must be within [1, {config.joint_count}] or [0, {config.joint_count - 1}]")
        updated = replace(updated, identification=replace(updated.identification, target_joint=joint_index))

    if output:
        output_path = updated.resolve_project_path(output)
        updated = replace(
            updated,
            output=replace(updated.output, results_dir=Path(output_path).resolve()),
        )
    return updated


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=_default_config_argument(),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--joint",
        type=int,
        default=None,
        help="Target joint index (supports 1-7 or 0-based index).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output directory override.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified friction-identification CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run identification through the unified pipeline.")
    _add_common_arguments(run_parser)
    run_parser.add_argument(
        "--source",
        choices=("sim", "hw"),
        required=True,
        help="Choose the simulation or hardware data source.",
    )
    run_parser.add_argument(
        "--mode",
        choices=("collect", "compensate", "full_feedforward"),
        default="collect",
        help="Execution mode for the selected source.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config = _apply_overrides(config, joint=args.joint, output=args.output)

    if args.source == "sim":
        if args.mode == "compensate":
            parser.error("--mode compensate is only available with --source hw.")
        from friction_identification_core.pipeline import run_simulation

        run_simulation(config, mode=args.mode)
        return

    from friction_identification_core.pipeline import run_hardware

    run_hardware(config, mode=args.mode)


if __name__ == "__main__":
    main()
