from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from friction_identification_core.config import DEFAULT_CONFIG_PATH, Config, load_config
from friction_identification_core.runtime import log_info


def _default_config_argument() -> str:
    return str(DEFAULT_CONFIG_PATH.relative_to(DEFAULT_CONFIG_PATH.parents[1]))


def _apply_overrides(config: Config, *, output: str | None) -> Config:
    if not output:
        return config
    output_path = config.resolve_project_path(output)
    return replace(
        config,
        output=replace(config.output, results_dir=Path(output_path).resolve()),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hardware-only parallel friction-identification CLI.")
    parser.add_argument(
        "--config",
        default=_default_config_argument(),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--mode",
        choices=("collect", "compensate", "compare"),
        default="collect",
        help="Run parallel collection/identification, compensation validation, or cross-run comparison.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--compare-limit",
        type=int,
        default=5,
        help="Number of archived collect runs to include in compare mode.",
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Compare all archived collect runs instead of only the latest ones.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    try:
        parser = build_parser()
        args = parser.parse_args(argv)

        config = load_config(args.config)
        config = _apply_overrides(config, output=args.output)

        if args.mode == "compare":
            from friction_identification_core.results import compare_saved_runs

            try:
                compare_saved_runs(
                    config,
                    limit=args.compare_limit,
                    compare_all=bool(args.compare_all),
                )
            except FileNotFoundError as exc:
                raise SystemExit(f"[ERROR] {exc}") from exc
            return

        from friction_identification_core.pipeline import run_hardware

        run_hardware(config, mode=args.mode)
    except KeyboardInterrupt:
        log_info("已收到 Ctrl+C，中止当前运行。")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
