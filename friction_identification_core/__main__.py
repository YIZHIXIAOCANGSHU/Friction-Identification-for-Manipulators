from __future__ import annotations

import argparse

from friction_identification_core.config import DEFAULT_CONFIG_PATH, apply_overrides, load_config
from friction_identification_core.pipeline import run_compensation_validation, run_sequential_identification
from friction_identification_core.runtime import log_info


def _default_config_argument() -> str:
    return str(DEFAULT_CONFIG_PATH.relative_to(DEFAULT_CONFIG_PATH.parents[1]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-motor runtime identify/compensate CLI.")
    parser.add_argument(
        "--config",
        default=_default_config_argument(),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--mode",
        choices=("identify", "compensate", "default", "sequential"),
        default="identify",
        help="identify/compensate are the primary modes; default/sequential alias to identify.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--motors",
        default=None,
        help="Target motor ids, for example '1,3,5' or 'all'.",
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=None,
        help="Override the configured group count.",
    )
    return parser


def _normalize_mode(mode: str) -> str:
    if mode in ("default", "sequential"):
        return "identify"
    return str(mode)


def main(argv: list[str] | None = None) -> None:
    try:
        parser = build_parser()
        args = parser.parse_args(argv)
        config = load_config(args.config)
        config = apply_overrides(config, output=args.output, motors=args.motors, groups=args.groups)
        mode = _normalize_mode(str(args.mode))
        if mode == "compensate":
            run_compensation_validation(config, show_rerun_viewer=True)
        else:
            run_sequential_identification(config, show_rerun_viewer=True)
    except ValueError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
    except KeyboardInterrupt:
        log_info("Interrupted by user.")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
