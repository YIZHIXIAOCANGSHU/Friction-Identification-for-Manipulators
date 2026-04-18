from __future__ import annotations

import argparse

from friction_identification_core.config import DEFAULT_CONFIG_PATH, apply_overrides, load_config
from friction_identification_core.pipeline import run_sequential_identification
from friction_identification_core.runtime import log_info


def _default_config_argument() -> str:
    return str(DEFAULT_CONFIG_PATH.relative_to(DEFAULT_CONFIG_PATH.parents[1]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sequential single-motor friction identification CLI.")
    parser.add_argument(
        "--config",
        default=_default_config_argument(),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--mode",
        choices=("default", "sequential"),
        default="sequential",
        help="default is an alias of sequential.",
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


def main(argv: list[str] | None = None) -> None:
    try:
        parser = build_parser()
        args = parser.parse_args(argv)
        config = load_config(args.config)
        config = apply_overrides(config, output=args.output, motors=args.motors, groups=args.groups)
        run_sequential_identification(config)
    except ValueError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
    except KeyboardInterrupt:
        log_info("Interrupted by user.")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
