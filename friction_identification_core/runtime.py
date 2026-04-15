from __future__ import annotations

"""Shared runtime helpers for CLI entry points and result persistence."""

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def ensure_project_root_on_sys_path() -> Path:
    """Make sure absolute package imports work when a module is run as a script."""

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    return PROJECT_ROOT


def ensure_results_dir() -> Path:
    """Create and return the default result directory."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def log_info(message: str) -> None:
    """Emit a flushed info log line."""

    print(f"[INFO] {message}", flush=True)


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write UTF-8 JSON with stable formatting."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return path
