#!/usr/bin/env python3

from __future__ import annotations

"""Primary real-UART entry point for the friction-identification CLI."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from friction_identification_core.cli.real_uart_cli import main


if __name__ == "__main__":
    main()
