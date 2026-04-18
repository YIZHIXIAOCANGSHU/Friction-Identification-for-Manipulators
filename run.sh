#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG_FILE="friction_identification_core/default.yaml"

print_usage() {
    cat <<EOF
Usage:
  ./run.sh
  ./run.sh [default|sequential] [extra args]
  ./run.sh --config custom.yaml --motors 1,3,5 --groups 2
  ./run.sh help

Examples:
  ./run.sh
  ./run.sh sequential
  ./run.sh default --motors 1,3,5
  ./run.sh --config friction_identification_core/default.yaml --groups 3
EOF
}

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] python3 not found. Please install Python 3.10+ first." >&2
    exit 1
fi

MODE="default"
if [[ "${1:-}" == "help" || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    print_usage
    exit 0
fi
if [[ "${1:-}" == "default" || "${1:-}" == "sequential" ]]; then
    MODE="$1"
    shift
fi

COMMAND=(python3 -m friction_identification_core --mode "$MODE")
HAS_CONFIG=0
for ARG in "$@"; do
    if [[ "$ARG" == "--config" ]]; then
        HAS_CONFIG=1
        break
    fi
done
if [[ $HAS_CONFIG -eq 0 ]]; then
    COMMAND+=(--config "$DEFAULT_CONFIG_FILE")
fi
COMMAND+=("$@")

cd "$PROJECT_ROOT"
exec "${COMMAND[@]}"
