#!/bin/bash

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="$PROJECT_ROOT/requirements.txt"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_usage() {
    cat <<EOF
Usage:
  ./run.sh [sim|real] [args]

Examples:
  ./run.sh
  ./run.sh sim --config friction_identification_core/config/default.yaml
  ./run.sh real
  ./run.sh real --mode compensate
EOF
}

log_step() {
    echo -e "${YELLOW}[STEP] $1...${NC}"
}

log_success() {
    echo -e "${GREEN}[OK] $1${NC}"
}

log_error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

check_python() {
    log_step "Checking Python environment"
    if ! command -v python3 >/dev/null 2>&1; then
        log_error "python3 not found. Please install Python 3.10+"
    fi
    log_success "Python3 detected"
}

sync_dependencies() {
    if [ ! -f "$REQ_FILE" ]; then
        return
    fi

    log_step "Syncing dependencies"
    if pip3 install -q -r "$REQ_FILE"; then
        log_success "Dependencies ready"
    else
        log_error "Dependency install failed. Run: pip3 install -r $REQ_FILE"
    fi
}

verify_core() {
    log_step "Verifying MuJoCo"
    if python3 -c "import mujoco" >/dev/null 2>&1; then
        log_success "MuJoCo loaded successfully"
    else
        log_error "Failed to load MuJoCo. Check your environment."
    fi
}

verify_real_dependencies() {
    log_step "Verifying real-mode dependencies"
    if python3 -c "import numpy, serial" >/dev/null 2>&1; then
        log_success "Real-mode dependencies loaded successfully"
    else
        log_error "Failed to load real-mode dependencies. Check pyserial/numpy installation."
    fi
}

main() {
    case "${1:-}" in
        help|-h|--help)
            print_usage
            exit 0
            ;;
    esac

    local mode="sim"
    if [ "${1:-}" = "sim" ] || [ "${1:-}" = "real" ]; then
        mode="$1"
        shift
    fi

    cd "$PROJECT_ROOT"
    check_python
    sync_dependencies
    if [ "$mode" = "sim" ]; then
        verify_core
        log_step "Launching friction identification (sim mode)"
        python3 -m friction_identification_core.cli.simulate "$@"
    elif [ "$mode" = "real" ]; then
        verify_real_dependencies
        log_step "Launching UART real mode"
        python3 -m friction_identification_core.cli.deploy "$@"
    else
        log_error "Unknown mode: $mode"
    fi
}

main "$@"
