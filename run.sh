#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG_FILE="friction_identification_core/default.yaml"
HISTORY_FILE="$PROJECT_ROOT/.run_history"

CONFIG_FILE="$DEFAULT_CONFIG_FILE"
TARGET_JOINT=""

LAST_SOURCE=""
LAST_MODE=""
LAST_CONFIG_FILE=""
LAST_TARGET_JOINT=""

RESOLVED_SOURCE=""
RESOLVED_MODE=""
RESOLVED_CONFIG_FILE=""
RESOLVED_JOINT=""
FORWARDED_ARGS=()

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

print_usage() {
    cat <<EOF
Usage:
  ./run.sh
  ./run.sh [sim|sim-ff|hw|hw-comp|hw-ff] [extra args]
  ./run.sh real [extra args]
  ./run.sh help

Interactive:
  ./run.sh
    Open the menu and choose a mode by number.

Shortcuts:
  sim      Simulation collect
  sim-ff   Simulation full feedforward
  hw       Hardware collect
  hw-comp  Hardware compensation validation
  hw-ff    Hardware full feedforward
  real     Legacy alias of hw

Examples:
  ./run.sh
  ./run.sh sim --joint 5
  ./run.sh sim-ff --config friction_identification_core/default.yaml
  ./run.sh hw-comp
  ./run.sh hw --mode full_feedforward --output results/debug
EOF
}

log_info() {
    echo -e "${YELLOW}[INFO] $1${NC}"
}

log_success() {
    echo -e "${GREEN}[OK] $1${NC}"
}

fatal() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
    exit 1
}

relative_path() {
    local path="$1"
    case "$path" in
        "$PROJECT_ROOT"/*)
            printf '%s\n' "${path#$PROJECT_ROOT/}"
            ;;
        *)
            printf '%s\n' "$path"
            ;;
    esac
}

check_python() {
    if ! command -v python3 >/dev/null 2>&1; then
        fatal "python3 not found. Please install Python 3.10+ first."
    fi
}

load_history() {
    if [ ! -f "$HISTORY_FILE" ]; then
        return
    fi

    # shellcheck disable=SC1090
    . "$HISTORY_FILE"
    LAST_SOURCE="${LAST_SOURCE:-}"
    LAST_MODE="${LAST_MODE:-}"
    LAST_CONFIG_FILE="${LAST_CONFIG_FILE:-}"
    LAST_TARGET_JOINT="${LAST_TARGET_JOINT:-}"
}

save_history() {
    local last_source="$1"
    local last_mode="$2"
    local last_config="$3"
    local last_joint="$4"

    {
        printf 'LAST_SOURCE=%q\n' "$last_source"
        printf 'LAST_MODE=%q\n' "$last_mode"
        printf 'LAST_CONFIG_FILE=%q\n' "$last_config"
        printf 'LAST_TARGET_JOINT=%q\n' "$last_joint"
    } > "$HISTORY_FILE"
}

read_joint_from_config() {
    local config_path="$1"

    if [ ! -f "$config_path" ]; then
        printf '?\n'
        return
    fi

    if ! command -v python3 >/dev/null 2>&1; then
        printf '?\n'
        return
    fi

    python3 - "$config_path" <<'PY'
import sys
from pathlib import Path

try:
    import yaml
except Exception:
    print("?")
    raise SystemExit(0)

path = Path(sys.argv[1])
try:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    joint = payload.get("identification", {}).get("target_joint")
    if isinstance(joint, int):
        print(joint + 1)
    else:
        print("?")
except Exception:
    print("?")
PY
}

current_joint_label() {
    if [ -n "$TARGET_JOINT" ]; then
        printf '%s (override)\n' "$TARGET_JOINT"
        return
    fi
    read_joint_from_config "$CONFIG_FILE"
}

history_joint_label() {
    if [ -n "$LAST_TARGET_JOINT" ]; then
        printf '%s\n' "$LAST_TARGET_JOINT"
        return
    fi

    if [ -n "$LAST_CONFIG_FILE" ]; then
        read_joint_from_config "$LAST_CONFIG_FILE"
        return
    fi

    printf '?\n'
}

mode_label() {
    local selected_source="$1"
    local selected_mode="$2"

    case "${selected_source}:${selected_mode}" in
        sim:collect) printf 'sim collect' ;;
        sim:full_feedforward) printf 'sim full_feedforward' ;;
        hw:collect) printf 'hw collect' ;;
        hw:compensate) printf 'hw compensate' ;;
        hw:full_feedforward) printf 'hw full_feedforward' ;;
        *) printf '%s %s' "$selected_source" "$selected_mode" ;;
    esac
}

show_menu() {
    local current_joint
    local current_config
    local prompt_options="1-5/j/c/q"

    if [[ -t 1 ]] && command -v clear >/dev/null 2>&1; then
        clear
    fi

    current_joint="$(current_joint_label)"
    current_config="$(relative_path "$CONFIG_FILE")"

    echo -e "${BOLD}========================================"
    echo -e "   OpenArm 摩擦力辨识工具"
    echo -e "========================================${NC}"
    echo ""
    echo "请选择运行模式:"
    echo ""
    echo -e "  ${CYAN}[仿真]${NC}"
    echo -e "    ${GREEN}1)${NC} sim        仿真采集"
    echo -e "    ${GREEN}2)${NC} sim-ff     仿真全前馈"
    echo ""
    echo -e "  ${CYAN}[真机]${NC}"
    echo -e "    ${GREEN}3)${NC} hw         真机采集"
    echo -e "    ${GREEN}4)${NC} hw-comp    真机补偿验证"
    echo -e "    ${GREEN}5)${NC} hw-ff      真机全前馈"
    echo ""
    echo -e "  ${CYAN}[设置]${NC}"
    echo -e "    ${YELLOW}j)${NC} 修改目标关节 ${BLUE}(当前: ${current_joint})${NC}"
    echo -e "    ${YELLOW}c)${NC} 切换配置文件 ${BLUE}(当前: ${current_config})${NC}"
    if [ -n "$LAST_SOURCE" ] && [ -n "$LAST_MODE" ]; then
        prompt_options="1-5/j/c/r/q"
        echo -e "    ${YELLOW}r)${NC} 重复上次 ${BLUE}($(mode_label "$LAST_SOURCE" "$LAST_MODE"), 关节: $(history_joint_label))${NC}"
    fi
    echo -e "    ${YELLOW}q)${NC} 退出"
    echo ""
    echo -ne "请输入选项 ${GREEN}[${prompt_options}]${NC}: "
}

select_joint() {
    local joint

    echo ""
    echo -e "${CYAN}请输入目标关节 (1-7，直接回车恢复配置文件设置):${NC} "
    read -r joint
    if [ -z "$joint" ]; then
        TARGET_JOINT=""
        log_success "已恢复为配置文件中的目标关节"
    elif [[ "$joint" =~ ^[1-7]$ ]]; then
        TARGET_JOINT="$joint"
        log_success "已设置目标关节: $joint"
    else
        echo -e "${RED}[WARN] 无效输入，关节必须是 1-7${NC}"
    fi
    sleep 1
}

select_config() {
    local config

    echo ""
    echo -e "${CYAN}请输入配置文件路径 (直接回车恢复默认):${NC} "
    read -r config
    if [ -z "$config" ]; then
        CONFIG_FILE="$DEFAULT_CONFIG_FILE"
        log_success "已恢复默认配置: $(relative_path "$CONFIG_FILE")"
    elif [ -f "$config" ]; then
        CONFIG_FILE="$config"
        log_success "已设置配置文件: $(relative_path "$CONFIG_FILE")"
    else
        echo -e "${RED}[WARN] 文件不存在: $config${NC}"
    fi
    sleep 1
}

parse_runtime_overrides() {
    RESOLVED_SOURCE="$1"
    RESOLVED_MODE="$2"
    RESOLVED_CONFIG_FILE="$CONFIG_FILE"
    RESOLVED_JOINT="$TARGET_JOINT"
    FORWARDED_ARGS=()
    shift 2

    while [ "$#" -gt 0 ]; do
        case "$1" in
            --source)
                [ "$#" -ge 2 ] || fatal "--source requires a value."
                RESOLVED_SOURCE="$2"
                shift 2
                ;;
            --source=*)
                RESOLVED_SOURCE="${1#*=}"
                shift
                ;;
            --mode)
                [ "$#" -ge 2 ] || fatal "--mode requires a value."
                RESOLVED_MODE="$2"
                shift 2
                ;;
            --mode=*)
                RESOLVED_MODE="${1#*=}"
                shift
                ;;
            --config)
                [ "$#" -ge 2 ] || fatal "--config requires a value."
                RESOLVED_CONFIG_FILE="$2"
                shift 2
                ;;
            --config=*)
                RESOLVED_CONFIG_FILE="${1#*=}"
                shift
                ;;
            --joint)
                [ "$#" -ge 2 ] || fatal "--joint requires a value."
                RESOLVED_JOINT="$2"
                shift 2
                ;;
            --joint=*)
                RESOLVED_JOINT="${1#*=}"
                shift
                ;;
            *)
                FORWARDED_ARGS+=("$1")
                shift
                ;;
        esac
    done
}

validate_selection() {
    case "$RESOLVED_SOURCE" in
        sim|hw) ;;
        *) fatal "Unsupported source: $RESOLVED_SOURCE" ;;
    esac

    case "$RESOLVED_MODE" in
        collect|compensate|full_feedforward) ;;
        *) fatal "Unsupported mode: $RESOLVED_MODE" ;;
    esac

    if [ "$RESOLVED_SOURCE" = "sim" ] && [ "$RESOLVED_MODE" = "compensate" ]; then
        fatal "compensate mode is only available with hardware source."
    fi

    if [ ! -f "$RESOLVED_CONFIG_FILE" ]; then
        fatal "Config file not found: $RESOLVED_CONFIG_FILE"
    fi
}

verify_dependencies() {
    local selected_source="$1"

    log_info "Checking Python dependencies"
    if [ "$selected_source" = "sim" ]; then
        if python3 -c "import numpy, yaml, mujoco" >/dev/null 2>&1; then
            log_success "Simulation dependencies ready"
        else
            fatal "Missing simulation dependencies. Run: pip3 install -r requirements.txt"
        fi
    else
        if python3 -c "import numpy, yaml, serial" >/dev/null 2>&1; then
            log_success "Hardware dependencies ready"
        else
            fatal "Missing hardware dependencies. Run: pip3 install -r requirements.txt"
        fi
    fi
}

run_command() {
    local default_source="$1"
    local default_mode="$2"
    shift 2

    local command=()
    local rendered_command

    check_python
    parse_runtime_overrides "$default_source" "$default_mode" "$@"
    validate_selection
    verify_dependencies "$RESOLVED_SOURCE"

    command=(
        python3 -m friction_identification_core run
        --source "$RESOLVED_SOURCE"
        --mode "$RESOLVED_MODE"
        --config "$RESOLVED_CONFIG_FILE"
    )

    if [ -n "$RESOLVED_JOINT" ]; then
        command+=(--joint "$RESOLVED_JOINT")
    fi

    if [ "${#FORWARDED_ARGS[@]}" -gt 0 ]; then
        command+=("${FORWARDED_ARGS[@]}")
    fi

    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${GREEN}启动: $(mode_label "$RESOLVED_SOURCE" "$RESOLVED_MODE")${NC}"
    echo -e "${GREEN}配置: $(relative_path "$RESOLVED_CONFIG_FILE")${NC}"
    if [ -n "$RESOLVED_JOINT" ]; then
        echo -e "${GREEN}关节: $RESOLVED_JOINT${NC}"
    else
        echo -e "${GREEN}关节: $(read_joint_from_config "$RESOLVED_CONFIG_FILE") (config)${NC}"
    fi
    echo -e "${YELLOW}========================================${NC}"
    echo ""

    printf -v rendered_command '%q ' "${command[@]}"
    echo -e "${BLUE}执行: ${rendered_command% }${NC}"
    echo ""

    save_history "$RESOLVED_SOURCE" "$RESOLVED_MODE" "$RESOLVED_CONFIG_FILE" "$RESOLVED_JOINT"
    "${command[@]}"
}

run_last_command() {
    if [ -z "$LAST_SOURCE" ] || [ -z "$LAST_MODE" ]; then
        echo -e "${RED}[WARN] 暂无可重复的历史记录${NC}"
        sleep 1
        return 1
    fi

    CONFIG_FILE="${LAST_CONFIG_FILE:-$DEFAULT_CONFIG_FILE}"
    TARGET_JOINT="${LAST_TARGET_JOINT:-}"
    run_command "$LAST_SOURCE" "$LAST_MODE"
    return 0
}

run_shortcut() {
    local shortcut="$1"
    shift

    case "$shortcut" in
        sim)
            run_command "sim" "collect" "$@"
            ;;
        sim-ff)
            run_command "sim" "full_feedforward" "$@"
            ;;
        hw|real)
            run_command "hw" "collect" "$@"
            ;;
        hw-comp)
            run_command "hw" "compensate" "$@"
            ;;
        hw-ff)
            run_command "hw" "full_feedforward" "$@"
            ;;
        *)
            fatal "Unknown shortcut: $shortcut"
            ;;
    esac
}

main() {
    local choice

    cd "$PROJECT_ROOT"
    load_history

    case "${1:-}" in
        help|-h|--help)
            print_usage
            exit 0
            ;;
    esac

    if [ "$#" -gt 0 ]; then
        run_shortcut "$@"
        exit 0
    fi

    while true; do
        show_menu
        if ! read -r choice; then
            echo ""
            exit 0
        fi

        case "$choice" in
            1) run_command "sim" "collect"; break ;;
            2) run_command "sim" "full_feedforward"; break ;;
            3) run_command "hw" "collect"; break ;;
            4) run_command "hw" "compensate"; break ;;
            5) run_command "hw" "full_feedforward"; break ;;
            j|J) select_joint ;;
            c|C) select_config ;;
            r|R)
                if run_last_command; then
                    break
                fi
                ;;
            q|Q) exit 0 ;;
            *) echo -e "${RED}[WARN] 无效选项${NC}"; sleep 1 ;;
        esac
    done
}

main "$@"
