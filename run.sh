#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG_FILE="friction_identification_core/default.yaml"
HISTORY_FILE="$PROJECT_ROOT/.run_history"

CONFIG_FILE="$DEFAULT_CONFIG_FILE"
CONFIG_MODE="sequential"
LAST_MODE=""
LAST_CONFIG_FILE=""

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
  ./run.sh [default|sequential|collect|compensate|compare] [extra args]
  ./run.sh help

Examples:
  ./run.sh
  ./run.sh default
  ./run.sh sequential
  ./run.sh collect
  ./run.sh compensate --output results/debug
  ./run.sh compare
  ./run.sh compare --compare-all
EOF
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

describe_mode() {
    case "$1" in
        default)
            printf '%s\n' "按配置默认模式运行"
            ;;
        sequential)
            printf '%s\n' "逐个关节辨识"
            ;;
        collect)
            printf '%s\n' "7轴并行采集 + 并行辨识"
            ;;
        compensate)
            printf '%s\n' "仅发送摩擦补偿力矩验证"
            ;;
        compare)
            printf '%s\n' "对比多次启动之间的辨识结果"
            ;;
        *)
            printf '%s\n' "$1"
            ;;
    esac
}

resolve_requested_mode() {
    case "$1" in
        ""|default|auto)
            printf '%s\n' "$CONFIG_MODE"
            ;;
        parallel)
            printf '%s\n' "collect"
            ;;
        *)
            printf '%s\n' "$1"
            ;;
    esac
}

load_config_mode() {
    if ! command -v python3 >/dev/null 2>&1; then
        CONFIG_MODE="sequential"
        return
    fi

    local detected_mode=""
    detected_mode="$(
        cd "$PROJECT_ROOT" && python3 - "$CONFIG_FILE" <<'PY'
import sys

import yaml

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    payload = yaml.safe_load(handle) or {}

identification = payload.get("identification") or {}
mode = str(identification.get("mode", "sequential")).strip().lower()
print(mode)
PY
    )" || detected_mode="sequential"

    case "$detected_mode" in
        sequential)
            CONFIG_MODE="sequential"
            ;;
        parallel)
            CONFIG_MODE="collect"
            ;;
        *)
            CONFIG_MODE="sequential"
            ;;
    esac
}

load_history() {
    if [ ! -f "$HISTORY_FILE" ]; then
        return
    fi

    # shellcheck disable=SC1090
    . "$HISTORY_FILE"
    LAST_MODE="${LAST_MODE:-}"
    LAST_CONFIG_FILE="${LAST_CONFIG_FILE:-}"
}

save_history() {
    local last_mode="$1"
    local last_config="$2"

    {
        printf 'LAST_MODE=%q\n' "$last_mode"
        printf 'LAST_CONFIG_FILE=%q\n' "$last_config"
    } > "$HISTORY_FILE"
}

show_menu() {
    local prompt_options="1-5/c/q"
    local config_mode_desc
    config_mode_desc="$(describe_mode "$CONFIG_MODE")"
    if [[ -t 1 ]] && command -v clear >/dev/null 2>&1; then
        clear
    fi

    echo -e "${BOLD}========================================"
    echo -e "      OpenArm 摩擦力辨识"
    echo -e "========================================${NC}"
    echo ""
    echo "请选择运行模式:"
    echo ""
    echo -e "  ${CYAN}[默认]${NC}"
    echo -e "    ${GREEN}1)${NC} default      按配置默认模式运行 ${BLUE}(当前: $CONFIG_MODE, $config_mode_desc)${NC}"
    echo ""
    echo -e "  ${CYAN}[真机]${NC}"
    echo -e "    ${GREEN}2)${NC} sequential   逐个关节辨识 ${BLUE}(当前主流程)${NC}"
    echo -e "    ${GREEN}3)${NC} collect      7轴并行采集 + 并行辨识 ${BLUE}(兼容旧模式)${NC}"
    echo -e "    ${GREEN}4)${NC} compensate   仅发送摩擦补偿力矩验证"
    echo -e "    ${GREEN}5)${NC} compare      对比多次启动之间的辨识结果"
    echo ""
    echo -e "  ${CYAN}[设置]${NC}"
    echo -e "    ${YELLOW}c)${NC} 切换配置文件 ${BLUE}(当前: $(relative_path "$CONFIG_FILE"), 默认模式: $CONFIG_MODE)${NC}"
    if [ -n "$LAST_MODE" ]; then
        prompt_options="1-5/c/r/q"
        echo -e "    ${YELLOW}r)${NC} 重复上次 ${BLUE}($LAST_MODE, $(describe_mode "$LAST_MODE"), 配置: $(relative_path "${LAST_CONFIG_FILE:-$CONFIG_FILE}"))${NC}"
    fi
    echo -e "    ${YELLOW}q)${NC} 退出"
    echo ""
    echo -ne "请输入选项 ${GREEN}[${prompt_options}]${NC}: "
}

select_config() {
    local config

    echo ""
    echo -e "${CYAN}请输入配置文件路径 (直接回车恢复默认):${NC} "
    read -r config
    if [ -z "$config" ]; then
        CONFIG_FILE="$DEFAULT_CONFIG_FILE"
        load_config_mode
        log_success "已恢复默认配置: $(relative_path "$CONFIG_FILE")"
    elif [ -f "$config" ]; then
        CONFIG_FILE="$config"
        load_config_mode
        log_success "已设置配置文件: $(relative_path "$CONFIG_FILE")"
    else
        echo -e "${RED}[WARN] 文件不存在: $config${NC}"
    fi
    sleep 1
}

run_command() {
    local requested_mode="$1"
    shift

    check_python
    load_config_mode

    local mode
    mode="$(resolve_requested_mode "$requested_mode")"
    case "$mode" in
        sequential|collect|compensate|compare)
            ;;
        *)
            fatal "Unknown mode: $requested_mode"
            ;;
    esac

    local command=(python3 -m friction_identification_core --mode "$mode" --config "$CONFIG_FILE")
    if [ "$#" -gt 0 ]; then
        command+=("$@")
    fi

    if [ "$requested_mode" = "default" ] || [ "$requested_mode" = "auto" ] || [ -z "$requested_mode" ]; then
        echo -e "${YELLOW}[INFO] 配置默认模式:${NC} $mode ($(describe_mode "$mode"))"
    fi
    echo -e "${YELLOW}[INFO] Running:${NC} ${command[*]}"
    (
        cd "$PROJECT_ROOT"
        "${command[@]}"
    )
    save_history "$mode" "$CONFIG_FILE"
}

main() {
    load_history
    load_config_mode

    case "${1:-}" in
        "" )
            ;;
        help|-h|--help)
            print_usage
            exit 0
            ;;
        default|auto|sequential|collect|parallel|compensate)
            local mode="$1"
            shift
            run_command "$mode" "$@"
            exit $?
            ;;
        compare)
            local mode="$1"
            shift
            run_command "$mode" "$@"
            exit $?
            ;;
        *)
            fatal "Unknown mode: $1"
            ;;
    esac

    while true; do
        show_menu
        local choice
        read -r choice
        case "$choice" in
            1) run_command "default"; break ;;
            2) run_command "sequential"; break ;;
            3) run_command "collect"; break ;;
            4) run_command "compensate"; break ;;
            5) run_command "compare"; break ;;
            c|C) select_config ;;
            r|R)
                if [ -n "$LAST_MODE" ]; then
                    if [ -n "$LAST_CONFIG_FILE" ]; then
                        CONFIG_FILE="$LAST_CONFIG_FILE"
                        load_config_mode
                    fi
                    run_command "$LAST_MODE"
                    break
                fi
                ;;
            q|Q) exit 0 ;;
            *)
                echo -e "${RED}[WARN] 无效选项，请重试${NC}"
                sleep 1
                ;;
        esac
    done
}

main "$@"
