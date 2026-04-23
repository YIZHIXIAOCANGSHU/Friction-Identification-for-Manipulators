#!/bin/bash

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG_FILE="friction_identification_core/default.yaml"
DEFAULT_OUTPUT_DIR="results"

print_usage() {
    cat <<'EOF'
Usage:
  ./run.sh
  ./run.sh help
  ./run.sh -h
  ./run.sh --help

Notes:
  - `./run.sh` 会启动交互式数字向导。
  - `./run.sh identify --motors 1,3,4` 这类旧式非交互调用已不再支持。
  - 自动化或脚本化调用请改用：
      python3 -m friction_identification_core --mode identify --config friction_identification_core/default.yaml
EOF
}

legacy_usage_error() {
    cat >&2 <<'EOF'
[ERROR] `run.sh` 现在只支持交互式启动。
请改用 `./run.sh` 进入菜单，或直接使用：
python3 -m friction_identification_core --mode identify --config friction_identification_core/default.yaml ...
EOF
    exit 1
}

print_menu() {
    local title="$1"
    shift
    echo
    echo "$title"
    for option in "$@"; do
        echo "  $option"
    done
}

read_menu_choice() {
    local result_var="$1"
    shift
    local prompt="$1"
    shift
    local allowed=("$@")
    local reply
    while true; do
        printf '%s' "$prompt"
        IFS= read -r reply || exit 1
        if [[ -z "$reply" || ! "$reply" =~ ^[0-9]+$ ]]; then
            echo "输入无效，请输入数字菜单项。"
            continue
        fi
        for item in "${allowed[@]}"; do
            if [[ "$reply" == "$item" ]]; then
                printf -v "$result_var" '%s' "$reply"
                return 0
            fi
        done
        echo "输入无效，请选择菜单中的数字。"
    done
}

read_custom_motors() {
    local result_var="$1"
    local motors_input
    while true; do
        printf '请输入 motor id 列表（例如 1,3,4，或 all）: '
        IFS= read -r motors_input || exit 1
        if [[ "$motors_input" == "all" || "$motors_input" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
            printf -v "$result_var" '%s' "$motors_input"
            return 0
        fi
        echo "输入无效，请输入 all 或逗号分隔的整数列表。"
    done
}

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] python3 not found. Please install Python 3.10+ first." >&2
    exit 1
fi

case "${1:-}" in
    help|-h|--help)
        print_usage
        exit 0
        ;;
esac

if [[ $# -gt 0 ]]; then
    legacy_usage_error
fi

echo "欢迎使用单电机顺序摩擦辨识交互式启动向导。"
echo "默认配置路径: ${DEFAULT_CONFIG_FILE}"
echo "除模式和电机外，其余参数将使用默认值。"

print_menu "模式菜单" \
    "1. identify" \
    "2. compensate" \
    "0. exit"
mode_choice=""
read_menu_choice mode_choice '请选择模式: ' 0 1 2
if [[ "$mode_choice" == "0" ]]; then
    echo "已退出。"
    exit 0
fi
MODE="identify"
if [[ "$mode_choice" == "2" ]]; then
    MODE="compensate"
fi
CONFIG_PATH="$DEFAULT_CONFIG_FILE"

print_menu "电机菜单" \
    "1. all" \
    "2. 输入 motor id 列表，例如 1,3,4"
motors_choice=""
read_menu_choice motors_choice '请选择电机: ' 1 2
MOTORS="all"
if [[ "$motors_choice" == "2" ]]; then
    read_custom_motors MOTORS
fi
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"

COMMAND=(
    python3
    -m
    friction_identification_core
    --mode
    "$MODE"
    --config
    "$CONFIG_PATH"
    --motors
    "$MOTORS"
)
COMMAND+=(--output "$OUTPUT_DIR")

echo
echo "最终命令:"
printf '  %q' "${COMMAND[@]}"
printf '\n'

echo
cd "$PROJECT_ROOT"
exec "${COMMAND[@]}"
