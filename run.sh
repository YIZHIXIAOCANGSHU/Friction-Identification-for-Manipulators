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
  - `./run.sh sequential --motors 1,3,5` 这类旧式非交互调用已不再支持。
  - 自动化或脚本化调用请改用：
      python3 -m friction_identification_core --mode sequential --config friction_identification_core/default.yaml
EOF
}

legacy_usage_error() {
    cat >&2 <<'EOF'
[ERROR] `run.sh` 现在只支持交互式启动。
请改用 `./run.sh` 进入菜单，或直接使用：
python3 -m friction_identification_core --mode sequential --config friction_identification_core/default.yaml ...
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

resolve_existing_path() {
    local raw_path="$1"
    if [[ -z "$raw_path" ]]; then
        return 1
    fi
    if [[ "$raw_path" == /* ]]; then
        [[ -f "$raw_path" ]] || return 1
        printf '%s\n' "$raw_path"
        return 0
    fi
    [[ -f "$raw_path" ]] || return 1
    local parent_dir
    parent_dir="$(cd "$(dirname "$raw_path")" && pwd)"
    printf '%s/%s\n' "$parent_dir" "$(basename "$raw_path")"
}

read_existing_config_path() {
    local result_var="$1"
    local config_input
    local resolved_path
    while true; do
        printf '请输入自定义配置路径: '
        IFS= read -r config_input || exit 1
        if resolved_path="$(resolve_existing_path "$config_input")"; then
            printf -v "$result_var" '%s' "$resolved_path"
            return 0
        fi
        echo "配置文件不存在，请重新输入。"
    done
}

read_custom_motors() {
    local result_var="$1"
    local motors_input
    while true; do
        printf '请输入 motor id 列表（例如 1,3,5，或 all）: '
        IFS= read -r motors_input || exit 1
        if [[ "$motors_input" == "all" || "$motors_input" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
            printf -v "$result_var" '%s' "$motors_input"
            return 0
        fi
        echo "输入无效，请输入 all 或逗号分隔的整数列表。"
    done
}

read_positive_integer() {
    local result_var="$1"
    shift
    local value
    local prompt="$1"
    while true; do
        printf '%s' "$prompt"
        IFS= read -r value || exit 1
        if [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
            printf -v "$result_var" '%s' "$value"
            return 0
        fi
        echo "输入无效，请输入正整数。"
    done
}

read_nonempty_text() {
    local result_var="$1"
    shift
    local value
    local prompt="$1"
    while true; do
        printf '%s' "$prompt"
        IFS= read -r value || exit 1
        if [[ -n "$value" ]]; then
            printf -v "$result_var" '%s' "$value"
            return 0
        fi
        echo "输入无效，请输入非空内容。"
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

print_menu "模式菜单" \
    "1. default" \
    "2. sequential" \
    "0. exit"
mode_choice=""
read_menu_choice mode_choice '请选择模式: ' 0 1 2
if [[ "$mode_choice" == "0" ]]; then
    echo "已退出。"
    exit 0
fi
MODE="default"
if [[ "$mode_choice" == "2" ]]; then
    MODE="sequential"
fi

print_menu "配置菜单" \
    "1. 使用默认配置 ${DEFAULT_CONFIG_FILE}" \
    "2. 输入自定义配置路径"
config_choice=""
read_menu_choice config_choice '请选择配置: ' 1 2
CONFIG_PATH="$DEFAULT_CONFIG_FILE"
if [[ "$config_choice" == "2" ]]; then
    read_existing_config_path CONFIG_PATH
fi

print_menu "电机菜单" \
    "1. all" \
    "2. 输入 motor id 列表，例如 1,3,5"
motors_choice=""
read_menu_choice motors_choice '请选择电机: ' 1 2
MOTORS="all"
if [[ "$motors_choice" == "2" ]]; then
    read_custom_motors MOTORS
fi

print_menu "group 菜单" \
    "1. 使用配置中的默认值" \
    "2. 输入自定义 groups"
groups_choice=""
read_menu_choice groups_choice '请选择 groups: ' 1 2
GROUP_COUNT_OVERRIDE=""
if [[ "$groups_choice" == "2" ]]; then
    read_positive_integer GROUP_COUNT_OVERRIDE '请输入自定义 groups: '
fi

print_menu "输出目录菜单" \
    "1. 使用默认 results" \
    "2. 输入自定义输出目录"
output_choice=""
read_menu_choice output_choice '请选择输出目录: ' 1 2
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
if [[ "$output_choice" == "2" ]]; then
    read_nonempty_text OUTPUT_DIR '请输入自定义输出目录: '
fi

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
if [[ -n "$GROUP_COUNT_OVERRIDE" ]]; then
    COMMAND+=(--groups "$GROUP_COUNT_OVERRIDE")
fi
COMMAND+=(--output "$OUTPUT_DIR")

echo
echo "最终命令:"
printf '  %q' "${COMMAND[@]}"
printf '\n'

print_menu "确认菜单" \
    "1. 开始运行" \
    "0. 取消退出"
confirm_choice=""
read_menu_choice confirm_choice '请选择: ' 0 1
if [[ "$confirm_choice" == "0" ]]; then
    echo "已取消，安全退出。"
    exit 0
fi

echo
cd "$PROJECT_ROOT"
exec "${COMMAND[@]}"
