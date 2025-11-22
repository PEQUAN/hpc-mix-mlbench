#!/bin/bash

# -------------------------------------------------------------------------------------------
# Usage:
#   ./run_benchmarks.sh <run_exp> <run_plot> [folder1 folder2 ...] [--parallel]
#
#   • run_exp: 1|true|y → run experiments
#   • run_plot: 1|true|y → run plots
#   • folders: optional, if none → all valid sub-subdirectories of current folder
#   • --parallel: (optional) run sub-subdirectories in parallel (requires GNU parallel)
#
#   Each sub-subdirectory to be operated must contain:
#     - run_setting_*.py (any number, run in numerical order)
#     - promise.yml
#     - prec_setting_1.json (only required if plotting and experiments are not run)
# -------------------------------------------------------------------------------------------
#
# Author: Xinye Chen (xinyechenai@gmail.com)
# Last Updated: 2025-11-21

# ---------- 1. Parse arguments ----------
RUN_EXPERIMENTS=${1:-true}
RUN_PLOTTING=${2:-true}
shift 2

PARALLEL=false
TARGET_TOP_FOLDERS=()
for arg in "$@"; do
    if [[ "$arg" == "--parallel" ]]; then
        PARALLEL=true
    else
        TARGET_TOP_FOLDERS+=("$arg")
    fi
done

# ---------- 2. Normalize booleans ----------
normalize_bool() {
    case "$1" in
        1|true|True|TRUE|y|Y|yes|Yes|YES|t|T) echo "true" ;;
        0|false|False|FALSE|n|N|no|No|NO|f|F) echo "false" ;;
        *) echo "true" ;;
    esac
}
RUN_EXPERIMENTS=$(normalize_bool "$RUN_EXPERIMENTS")
RUN_PLOTTING=$(normalize_bool "$RUN_PLOTTING")

# ---------- 3. Check for at least one matching plot pair ----------
check_plot_pairs() {
    local dir="$1"

    shopt -s nullglob
    local prec_files=("$dir"/prec_setting_*.json)
    local runtime_files=("$dir"/runtimes*.csv)
    shopt -u nullglob

    local matched_pairs=()
    for prec in "${prec_files[@]}"; do
        local base=$(basename "$prec")
        if [[ $base =~ prec_setting_([0-9]+)\.json ]]; then
            local i="${BASH_REMATCH[1]}"
            if [[ -f "$dir/runtimes${i}.csv" ]]; then
                matched_pairs+=("$i")
            fi
        fi
    done

    if (( ${#matched_pairs[@]} > 0 )); then
        return 0
    else
        return 1
    fi
}

# ---------- 4. Run one folder ----------
run_folder() {
    local dir="$1"
    echo "=== Processing folder: $dir ==="

    missing_data=()
    [[ -f "$dir/promise.yml" ]] || missing_data+=("promise.yml")

    if [[ "$RUN_PLOTTING" == "true" ]]; then
        if ! check_plot_pairs "$dir"; then
            missing_data+=("prec_setting_*.json + runtimes*.csv (matching index required)")
        fi
    fi

    if (( ${#missing_data[@]} > 0 )); then
        echo "  [Missing required files] ${missing_data[*]}"
        echo "  Skipping $dir"
        echo
        return 1
    fi

    shopt -s nullglob
    scripts=("$dir"/run_setting_*.py)
    shopt -u nullglob

    if (( ${#scripts[@]} == 0 )); then
        echo "  [No run_setting_*.py files found]"
        echo "  Skipping $dir"
        echo
        return 1
    fi

    IFS=$'\n' scripts=($(printf "%s\n" "${scripts[@]}" | sort -V))

    for script in "${scripts[@]}"; do
        local script_name=$(basename "$script")
        echo "  → Running: $script_name"
        (
            cd "$dir" || exit
            python3 "$script_name" "$RUN_EXPERIMENTS" "$RUN_PLOTTING"
        )
        if (( $? != 0 )); then
            echo "  [Failed] $script_name"
        else
            echo "  [Success] $script_name"
        fi
    done
    echo
}

export -f run_folder normalize_bool check_plot_pairs
export RUN_EXPERIMENTS RUN_PLOTTING

# ---------- 5. Collect sub-subdirectories ----------
TARGET_FOLDERS=()

if (( ${#TARGET_TOP_FOLDERS[@]} == 0 )); then
    for folder in */; do
        folder="${folder%/}"
        [ -d "$folder" ] || continue
        mapfile -t subs < <(find "$folder" -mindepth 1 -maxdepth 1 -type d)
        for d in "${subs[@]}"; do
            TARGET_FOLDERS+=( "$(realpath "$d")" )
        done
    done
else
    for folder in "${TARGET_TOP_FOLDERS[@]}"; do
        [[ -d "$folder" ]] || { echo "Warning: '$folder' not found. Skipping."; continue; }
        mapfile -t subs < <(find "$folder" -mindepth 1 -maxdepth 1 -type d)
        for d in "${subs[@]}"; do
            TARGET_FOLDERS+=( "$(realpath "$d")" )
        done
    done
fi

if (( ${#TARGET_FOLDERS[@]} == 0 )); then
    echo "No sub-subdirectories found. Exiting."
    exit 1
fi

echo "=========================================="
echo "Run experiments : $RUN_EXPERIMENTS"
echo "Run plotting    : $RUN_PLOTTING"
echo "Parallel mode   : $PARALLEL"
echo "Sub-subdirectories to process:"
printf ' - %s\n' "${TARGET_FOLDERS[@]}"
echo "=========================================="

# ---------- 6. Run folders ----------
if [[ "$PARALLEL" == "true" ]] && command -v parallel >/dev/null 2>&1; then
    printf '%s\n' "${TARGET_FOLDERS[@]}" | parallel -j 4 run_folder {}
else
    for dir in "${TARGET_FOLDERS[@]}"; do run_folder "$dir"; done
fi

echo "=========================================="
echo "All done!"
