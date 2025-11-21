#!/bin/bash

# ------------------------------------------------------------
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
# ------------------------------------------------------------
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

# ---------- 3. Helper: run one folder ----------
run_folder() {
    local dir="$1"
    echo "=== Processing folder: $dir ==="

    # Check required data files
    missing_data=()
    [[ -f "$dir/promise.yml" ]] || missing_data+=("promise.yml")
    
    if [[ "$RUN_PLOTTING" == "true" && "$RUN_EXPERIMENTS" != "true" ]]; then
        [[ -f "$dir/prec_setting_1.json" ]] || missing_data+=("prec_setting_1.json")
    fi

    if (( ${#missing_data[@]} > 0 )); then
        echo "  [Missing required files] ${missing_data[*]}"
        echo "  Skipping $dir"
        echo
        return 1
    fi

    # Find all run_setting_*.py scripts, sorted numerically
    local scripts=($(find "$dir" -maxdepth 1 -name "run_setting_*.py" 2>/dev/null | sort -V))
    if (( ${#scripts[@]} == 0 )); then
        echo "  [No run_setting_*.py files found]"
        echo "  Skipping $dir"
        echo
        return 1
    fi

    # Run each script INSIDE the folder using basename to avoid duplicated paths
    for script in "${scripts[@]}"; do
        script_name=$(basename "$script")
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

# Export functions/vars for parallel
export -f normalize_bool run_folder
export RUN_EXPERIMENTS RUN_PLOTTING

# ---------- 4. Determine sub-subdirectories ----------
echo "=========================================="
echo "Run experiments : $RUN_EXPERIMENTS"
echo "Run plotting    : $RUN_PLOTTING"
echo "Parallel mode   : $PARALLEL"

# Collect sub-subdirectories
TARGET_FOLDERS=()
if (( ${#TARGET_TOP_FOLDERS[@]} == 0 )); then
    # No top-level folders specified → all subdirectories in current dir
    for folder in */; do
        folder="${folder%/}"
        [ -d "$folder" ] || continue
        mapfile -t subs < <(find "$folder" -mindepth 1 -maxdepth 1 -type d)
        TARGET_FOLDERS+=("${subs[@]}")
    done
else
    # Use provided top-level folders → collect their subdirectories
    for folder in "${TARGET_TOP_FOLDERS[@]}"; do
        [[ -d "$folder" ]] || { echo "Warning: '$folder' not found. Skipping."; continue; }
        mapfile -t subs < <(find "$folder" -mindepth 1 -maxdepth 1 -type d)
        TARGET_FOLDERS+=("${subs[@]}")
    done
fi

if (( ${#TARGET_FOLDERS[@]} == 0 )); then
    echo "No sub-subdirectories found. Exiting."
    exit 1
fi

echo "Sub-subdirectories to process:"
printf ' - %s\n' "${TARGET_FOLDERS[@]}"
echo "=========================================="

# ---------- 5. Run folders ----------
if [[ "$PARALLEL" == "true" ]]; then
    if command -v parallel >/dev/null 2>&1; then
        echo "Running ${#TARGET_FOLDERS[@]} folders in parallel (max 4 jobs)..."
        printf '%s\n' "${TARGET_FOLDERS[@]}" | parallel -j 4 run_folder {}
    else
        echo "GNU parallel not found. Falling back to sequential execution."
        for dir in "${TARGET_FOLDERS[@]}"; do
            run_folder "$dir"
        done
    fi
else
    echo "Running ${#TARGET_FOLDERS[@]} folders sequentially..."
    for dir in "${TARGET_FOLDERS[@]}"; do
        run_folder "$dir"
    done
fi

echo "=========================================="
echo "All done!"
