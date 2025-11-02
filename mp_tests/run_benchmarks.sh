#!/bin/bash

# ------------------------------------------------------------
# Usage:
#   ./run_plots.sh <run_exp> <run_plot> [folder1 folder2 ...]
#
#   • run_exp: 1|true|y → run experiments
#   • run_plot: 1|true|y → run plots
#   • folders: optional, if none → all valid folders
#
#   Each folder must contain:
#     - plot_1.py to plot_4.py
#     - precision_settings_1.json
#     - promise.yml
# ------------------------------------------------------------

# ---------- 1. Parse arguments ----------
RUN_EXPERIMENTS=${1:-true}
RUN_PLOTTING=${2:-true}
shift 2
TARGET_FOLDERS=("$@")

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
    local abs_dir=$(realpath "$dir")  # full path for safety

    echo "=== Processing folder: $abs_dir ==="

    # Check required files
    local missing=()
    for file in plot_{1,2,3,4}.py precision_settings_1.json promise.yml; do
        [[ -f "$dir/$file" ]] || missing+=("$file")
    done

    if (( ${#missing[@]} > 0 )); then
        echo "  [Missing files] ${missing[*]}"
        echo "  Skipping $dir"
        echo
        return 1
    fi

    # Run each script INSIDE the folder
    for script in plot_{1,2,3,4}.py; do
        echo "  → Running: $script"
        (
            cd "$dir"  # Critical: change to folder
            python3 "$script" "$RUN_EXPERIMENTS" "$RUN_PLOTTING"
        )
        if (( $? != 0 )); then
            echo "  [Failed] $script"
        else
            echo "  [Success] $script"
        fi
    done
    echo
}

# ---------- 4. Main logic ----------
echo "=========================================="
echo "Run experiments : $RUN_EXPERIMENTS"
echo "Run plotting    : $RUN_PLOTTING"
if (( ${#TARGET_FOLDERS[@]} == 0 )); then
    echo "Target folders  : ALL valid folders"
else
    echo "Target folders  : ${TARGET_FOLDERS[*]}"
fi
echo "=========================================="

# ---------- 5. Run folders ----------
if (( ${#TARGET_FOLDERS[@]} == 0 )); then
    # Find all folders with plot_1.py (then validate)
    found_any=false
    while IFS= read -r plot1; do
        dir=$(dirname "$plot1")
        # Quick pre-check
        [[ -f "$dir/precision_settings_1.json" && -f "$dir/promise.yml" ]] || continue
        found_any=true
        run_folder "$dir"
    done < <(find . -maxdepth 2 -type f -name "plot_1.py")

    $found_any || echo "Warning: No complete folder found (missing files)."

else
    for folder in "${TARGET_FOLDERS[@]}"; do
        [[ -d "$folder" ]] || { echo "Error: '$folder' is not a directory."; continue; }
        run_folder "$folder"
    done
fi

echo "=========================================="
echo "All done!"