#!/bin/bash

# ------------------------------------------------------------
# Usage:
#   ./run_benchmarks.sh <run_exp> <run_plot> [folder1 folder2 ...] [--parallel]
#
#   • run_exp: 1|true|y → run experiments
#   • run_plot: 1|true|y → run plots
#   • folders: optional, if none → all valid folders
#   • --parallel: (optional) run folders in parallel (requires GNU parallel)
#
#   Each folder must contain:
#     - run_setting_*.py (any number, run in numerical order)
#     - promise.yml
#     - prec_setting_1.json (only required if plotting and experiments are not run)
# ------------------------------------------------------------
#
# Author: Xinye Chen (xinyechenai@gmail.com)
# Last Updated: November 18, 2025

# ---------- 1. Parse arguments ----------
RUN_EXPERIMENTS=${1:-true}
RUN_PLOTTING=${2:-true}
shift 2

# Check for --parallel flag
PARALLEL=false
TARGET_FOLDERS=()
for arg in "$@"; do
    if [[ "$arg" == "--parallel" ]]; then
        PARALLEL=true
    else
        TARGET_FOLDERS+=("$arg")
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
    local input_dir="$1"
    local dir=$(realpath "$input_dir")  # Ensure absolute path

    echo "=== Processing folder: $dir ==="

    # Check required data files
    missing_data=()
    [[ -f "$dir/promise.yml" ]] || missing_data+=("promise.yml")
    
    # prec_setting_1.json is needed only if plotting AND experiments are NOT run
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

    # Run each script INSIDE the folder
    for script in "${scripts[@]}"; do
        echo "  → Running: $(basename "$script")"
        (
            cd "$dir"
            python3 "$script" "$RUN_EXPERIMENTS" "$RUN_PLOTTING"
        )
        if (( $? != 0 )); then
            echo "  [Failed] $(basename "$script")"
        else
            echo "  [Success] $(basename "$script")"
        fi
    done
    echo
}

# ---------- 4. Export vars for parallel ----------
export -f normalize_bool run_folder
export RUN_EXPERIMENTS RUN_PLOTTING

# ---------- 5. Main logic ----------
echo "=========================================="
echo "Run experiments : $RUN_EXPERIMENTS"
echo "Run plotting    : $RUN_PLOTTING"
echo "Parallel mode   : $PARALLEL"
if (( ${#TARGET_FOLDERS[@]} == 0 )); then
    echo "Target folders  : ALL valid folders"
else
    echo "Target folders  : ${TARGET_FOLDERS[*]}"
fi
echo "=========================================="

# ---------- 6. Run folders ----------
if (( ${#TARGET_FOLDERS[@]} == 0 )); then
    # Collect all valid folders first
    valid_folders=()
    while IFS= read -r script1; do
        dir=$(dirname "$script1")
        dir=$(realpath "$dir")
        
        # Auto-discovery: require promise.yml always
        [[ -f "$dir/promise.yml" ]] || continue

        # prec_setting_1.json only if plotting AND experiments are NOT run
        if [[ "$RUN_PLOTTING" == "true" && "$RUN_EXPERIMENTS" != "true" ]]; then
            [[ -f "$dir/prec_setting_1.json" ]] || continue
        fi
        
        valid_folders+=("$dir")
    done < <(find . -maxdepth 2 -type f -name "run_setting_1.py")

    if (( ${#valid_folders[@]} == 0 )); then
        echo "Warning: No complete folder found (missing files or run_setting_1.py)."
    else
        if [[ "$PARALLEL" == "true" ]]; then
            if command -v parallel >/dev/null 2>&1; then
                echo "Running ${#valid_folders[@]} folders in parallel (max 4 jobs)..."
                printf '%s\n' "${valid_folders[@]}" | parallel -j 4 run_folder {}
            else
                echo "GNU parallel not found. Falling back to sequential execution."
                for dir in "${valid_folders[@]}"; do
                    run_folder "$dir"
                done
            fi
        else
            echo "Running ${#valid_folders[@]} folders sequentially..."
            for dir in "${valid_folders[@]}"; do
                run_folder "$dir"
            done
        fi
    fi

else
    # Filter valid specified folders
    valid_folders=()
    for folder in "${TARGET_FOLDERS[@]}"; do
        [[ -d "$folder" ]] || { echo "Error: '$folder' is not a directory."; continue; }
        [[ -f "$folder/run_setting_1.py" && -f "$folder/promise.yml" ]] || { echo "Error: '$folder' missing required files. Skipping."; continue; }

        if [[ "$RUN_PLOTTING" == "true" && "$RUN_EXPERIMENTS" != "true" && ! -f "$folder/prec_setting_1.json" ]]; then
            echo "Error: '$folder' missing prec_setting_1.json for plotting without experiments. Skipping."
            continue
        fi
        valid_folders+=("$folder")
    done

    if (( ${#valid_folders[@]} == 0 )); then
        echo "No valid folders specified."
    else
        if [[ "$PARALLEL" == "true" ]]; then
            if command -v parallel >/dev/null 2>&1; then
                echo "Running ${#valid_folders[@]} folders in parallel (max 4 jobs)..."
                printf '%s\n' "${valid_folders[@]}" | parallel -j 4 run_folder {}
            else
                echo "GNU parallel not found. Falling back to sequential execution."
                for dir in "${valid_folders[@]}"; do
                    run_folder "$dir"
                done
            fi
        else
            echo "Running ${#valid_folders[@]} folders sequentially..."
            for dir in "${valid_folders[@]}"; do
                run_folder "$dir"
            done
        fi
    fi
fi

echo "=========================================="
echo "All done!"
