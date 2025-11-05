#!/bin/bash

# ------------------------------------------------------------
# Usage:
#   ./organize_plots.sh [folder1 folder2 ...]
#
#   • If no folders specified → process ALL folders containing the PNG files
#   • If folders specified → process ONLY those (must contain the PNG files)
#
#   Moves and renames:
#     precision1_with_runtime.jpg → precision1_[folder_name]_runtime.jpg
#     ... (for 2,3,4)
#   Into a top-level 'plots/' directory (creates if missing).
# ------------------------------------------------------------

# ---------- 1. Parse arguments ----------
shift 0  # No flags, just folders
TARGET_FOLDERS=("$@")

# ---------- 2. Helper: process one folder ----------
process_folder() {
    local dir="$1"
    local folder_name=$(basename "$dir")
    local plots_dir="../param_plots"  # Relative to current dir; adjust if needed

    echo "=== Processing folder: $dir ==="

    # Check if all four PNGs exist
    local missing=()
    for i in {1,2,3,4}; do
        local png="precision${i}_with_runtime.jpg"
        [[ -f "$dir/$png" ]] || missing+=("$png")
    done

    if (( ${#missing[@]} > 0 )); then
        echo "  [Missing files] ${missing[*]}"
        echo "  Skipping $dir"
        echo
        return 1
    fi

    # Create plots dir if missing
    mkdir -p "$plots_dir"

    # Move and rename each file
    for i in {1,2,3,4}; do
        local old_png="precision${i}_with_runtime.jpg"
        local new_name="precision${i}_${folder_name}_runtime.jpg"
        mv "$dir/$old_png" "$plots_dir/$new_name"
        if [ $? -eq 0 ]; then
            echo "  → Moved: $old_png → $new_name"
        else
            echo "  [Failed] to move $old_png"
        fi
    done
    echo
}

# ---------- 3. Main logic ----------
echo "=========================================="
if (( ${#TARGET_FOLDERS[@]} == 0 )); then
    echo "Target folders: ALL with PNG files"
else
    echo "Target folders: ${TARGET_FOLDERS[*]}"
fi
echo "Destination: param_plots/"
echo "=========================================="

# ---------- 4. Run folders ----------
if (( ${#TARGET_FOLDERS[@]} == 0 )); then
    # Find all folders with precision1_with_runtime.jpg (then validate)
    found_any=false
    while IFS= read -r png1; do
        dir=$(dirname "$png1")
        # Quick pre-check for other PNGs
        [[ -f "$dir/precision2_with_runtime.jpg" && \
           -f "$dir/precision3_with_runtime.jpg" && \
           -f "$dir/precision4_with_runtime.jpg" ]] || continue
        found_any=true
        process_folder "$dir"
    done < <(find . -maxdepth 2 -type f -name "precision1_with_runtime.jpg")

    $found_any || echo "Warning: No folders with PNG files found."

else
    for folder in "${TARGET_FOLDERS[@]}"; do
        [[ -d "$folder" ]] || { echo "Error: '$folder' is not a directory."; continue; }
        process_folder "$folder"
    done
fi

echo "=========================================="
echo "All done! Check 'param_plots/' for organized files."