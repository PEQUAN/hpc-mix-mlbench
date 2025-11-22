#!/bin/bash

# ------------------------------------------------------------
# Usage:
#   ./organize_plots.sh [folder1 folder2 ...]
#
#   • If no folders specified → process ALL sub-subdirectories containing matching JPG files
#   • If folders specified → process ONLY subdirectories of those
#
#   Moves and renames:
#     precision<i>_with_runtime.jpg → precision<i>_<folder>_runtime.jpg
#     for all i that exist
#   Into a top-level 'param_plots/' directory (creates if missing).
# ------------------------------------------------------------
# Author: Xinye Chen (xinyechenai@gmail.com)
# Last Updated: November 18, 2025
# ------------------------------------------------------------

TARGET_TOP_FOLDERS=("$@")

# ---------- Helper: process one folder ----------
process_folder() {
    local dir="$1"
    local folder_name=$(basename "$dir")
    local plots_dir="./param_plots/"

    echo "=== Processing folder: $dir ==="

    # Find all matching files
    mapfile -t files < <(find "$dir" -maxdepth 1 -type f -regex ".*/precision[0-9]+_with_runtime\.jpg")

    if (( ${#files[@]} == 0 )); then
        echo "  [No matching precision<i> files found]"
        echo
        return
    fi

    mkdir -p "$plots_dir"

    for filepath in "${files[@]}"; do
        filename=$(basename "$filepath")

        # Extract index i from precision<i>_with_runtime.jpg
        if [[ $filename =~ precision([0-9]+)_with_runtime\.jpg ]]; then
            i="${BASH_REMATCH[1]}"
            new_name="precision${i}_${folder_name}_runtime.jpg"

            mv "$filepath" "$plots_dir/$new_name"
            if [[ $? -eq 0 ]]; then
                echo "  → Moved: $filename → $new_name"
            else
                echo "  [Failed] to move $filename"
            fi
        fi
    done
    echo
}

# ---------- Main logic ----------
echo "=========================================="

TARGET_FOLDERS=()

if (( ${#TARGET_TOP_FOLDERS[@]} == 0 )); then
    echo "No folders specified → auto-detecting sub-subdirectories with matching JPG files..."
    found_any=false

    for folder in */; do
        folder="${folder%/}"
        [ -d "$folder" ] || continue

        # Collect subdirectories
        mapfile -t subs < <(find "$folder" -mindepth 1 -maxdepth 1 -type d)
        for sub in "${subs[@]}"; do
            # Only include if it contains matching JPG files
            if find "$sub" -maxdepth 1 -type f -regex ".*/precision[0-9]+_with_runtime\.jpg" | grep -q .; then
                TARGET_FOLDERS+=( "$sub" )
                found_any=true
            fi
        done
    done

    $found_any || { echo "Warning: No sub-subdirectories with precision<i> files found."; exit 1; }

else
    # Use provided top-level folders → collect their subdirectories
    for folder in "${TARGET_TOP_FOLDERS[@]}"; do
        [[ -d "$folder" ]] || { echo "Warning: '$folder' not found. Skipping."; continue; }
        mapfile -t subs < <(find "$folder" -mindepth 1 -maxdepth 1 -type d)
        TARGET_FOLDERS+=( "${subs[@]}" )
    done
fi

echo "Sub-subdirectories to process:"
printf ' - %s\n' "${TARGET_FOLDERS[@]}"
echo "Destination: param_plots/"
echo "=========================================="

# ---------- Process folders ----------
for dir in "${TARGET_FOLDERS[@]}"; do
    process_folder "$dir"
done

echo "=========================================="
echo "All done! Check 'param_plots/' for organized files."
echo "=========================================="
