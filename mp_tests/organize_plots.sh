#!/bin/bash

# ------------------------------------------------------------
# Usage:
#   ./organize_plots.sh [folder1 folder2 ...]
#
#   • If no folders specified → process ALL folders containing JPG files
#   • If folders specified → process ONLY those
#
#   Moves and renames:
#     precision<i>_with_runtime.jpg → precision<i>_<folder>_runtime.jpg
#     for all i that exist
#   Into a top-level 'plots/' directory (creates if missing).
# ------------------------------------------------------------

TARGET_FOLDERS=("$@")

# ---------- Helper: process one folder ----------
process_folder() {
    local dir="$1"
    local folder_name=$(basename "$dir")
    local plots_dir="./plots"

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
if (( ${#TARGET_FOLDERS[@]} == 0 )); then
    echo "Target folders: ALL with matching JPG files"
else
    echo "Target folders: ${TARGET_FOLDERS[*]}"
fi
echo "Destination: plots/"
echo "=========================================="

# ---------- Run folders ----------
if (( ${#TARGET_FOLDERS[@]} == 0 )); then
    found_any=false

    while IFS= read -r file; do
        dir=$(dirname "$file")
        found_any=true
        process_folder "$dir"
    done < <(find . -maxdepth 2 -type f -regex ".*/precision[0-9]+_with_runtime\.jpg")

    $found_any || echo "Warning: No folders with precision<i> files found."

else
    for folder in "${TARGET_FOLDERS[@]}"; do
        [[ -d "$folder" ]] || { echo "Error: '$folder' is not a directory."; continue; }
        process_folder "$folder"
    done
fi

echo "=========================================="
echo "All done! Check 'plots/' for organized files."
echo "=========================================="