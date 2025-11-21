#!/bin/bash

# This script deletes/copies run_setting_*.py and fp.json files across directories.
# NEW FEATURE:
# - Operates on **sub-subdirectories only** (./folder/*)
# - If no folder paths are specified, all subdirectories in the current folder are used
#   and the operation is applied to their subdirectories.
# - If folder paths ARE provided, only the sub-subdirectories of those folders are modified.

DO_DELETE=false
DO_COPY=false
DO_FP_COPY=false
DO_FP_DELETE=false

usage() {
    echo "Usage: $0 [options] [folders...]"
    echo ""
    echo "Options:"
    echo "  --delete,     -d        Delete run_setting_*.py"
    echo "  --copy,       -c        Copy run_setting_*.py"
    echo "  --fp,         -f        Copy fp.json"
    echo "  --fp-delete,  -F        Delete fp.json"
    echo ""
    echo "If folders are provided, only their sub-subdirectories are processed."
    echo "If no folders are given, operates on subdirectories of current folder and their subdirectories."
}

# ------------------------------------------
# Parse command-line arguments (flags only)
# ------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --delete|-d) DO_DELETE=true; shift ;;
        --copy|-c) DO_COPY=true; shift ;;
        --fp|-f) DO_FP_COPY=true; shift ;;
        --fp-delete|-F) DO_FP_DELETE=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) break ;;  # Stop parsing when first non-flag appears
    esac
done

# Remaining arguments are folder paths (if any)
TARGET_FOLDERS=("$@")

# ------------------------------------------
# If no flags → enable all operations
# ------------------------------------------
if ! $DO_DELETE && ! $DO_COPY && ! $DO_FP_COPY && ! $DO_FP_DELETE; then
    DO_DELETE=true
    DO_COPY=true
    DO_FP_COPY=true
    DO_FP_DELETE=true
fi

# ------------------------------------------
# Validate run_settings folder for copy
# ------------------------------------------
if ( $DO_COPY || $DO_FP_COPY ) && [ ! -d "../run_settings" ]; then
    echo "Error: ../run_settings folder missing."
    exit 1
fi

# ------------------------------------------
# Determine sub-subdirectories to operate on
# ------------------------------------------
subdirs=()

if [ ${#TARGET_FOLDERS[@]} -gt 0 ]; then
    echo "Using explicitly provided folders:"
    for folder in "${TARGET_FOLDERS[@]}"; do
        echo " - $folder"
        if [ ! -d "$folder" ]; then
            echo "Error: $folder is not a directory."
            exit 1
        fi
        # Add all subdirectories of this folder (sub-subdirectories)
        mapfile -t subs < <(find "$folder" -mindepth 1 -maxdepth 1 -type d)
        subdirs+=("${subs[@]}")
    done
else
    echo "No folder arguments provided. Operating on sub-subdirectories of all subdirectories in current directory."
    for folder in */; do
        folder="${folder%/}"
        [ -d "$folder" ] || continue
        mapfile -t subs < <(find "$folder" -mindepth 1 -maxdepth 1 -type d)
        subdirs+=("${subs[@]}")
    done
fi

if [ ${#subdirs[@]} -eq 0 ]; then
    echo "No target sub-subdirectories found."
    exit 1
fi

echo "Target sub-subdirectories:"
printf ' - %s\n' "${subdirs[@]}"

# ------------------------------------------
# Step 1: Delete run_setting_*.py
# ------------------------------------------
if $DO_DELETE; then
    echo "Deleting run_setting_*.py..."
    for subdir in "${subdirs[@]}"; do
        find "$subdir" -maxdepth 1 -name "run_setting_*.py" -delete
        echo "Cleaned $subdir"
    done
fi

# ------------------------------------------
# Step 2: Delete fp.json
# ------------------------------------------
if $DO_FP_DELETE; then
    echo "Deleting fp.json..."
    for subdir in "${subdirs[@]}"; do
        find "$subdir" -maxdepth 1 -name "fp.json" -delete
        echo "Removed fp.json in $subdir"
    done
fi

# ------------------------------------------
# Step 3: Copy run_setting_*.py
# ------------------------------------------
if $DO_COPY; then
    echo "Copying run_setting_*.py..."
    shopt -s nullglob
    files=( ../run_settings/run_setting_*.py )
    shopt -u nullglob

    if [ ${#files[@]} -eq 0 ]; then
        echo "Error: No run_setting_*.py files found in ../../run_settings/"
        exit 1
    fi

    for subdir in "${subdirs[@]}"; do
        cp "${files[@]}" "$subdir/"
        echo "Copied run_setting_*.py to $subdir"
    done
fi

# ------------------------------------------
# Step 4: Copy fp.json
# ------------------------------------------
if $DO_FP_COPY; then
    FP_SOURCE="../run_settings/fp.json"

    if [ ! -f "$FP_SOURCE" ]; then
        echo "Warning: fp.json not found in ../run_settings/ — skipping copy."
    else
        echo "Copying fp.json..."
        for subdir in "${subdirs[@]}"; do
            cp "$FP_SOURCE" "$subdir/"
            echo "Copied fp.json to $subdir"
        done
    fi
fi

echo "Operation completed!"
