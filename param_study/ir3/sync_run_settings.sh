#!/bin/bash

# There is a folder named run_settings in the parent directory containing all run_setting_*.py files
# This script will delete all run_setting_*.py files in each subfolder under the current directory
# Then copy all run_setting_*.py files from ../run_settings/ to each subfolder

# Overview:
# This Bash script manages `run_setting_*.py` files across multiple subfolders in the current working directory.
# It performs two main steps:
# 1. Delete Step: Deletes all files matching `run_setting_*.py` in each subfolder under the current directory.
# 2. Copy Step: Copies `run_setting_*.py` files from the parent directory's `run_settings/` folder to each subfolder,
#    ensuring all subfolders use the same configuration files.
#
# The script is useful for automating the synchronization of experiment or run settings files across multiple folders,
# such as in machine learning experiments or batch tasks.
#
# Prerequisites:
# - Run in the current working directory containing multiple subfolders (targets).
# - Parent directory (`..`) must have a `run_settings` folder with `run_setting_*.py` files (for copy step).
# - Uses `find` and `cp` commands (standard on Unix-like systems).
#
# Usage:
# bash sync_run_settings.sh [options]
#
# Options:
#   --delete or -d: Execute Step 1 (delete files). Default: enabled if no options.
#   --copy   or -c: Execute Step 2 (copy files).   Default: enabled if no options.
#
# Examples:
#   # Full run (delete + copy)
#   bash sync_run_settings.sh
#
#   # Delete only
#   bash sync_run_settings.sh --delete
#   # Or short form
#   bash sync_run_settings.sh -d
#
#   # Copy only
#   bash sync_run_settings.sh --copy
#   # Or short form
#   bash sync_run_settings.sh -c
#
#   # Both explicitly (mix long/short forms)
#   bash sync_run_settings.sh --delete -c
#
# Notes:
# - If no options provided, both steps run.
# - Supports mixing long (--delete, --copy) and short (-d, -c) options.
# - Backup files before delete step!
# - Grant execute: chmod +x sync_run_settings.sh
#
# Author: Xinye Chen
# Last Updated: November 16, 2025

DO_DELETE=false
DO_COPY=false 

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --delete|-d)
            DO_DELETE=true
            shift
            ;;
        --copy|-c)
            DO_COPY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--delete|-d] [--copy|-c]"
            echo "  --delete/-d: Execute step 1 (delete run_setting_*.py files from subfolders)"
            echo "  --copy/-c:   Execute step 2 (copy run_setting_*.py files to subfolders)"
            echo "If no options are provided, both steps will be executed."
            exit 1
            ;;
    esac
done

# If no flags provided, enable both
if [ "$DO_DELETE" = false ] && [ "$DO_COPY" = false ]; then
    DO_DELETE=true
    DO_COPY=true
fi

# Check if the run_settings folder exists in the parent directory if copying is enabled
if [ "$DO_COPY" = true ] && [ ! -d "../../run_settings" ]; then
    echo "Error: The run_settings folder does not exist in the parent directory."
    exit 1
fi

# Get all subfolders in the current directory
subdirs=($(find . -maxdepth 1 -type d ! -name .))

# If there are no subfolders, exit
if [ ${#subdirs[@]} -eq 0 ]; then
    echo "No subfolders found in the current directory."
    exit 1
fi

# Step 1: Delete all run_setting_*.py files in each subfolder (if flag enabled)
if [ "$DO_DELETE" = true ]; then
    echo "Deleting run_setting_*.py files from each subfolder..."
    for subdir in "${subdirs[@]}"; do
        if [ -d "$subdir" ]; then
            find "$subdir" -name "run_setting_*.py" -delete
            echo "Cleaned: $subdir"
        fi
    done
    rm -rf plots
else
    echo "Step 1 (delete) skipped."
fi

# Step 2: Copy all run_setting_*.py files from ../run_settings/ to each subfolder (if flag enabled)
if [ "$DO_COPY" = true ]; then
    echo "Copying run_setting_*.py files from ../../run_settings/ to each subfolder..."
    source_files=($(find ../../run_settings -name "run_setting_*.py"))

    if [ ${#source_files[@]} -eq 0 ]; then
        echo "Warning: No run_setting_*.py files found in ../../run_settings/."
        exit 0
    fi

    for source_file in "${source_files[@]}"; do
        filename=$(basename "$source_file")
        for subdir in "${subdirs[@]}"; do
            if [ -d "$subdir" ]; then
                cp "$source_file" "$subdir/$filename"
                echo "Copied $filename to $subdir"
            fi
        done
    done
else
    echo "Step 2 (copy) skipped."
fi

echo "Operation completed!"