#!/bin/bash

# There is a folder named run_settings in the parent directory containing all run_setting_*.py files
# This script will delete all run_setting_*.py files in each subfolder under the current directory
# Then copy all run_setting_*.py files from ../../run_settings/ to each subfolder

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
# Author: Xinye Chen (xinyechenai@gmail.com)
# Last Updated: November 16, 2025

DO_DELETE=false
DO_COPY=false
DO_FP_COPY=false
DO_FP_DELETE=false

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --delete, -d        Delete run_setting_*.py"
    echo "  --copy,   -c        Copy run_setting_*.py"
    echo "  --fp,     -f        Copy fp.json"
    echo "  --fp-delete, -F     Delete fp.json"
    echo ""
    echo "If no options are given, all operations run."
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --delete|-d)
            DO_DELETE=true
            ;;
        --copy|-c)
            DO_COPY=true
            ;;
        --fp|-f)
            DO_FP_COPY=true
            ;;
        --fp-delete|-F)
            DO_FP_DELETE=true
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
    shift
done

# If no flags → run everything
if ! $DO_DELETE && ! $DO_COPY && ! $DO_FP_COPY && ! $DO_FP_DELETE; then
    DO_DELETE=true
    DO_COPY=true
    DO_FP_COPY=true
    DO_FP_DELETE=true
fi

# Validate run_settings folder for copy actions
if ( $DO_COPY || $DO_FP_COPY ) && [ ! -d "../../run_settings" ]; then
    echo "Error: ../../run_settings folder missing."
    exit 1
fi

# Get subdirectories
subdirs=($(find . -maxdepth 1 -type d ! -name .))

if [ ${#subdirs[@]} -eq 0 ]; then
    echo "No subfolders found."
    exit 1
fi

#############################################
# Step 1: Delete run_setting_*.py
#############################################
if $DO_DELETE; then
    echo "Deleting run_setting_*.py..."
    for subdir in "${subdirs[@]}"; do
        find "$subdir" -maxdepth 1 -name "run_setting_*.py" -delete
        echo "Cleaned $subdir"
    done
fi

#############################################
# Step 2: Delete fp.json
#############################################
if $DO_FP_DELETE; then
    echo "Deleting fp.json..."
    for subdir in "${subdirs[@]}"; do
        find "$subdir" -maxdepth 1 -name "fp.json" -delete
        echo "Removed fp.json in $subdir"
    done
fi

#############################################
# Step 3: Copy run_setting_*.py
#############################################
if $DO_COPY; then
    echo "Copying run_setting_*.py..."
    files=( ../../run_settings/run_setting_*.py )
    for subdir in "${subdirs[@]}"; do
        cp "${files[@]}" "$subdir/"
        echo "Copied run_setting_*.py to $subdir"
    done
fi

#############################################
# Step 4: Copy fp.json
#############################################
if $DO_FP_COPY; then
    FP_SOURCE="../../run_settings/fp.json"
    if [ ! -f "$FP_SOURCE" ]; then
        echo "Warning: fp.json not found in ../../run_settings/"
    else
        echo "Copying fp.json..."
        for subdir in "${subdirs[@]}"; do
            cp "$FP_SOURCE" "$subdir/"
            echo "Copied fp.json to $subdir"
        done
    fi
fi

echo "Operation completed!"