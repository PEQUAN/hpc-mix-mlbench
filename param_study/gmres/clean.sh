#!/bin/bash
set -euo pipefail

# Usage:
# ./script.sh <remove_csv> <remove_debug_folders> <remove_prec> <remove_runtimes>
# Parameters:
#   1 = enable deletion
#   0 = disable deletion

REMOVE_CSV=$1
REMOVE_DEBUG=$2
REMOVE_PREC=$3
REMOVE_RUNTIMES=$4

# Helper: return true if parameter is 1/yes/true
is_enabled() {
    [[ "$1" == "1" || "$1" == "yes" || "$1" == "true" ]]
}

### 1. Remove all *.csv files (except in root)
if is_enabled "$REMOVE_CSV"; then
    find . -mindepth 2 -type f -name "*.csv" -print -delete
    echo "All .csv files in subdirectories deleted."
else
    echo "Skipping general .csv file deletion."
fi

### 2. Remove debug-related folders
if is_enabled "$REMOVE_DEBUG"; then
    find . -mindepth 1 -type d \( -name "compileErrors" -o -name "*debug*" \) -print -exec rm -rf {} +
    echo "Folders named 'compileErrors' or containing 'debug' deleted."
else
    echo "Skipping debug folder deletion."
fi

### 3. Remove prec_setting_{k}.json
if is_enabled "$REMOVE_PREC"; then
    find . -type f -regex ".*/prec_setting_[0-9][0-9]*\.json" -print -delete
    echo "All prec_setting_{k}.json files deleted."
else
    echo "Skipping prec_setting_{k}.json deletion."
fi

### 4. Remove runtimes{k}.csv
if is_enabled "$REMOVE_RUNTIMES"; then
    # Cross-platform: matches runtimes followed by numbers, in all subdirectories
    find . -type f -name "runtimes*.csv" -print -delete
    echo "All runtimes{k}.csv files deleted."
else
    echo "Skipping runtimes{k}.csv deletion."
fi
