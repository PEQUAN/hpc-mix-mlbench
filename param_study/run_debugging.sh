#!/bin/bash
# Usage:
#   ./broadcast_rename_debug.sh [-r] [-b] [-x] [-p] [folders...]
#   ./broadcast_rename_debug.sh [--remove] [--broadcast] [--execute] [--parallel] [folders...]
#
# Short options can be bundled: -rbxp
# If folders are provided, only those directories are processed.
# If no folders are provided, operates on **sub-subdirectories only** (./*/*).

REMOVE=false
BROADCAST=false
EXECUTE=false
PARALLEL=false

# ---------------------------------------
# Parse parameters (support bundled short options)
# ---------------------------------------
POSITIONAL=()
for arg in "$@"; do
    if [[ "$arg" == --* ]]; then
        case "$arg" in
            --remove) REMOVE=true ;;
            --broadcast) BROADCAST=true ;;
            --execute) EXECUTE=true ;;
            --parallel) PARALLEL=true ;;
            *) echo "Unknown option: $arg"; exit 1 ;;
        esac
    elif [[ "$arg" == -* ]]; then
        for (( i=1; i<${#arg}; i++ )); do
            c="${arg:i:1}"
            case "$c" in
                r) REMOVE=true ;;
                b) BROADCAST=true ;;
                x) EXECUTE=true ;;
                p) PARALLEL=true ;;
                *) echo "Unknown option: -$c"; exit 1 ;;
            esac
        done
    else
        POSITIONAL+=("$arg")
    fi
done

# Remaining arguments are folder paths (if any)
TARGET_DIRS=("${POSITIONAL[@]}")

# ---------------------------------------
# Ensure at least one action is selected
# ---------------------------------------
if ! $REMOVE && ! $BROADCAST && ! $EXECUTE; then
    echo "No action selected. Choose -r, -b, -x or any combination."
    exit 1
fi

# Path to parent script for broadcasting
PARENT_SCRIPT="../run_settings/rename_debug.sh"
if $BROADCAST && [ ! -f "$PARENT_SCRIPT" ]; then
    echo "Error: $PARENT_SCRIPT not found. Cannot broadcast."
    exit 1
fi

# ---------------------------------------
# Determine target directories
# ---------------------------------------
if [ ${#TARGET_DIRS[@]} -gt 0 ]; then
    echo "Using explicitly provided target directories:"
    for t in "${TARGET_DIRS[@]}"; do
        echo " - $t"
        if [ ! -d "$t" ]; then
            echo "Error: $t is not a directory."
            exit 1
        fi
    done
    SUBDIRS=("${TARGET_DIRS[@]}")
else
    echo "No folders provided → using **sub-subdirectories only**."

    # Only sub-subdirectories (depth = 2)
    mapfile -t SUBDIRS < <(
        find . -mindepth 2 -maxdepth 2 -type d ! -path "./run_settings" ! -path "."
    )
fi

if [ ${#SUBDIRS[@]} -eq 0 ]; then
    echo "No target directories found."
    exit 1
fi

echo "Target directories:"
printf ' - %s\n' "${SUBDIRS[@]}"

# ---------------------------------------
# Keep track of background jobs
# ---------------------------------------
PIDS=()

shopt -s nullglob
for d in "${SUBDIRS[@]}"; do
    [ -d "$d" ] || continue

    # Step 1 — Remove
    if $REMOVE; then
        if [ -f "$d/rename_debug.sh" ]; then
            rm -f "$d/rename_debug.sh"
            echo "Removed $d/rename_debug.sh"
        fi
    fi

    # Step 2 — Broadcast
    if $BROADCAST; then
        cp "$PARENT_SCRIPT" "$d"
        echo "Broadcasted rename_debug.sh to $d"
    fi

    # Step 3 — Execute
    if $EXECUTE; then
        if [ -f "$d/rename_debug.sh" ] && [ -f "$d/promise.yml" ]; then
            chmod +x "$d/rename_debug.sh"

            if $PARALLEL; then
                (
                    cd "$d" || exit
                    ./rename_debug.sh
                ) &
                PIDS+=($!)
                echo "Started execution in parallel for: $d"
            else
                echo "Executing sequentially in: $d"
                (
                    cd "$d" || exit
                    ./rename_debug.sh
                )
            fi
        else
            echo "Skipping $d: missing rename_debug.sh or promise.yml"
        fi
    fi
done
shopt -u nullglob

# Wait for all parallel jobs
if $EXECUTE && $PARALLEL && [ ${#PIDS[@]} -gt 0 ]; then
    echo "Waiting for all parallel executions to finish..."
    wait "${PIDS[@]}"
    echo "All executions completed."
fi

echo "Done."
