#!/bin/bash
# Usage:
#   ./broadcast_rename_debug.sh [-r] [-b] [-x]
#   ./broadcast_rename_debug.sh [--remove] [--broadcast] [--execute]
#
# Actions:
#   -r, --remove       Remove rename_debug.sh from each subdirectory
#   -b, --broadcast    Copy run_settings/rename_debug.sh into each subdirectory
#   -x, --execute      Run rename_debug.sh inside each subdirectory (in parallel)
#                      Only runs in subdirectories containing promise.yml
# Short options can be bundled: e.g., -rbx

REMOVE=false
BROADCAST=false
EXECUTE=false

# Parse parameters (support bundled short options)
for arg in "$@"; do
    if [[ "$arg" == --* ]]; then
        # Long options
        case "$arg" in
            --remove) REMOVE=true ;;
            --broadcast) BROADCAST=true ;;
            --execute) EXECUTE=true ;;
            *) echo "Unknown option: $arg"; exit 1 ;;
        esac
    elif [[ "$arg" == -* ]]; then
        # Short options (possibly bundled)
        for (( i=1; i<${#arg}; i++ )); do
            c="${arg:i:1}"
            case "$c" in
                r) REMOVE=true ;;
                b) BROADCAST=true ;;
                x) EXECUTE=true ;;
                *) echo "Unknown option: -$c"; exit 1 ;;
            esac
        done
    else
        echo "Unknown argument: $arg"
        exit 1
    fi
done

# Ensure at least one action is selected
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

echo "Processing subdirectories..."

# Keep track of background jobs for parallel execution
PIDS=()

for d in */ ; do
    [ -d "$d" ] || continue

    # Skip the parent folder
    if [ "$d" = "run_settings/" ]; then
        continue
    fi

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

    # Step 3 — Execute in parallel (only if promise.yml exists)
    if $EXECUTE; then
        if [ -f "$d/rename_debug.sh" ]; then
            if [ -f "$d/promise.yml" ]; then
                chmod +x "$d/rename_debug.sh"
                (
                    cd "$d" || exit
                    ./rename_debug.sh
                ) &
                PIDS+=($!)
                echo "Started execution in parallel for: $d"
            else
                echo "Skipping $d: promise.yml not found"
            fi
        else
            echo "Skipping $d: no rename_debug.sh to execute"
        fi
    fi

done

# Wait for all parallel jobs to finish
if $EXECUTE; then
    echo "Waiting for all parallel executions to finish..."
    wait "${PIDS[@]}"
    echo "All executions completed."
fi

echo "Done."
