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
#!/bin/bash
# Usage:
#   ./broadcast_rename_debug.sh [-r] [-b] [-x] [-p]
#   ./broadcast_rename_debug.sh [--remove] [--broadcast] [--execute] [--parallel]
#
# Actions:
#   -r, --remove       Remove rename_debug.sh from each subdirectory
#   -b, --broadcast    Copy run_settings/rename_debug.sh into each subdirectory
#   -x, --execute      Run rename_debug.sh inside each subdirectory
#                      Only runs in subdirectories containing promise.yml
#   -p, --parallel     Execute in parallel (default is sequential)
#
# Short options can be bundled: e.g., -rbxp
# ------------------------------------------------------------
# Author: Xinye Chen (xinyechenai@gmail.com)
# Last Updated: November 18, 2025
# ---------------------------------------

REMOVE=false
BROADCAST=false
EXECUTE=false
PARALLEL=false

# ---------------------------------------
# Parse parameters (support bundled short options)
# ---------------------------------------
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

PIDS=()

for d in */ ; do
    [ -d "$d" ] || continue
    [ "$d" = "run_settings/" ] && continue

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

# Wait for parallel jobs
if $EXECUTE && $PARALLEL && [ ${#PIDS[@]} -gt 0 ]; then
    echo "Waiting for all parallel executions to finish..."
    wait "${PIDS[@]}"
    echo "All executions completed."
fi

echo "Done."
