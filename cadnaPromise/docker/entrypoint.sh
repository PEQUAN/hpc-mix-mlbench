#!/bin/bash
set -e

if ! command -v g++-13 >/dev/null 2>&1; then
    echo "[entrypoint] ERROR: g++-13 not found in container"
    exit 1
fi

export CC=gcc-13
export CXX=g++-13

echo "[entrypoint] Using CC=$CC ($(command -v $CC))"
echo "[entrypoint] Using CXX=$CXX ($(command -v $CXX))"

if command -v activate-promise >/dev/null 2>&1; then
    activate-promise --CC="$CC" --CXX="$CXX"
else
    echo "[entrypoint] activate-promise not found"
fi

exec "$@"
