#!/usr/bin/env bash
set -euo pipefail

# Competitive benchmark: crabllm vs Bifrost vs LiteLLM
# Everything runs inside Docker — no local dependencies beyond Docker itself.
#
# Usage:
#   ./bench.sh                                  # run all groups
#   ./bench.sh --group overhead                 # run specific group
#   ./bench.sh --group overhead --duration 5    # quick smoke test
#   ./bench.sh --rps "100 500"                  # custom RPS levels
#   ./bench.sh down                             # tear down containers

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "error: bench requires Linux (Docker images need Linux ELF binaries)" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

if [[ "${1:-}" == "down" ]]; then
    docker compose down --remove-orphans
    exit 0
fi

# Stage pre-built binaries for Docker
BIN_DIR="$REPO_ROOT/target/prod"
for bin in crabllm crabllm-bench; do
    if [[ ! -f "$BIN_DIR/$bin" ]]; then
        echo "error: $BIN_DIR/$bin not found" >&2
        echo "  run: cargo build --profile prod -p crabllm -p crabllm-bench" >&2
        exit 1
    fi
done

mkdir -p bin
cp "$BIN_DIR/crabllm" "$BIN_DIR/crabllm-bench" bin/

mkdir -p results

echo "==> Building images..."
docker compose build

echo "==> Starting services..."
docker compose up -d mock crabllm bifrost litellm

echo "==> Running benchmarks..."
docker compose run --rm runner ./compare.sh "$@"

echo "==> Tearing down..."
docker compose down
