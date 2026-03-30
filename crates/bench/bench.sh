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

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "${1:-}" == "down" ]]; then
    docker compose down --remove-orphans
    exit 0
fi

# Pass all flags to compare.sh inside the runner container
mkdir -p results

echo "==> Building images..."
docker compose build

echo "==> Starting services..."
docker compose up -d mock crabllm bifrost litellm

echo "==> Running benchmarks..."
docker compose run --rm runner ./compare.sh "$@"

echo "==> Tearing down..."
docker compose down
