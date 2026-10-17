#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PAIRS=(P1 P2 P3 P4 P5 P6)
SEEDS=(2024)
METHODS=(FlashST-Full Random-FT CoreSet-FT Proxy-FT ELFS-FT BADGE-FT UHerding-FT)

cd "$ROOT_DIR"
mkdir -p logs
rm -f results/time_matched.complete

run_step() {
  local attempt
  for attempt in 1 2 3; do
    if "$@"; then
      return 0
    fi
    echo "step failed (attempt ${attempt}/3): $*" >&2
    sleep 30
  done
  echo "step failed permanently: $*" >&2
  return 1
}

while [[ ! -f results/padtf_main.complete ]]; do
  echo "$(date --iso-8601=seconds) waiting for seed-2024 PAD-TF main results"
  sleep 300
done

for seed in "${SEEDS[@]}"; do
  for pair in "${PAIRS[@]}"; do
    for method in "${METHODS[@]}"; do
      run_step "$PYTHON_BIN" -m padtf.cli run-time-matched --pair "$pair" --seed "$seed" --method "$method"
    done
  done
  run_step "$PYTHON_BIN" -m padtf.cli aggregate-time-matched
done

run_step "$PYTHON_BIN" -m padtf.cli aggregate-time-matched
run_step "$PYTHON_BIN" scripts/plot_time_matched.py
touch results/time_matched.complete
