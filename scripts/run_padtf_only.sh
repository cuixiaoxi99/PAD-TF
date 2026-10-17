#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PAIRS=(P1 P2 P3 P4 P5 P6 P7 P8)

cd "$ROOT_DIR"
mkdir -p results/snapshots logs
rm -f results/padtf_only.complete
rm -f results/padtf_main.complete

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

snapshot_tables() {
  local name="$1"
  run_step "$PYTHON_BIN" -m padtf.cli aggregate
  mkdir -p "results/snapshots/$name"
  cp -a results/tables/. "results/snapshots/$name/"
}

run_pad() {
  local seed="$1"
  local force_flag="${2:-}"
  local pair
  for pair in "${PAIRS[@]}"; do
    if [[ "$force_flag" == "force" ]]; then
      run_step "$PYTHON_BIN" -m padtf.cli run --pair "$pair" --seed "$seed" --method PAD-TF --force
    else
      run_step "$PYTHON_BIN" -m padtf.cli run --pair "$pair" --seed "$seed" --method PAD-TF
    fi
  done
}

run_pad 2024
snapshot_tables seed2024_padtf_bootstrap

for pair in "${PAIRS[@]}"; do
  run_step "$PYTHON_BIN" -m padtf.cli run-grid --pair "$pair" --seed 2024
done
run_step "$PYTHON_BIN" -m padtf.cli fit-calibration

run_pad 2024 force
snapshot_tables seed2024_padtf_lopo
touch results/padtf_main.complete

bash scripts/launch_time_matched.sh
while [[ ! -f results/time_matched.complete ]]; do
  if [[ ! -f logs/time_matched.pid ]] || ! kill -0 "$(cat logs/time_matched.pid)" 2>/dev/null; then
    bash scripts/launch_time_matched.sh
  fi
  sleep 300
done
snapshot_tables requested_experiments_complete
touch results/padtf_only.complete
