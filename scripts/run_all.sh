#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PAIRS=(P1 P2 P3 P4 P5 P6 P7 P8)
BASELINES=(Direct Fixed-FT Retrain FlashST-Full Random-FT CoreSet-FT Proxy-FT ELFS-FT BADGE-FT UHerding-FT)

cd "$ROOT_DIR"
mkdir -p results/snapshots
rm -f results/paper_matrix.complete

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

run_baselines() {
  local seed="$1"
  local pair
  for pair in "${PAIRS[@]}"; do
    run_step "$PYTHON_BIN" -m padtf.cli run-pair \
      --pair "$pair" --seed "$seed" --methods "${BASELINES[@]}"
  done
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

# Fastest defensible first result: all P1--P8 methods for seed 2024, followed by
# a bootstrap-calibrated PAD-TF snapshot. This snapshot is explicitly provisional.
run_baselines 2024
run_pad 2024
snapshot_tables seed2024_bootstrap

# Build the declared full CoreSet budget grid once, fit held-out LOPO response
# constants, and replace provisional PAD-TF outputs for seed 2024.
for pair in "${PAIRS[@]}"; do
  run_step "$PYTHON_BIN" -m padtf.cli run-grid --pair "$pair" --seed 2024
done
run_step "$PYTHON_BIN" -m padtf.cli fit-calibration
run_pad 2024 force
snapshot_tables seed2024_lopo

# Extend the locked LOPO protocol to the remaining independent seeds and save a
# complete main-table snapshot after each seed.
for seed in 2025 2026; do
  run_baselines "$seed"
  run_pad "$seed"
  snapshot_tables "through_seed${seed}"
done

# Execute the declared P1--P6 ablations and cost-weight policies. Matching
# decisions reuse main PAD-TF outputs instead of retraining.
for policy in dynamic_only static_only fixed_b no_cost lambda_0.1 lambda_0.2 lambda_0.5 lambda_1.0; do
  for seed in 2024 2025 2026; do
    for pair in P1 P2 P3 P4 P5 P6; do
      run_step "$PYTHON_BIN" -m padtf.cli run-policy --pair "$pair" --seed "$seed" --policy "$policy"
    done
  done
done

for seed in 2024 2025 2026; do
  run_step "$PYTHON_BIN" -m padtf.cli run-control --control zero_shift --seed "$seed"
done
run_step "$PYTHON_BIN" -m padtf.cli run-control --control negative_transfer --seed 2024
snapshot_tables complete
touch results/paper_matrix.complete
