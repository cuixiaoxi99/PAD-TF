#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT_DIR/logs/paper_matrix.pid"
OUT_FILE="$ROOT_DIR/logs/paper_matrix.out"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "state=running pid=$(cat "$PID_FILE")"
else
  echo "state=stopped"
fi
if [[ -f "$ROOT_DIR/logs/watchdog.pid" ]] && kill -0 "$(cat "$ROOT_DIR/logs/watchdog.pid")" 2>/dev/null; then
  echo "watchdog=running pid=$(cat "$ROOT_DIR/logs/watchdog.pid") interval=600s"
else
  echo "watchdog=stopped"
fi
count_json() {
  local directory="$1"
  if [[ -d "$directory" ]]; then
    find "$directory" -name '*.json' | wc -l
  else
    echo 0
  fi
}

main_runs="$(count_json "$ROOT_DIR/results/runs")"
grid_runs="$(count_json "$ROOT_DIR/results/calibration/grid")"
policy_runs="$(count_json "$ROOT_DIR/results/analysis/policies")"
control_runs="$(count_json "$ROOT_DIR/results/analysis/controls")"
echo "completed_main_runs=${main_runs}/240"
echo "completed_grid_runs=${grid_runs}/336"
echo "completed_policy_runs=${policy_runs}/144"
echo "completed_control_runs=${control_runs}/4"
if [[ -f "$ROOT_DIR/results/calibration/lopo.json" ]]; then
  echo "lopo_calibration=ready"
else
  echo "lopo_calibration=pending"
fi
if [[ -f "$ROOT_DIR/results/paper_matrix.complete" ]]; then
  echo "paper_matrix_complete=yes"
else
  echo "paper_matrix_complete=no"
fi
if [[ -f "$OUT_FILE" ]]; then
  tail -n 20 "$OUT_FILE"
fi
