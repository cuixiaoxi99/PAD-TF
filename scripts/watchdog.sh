#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT_DIR/logs/paper_matrix.pid"
HISTORY_FILE="$ROOT_DIR/logs/progress_history.log"
LOCK_FILE="$ROOT_DIR/logs/watchdog.lock"

mkdir -p "$ROOT_DIR/logs"
exec 9>"$LOCK_FILE"
flock -n 9 || exit 0

{
  echo "===== $(date --iso-8601=seconds) ====="
  bash "$ROOT_DIR/scripts/status.sh"
} >>"$HISTORY_FILE" 2>&1

if [[ -f "$ROOT_DIR/results/paper_matrix.complete" ]]; then
  exit 0
fi

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  exit 0
fi

rm -f "$PID_FILE"
bash "$ROOT_DIR/scripts/launch_all.sh" >>"$HISTORY_FILE" 2>&1
