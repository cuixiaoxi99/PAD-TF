#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT_DIR/logs/time_matched.pid"
LOG_FILE="$ROOT_DIR/logs/time_matched.log"

mkdir -p "$ROOT_DIR/logs"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Time-matched runner already active: pid=$(cat "$PID_FILE")"
  exit 0
fi

nohup bash "$ROOT_DIR/scripts/run_time_matched.sh" >>"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"
echo "Time-matched runner queued: pid=$!"
