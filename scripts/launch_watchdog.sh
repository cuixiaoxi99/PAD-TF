#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT_DIR/logs/watchdog.pid"
OUT_FILE="$ROOT_DIR/logs/watchdog.out"

mkdir -p "$ROOT_DIR/logs"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "watchdog already running as PID $(cat "$PID_FILE")"
  exit 0
fi

nohup bash -c "while true; do bash '$ROOT_DIR/scripts/watchdog.sh'; sleep 600; done" >"$OUT_FILE" 2>&1 </dev/null &
echo $! >"$PID_FILE"
echo "started watchdog as PID $! (interval: 600s)"
