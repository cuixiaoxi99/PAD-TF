#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT_DIR/logs/padtf_only.pid"
LOG_FILE="$ROOT_DIR/logs/padtf_only.log"

cd "$ROOT_DIR"
mkdir -p logs

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "PAD-TF-only runner already active: pid=$(cat "$PID_FILE")"
  exit 0
fi

nohup bash "$ROOT_DIR/scripts/run_padtf_only.sh" >>"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"
echo "PAD-TF-only runner started: pid=$(cat "$PID_FILE")"
