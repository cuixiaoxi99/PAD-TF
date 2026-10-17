#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT_DIR/logs/paper_matrix.pid"
OUT_FILE="$ROOT_DIR/logs/paper_matrix.out"

mkdir -p "$ROOT_DIR/logs"
if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE")"
  if kill -0 "$existing_pid" 2>/dev/null; then
    echo "paper matrix is already running as PID $existing_pid"
    exit 0
  fi
fi

cd "$ROOT_DIR"
nohup bash scripts/run_all.sh >"$OUT_FILE" 2>&1 &
new_pid=$!
echo "$new_pid" >"$PID_FILE"
echo "started paper matrix as PID $new_pid"
echo "log: $OUT_FILE"
