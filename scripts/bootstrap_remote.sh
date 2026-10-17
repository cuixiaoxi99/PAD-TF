#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_NAME="${PADTF_ENV_NAME:-padtf}"
UPSTREAM_DIR="$ROOT_DIR/third_party/FlashST"
DATA_DIR="$ROOT_DIR/data"

if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  "$CONDA_BIN" create -y -n "$ENV_NAME" python=3.10
fi

PYTHON_BIN="${PYTHON_BIN:-$($CONDA_BIN run -n "$ENV_NAME" which python)}"
"$PYTHON_BIN" -m pip install -r "$ROOT_DIR/requirements.txt"

if [[ ! -d "$UPSTREAM_DIR/.git" ]]; then
  mkdir -p "$(dirname "$UPSTREAM_DIR")"
  git clone https://github.com/HKUDS/FlashST.git "$UPSTREAM_DIR"
fi
git -C "$UPSTREAM_DIR" fetch --depth 1 origin 39d4b8e445fde7ed5272151f7e74752d2c84c58e
git -C "$UPSTREAM_DIR" checkout --detach 39d4b8e445fde7ed5272151f7e74752d2c84c58e

mkdir -p "$DATA_DIR" "$ROOT_DIR/checkpoints" "$ROOT_DIR/logs" "$ROOT_DIR/results/runs" "$ROOT_DIR/results/tables"

FLASHST_ZIP="$DATA_DIR/flashst-data.zip"
if [[ ! -f "$DATA_DIR/PEMS04/PEMS04.npz" ]]; then
  if ! wget -c --timeout=30 --tries=3 -O "$FLASHST_ZIP" "https://hf-mirror.com/datasets/bjdwh/FlashST-DATA/resolve/main/data.zip"; then
    wget -c --timeout=30 --tries=3 -O "$FLASHST_ZIP" "https://huggingface.co/datasets/bjdwh/FlashST-DATA/resolve/main/data.zip"
  fi
  unzip -o "$FLASHST_ZIP" -d "$DATA_DIR"
  if [[ -d "$DATA_DIR/data" ]]; then
    cp -a "$DATA_DIR/data/." "$DATA_DIR/"
  fi
fi

mkdir -p "$DATA_DIR/METR_LA" "$DATA_DIR/PEMS_BAY"
if [[ ! -f "$DATA_DIR/METR_LA/METR-LA.csv" ]]; then
  wget -c -O "$DATA_DIR/METR_LA/METR-LA.csv" "https://zenodo.org/api/records/5146275/files/METR-LA.csv/content"
fi
if [[ ! -f "$DATA_DIR/METR_LA/adj_mx.pkl" ]]; then
  wget -c -O "$DATA_DIR/METR_LA/adj_mx.pkl" "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/adj_mx.pkl"
fi
if [[ ! -f "$DATA_DIR/PEMS_BAY/PEMS-BAY.csv" ]]; then
  wget -c -O "$DATA_DIR/PEMS_BAY/PEMS-BAY.csv" "https://zenodo.org/api/records/5146275/files/PEMS-BAY.csv/content"
fi
if [[ ! -f "$DATA_DIR/PEMS_BAY/adj_mx_bay.pkl" ]]; then
  wget -c -O "$DATA_DIR/PEMS_BAY/adj_mx_bay.pkl" "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/adj_mx_bay.pkl"
fi

cd "$ROOT_DIR"
"$PYTHON_BIN" -m padtf.cli doctor
