from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from .config import ExperimentConfig
from .utils import atomic_write_json


def _load_records(directory: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))]


def fit_lopo_calibration(config: ExperimentConfig) -> Path:
    calibration_dir = config.path_for("results") / "calibration" / "grid"
    records = _load_records(calibration_dir)
    probe_records = _load_records(config.path_for("results") / "calibration" / "probes")
    probe_gains = {
        (str(record["pair"]), int(record["seed"])): float(record["signals"]["probe_gain"])
        for record in probe_records
        if record.get("config_hash") == config.hash
        and record.get("probe", {}).get("split_protocol") == "temporally_blocked_stratified"
    }
    required_probe_keys = {(str(record["pair"]), int(record["seed"])) for record in records}
    missing_probe_keys = sorted(required_probe_keys - set(probe_gains))
    if missing_probe_keys:
        raise RuntimeError(f"Missing current blocked-probe records for calibration: {missing_probe_keys}")
    pair_ids = sorted(config.raw["pairs"])
    output: dict[str, dict[str, float | int | str]] = {}
    for held_out in pair_ids:
        training_records = [record for record in records if record["pair"] != held_out]
        if not training_records:
            raise RuntimeError("No disjoint calibration-grid results are available")

        def residual(parameters: np.ndarray) -> np.ndarray:
            a, k_max = parameters
            values = []
            for record in training_records:
                b = max(
                    probe_gains[(str(record["pair"]), int(record["seed"]))],
                    float(config.section("padtf")["b_min"]),
                )
                predicted = k_max * (1.0 - np.exp(-a * float(record["ratio"])))
                predicted *= 1.0 - np.exp(-b * int(record["epochs"]))
                values.append(predicted - float(record["captured_fraction"]))
            return np.asarray(values)

        fit = least_squares(residual, x0=np.asarray([12.0, 0.65]), bounds=([0.01, 0.01], [100.0, 1.0]))
        retrain_values = [
            float(record["retrain_captured_fraction"])
            for record in training_records
            if record.get("retrain_captured_fraction") is not None
        ]
        k_retrain = float(np.clip(np.mean(retrain_values), 0.0, 1.0)) if retrain_values else 0.6
        training_pairs = sorted({str(record["pair"]) for record in training_records})
        output[held_out] = {
            "a": float(fit.x[0]),
            "k_max": float(fit.x[1]),
            "k_retrain": k_retrain,
            "held_out_pair": held_out,
            "training_pairs": training_pairs,
            "calibration_pairs": len(training_pairs),
            "grid_points": len(training_records),
            "fit_rmse": float(np.sqrt(np.mean(np.square(residual(fit.x))))),
            "protocol": "leave-one-pair-out",
            "probe_protocol": "temporally_blocked_stratified",
            "probe_config_hash": config.hash,
        }
    destination = config.path_for("results") / "calibration" / "lopo.json"
    atomic_write_json(destination, output)
    return destination
