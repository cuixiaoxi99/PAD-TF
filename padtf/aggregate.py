from __future__ import annotations

import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import rankdata

from .config import ExperimentConfig
from .utility import deployment_utility, estimated_utility, normalized_target_cost, relative_error_reduction


DISPLAY_ORDER = [
    "Direct",
    "FlashST-Full",
    "Random-FT",
    "CoreSet-FT",
    "Proxy-FT",
    "ELFS-FT",
    "BADGE-FT",
    "UHerding-FT",
    "PAD-TF",
    "Retrain",
]


def _read_records(config: ExperimentConfig) -> list[dict[str, Any]]:
    root = config.path_for("results") / "runs"
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("P*/seed_*/*.json"))]


def _read_grid_records(config: ExperimentConfig) -> list[dict[str, Any]]:
    root = config.path_for("results") / "calibration" / "grid"
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("*.json"))]


def _read_policy_records(config: ExperimentConfig) -> list[dict[str, Any]]:
    root = config.path_for("results") / "analysis" / "policies"
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("*/P*/seed_*.json"))]


def _read_control_records(config: ExperimentConfig) -> list[dict[str, Any]]:
    root = config.path_for("results") / "analysis" / "controls"
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("*/seed_*.json"))]


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return float("nan"), float("nan")
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _enrich(records: list[dict[str, Any]], weight: float) -> list[dict[str, Any]]:
    lookup = {(record["pair"], int(record["seed"]), record["method"]): record for record in records}
    enriched = []
    for record in records:
        direct = lookup.get((record["pair"], int(record["seed"]), "Direct"))
        retrain = lookup.get((record["pair"], int(record["seed"]), "Retrain"))
        direct_rmse = None if direct is None else float(direct["metrics"]["rmse"])
        retrain_time = None if retrain is None else float(retrain["timing"]["total_target"])
        rmse = float(record["metrics"]["rmse"])
        total_time = float(record["timing"]["total_target"])
        relative_gain = None if not direct_rmse else relative_error_reduction(direct_rmse, rmse)
        normalized_cost = None if not retrain_time else normalized_target_cost(total_time, retrain_time)
        utility = (
            None
            if direct_rmse is None or retrain_time is None
            else deployment_utility(direct_rmse, rmse, total_time, retrain_time, weight)
        )
        item = dict(record)
        item["relative_gain"] = relative_gain
        item["normalized_cost"] = normalized_cost
        item["utility"] = utility
        enriched.append(item)
    return enriched


def _long_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        horizons = record["metrics"].get("horizons", {})
        row = {
            "pair": record["pair"],
            "source": record["source"],
            "target": record["target"],
            "seed": record["seed"],
            "method": record["method"],
            "selected_strategy": record["selected_strategy"],
            "rho": record["ratio"],
            "epochs": record["epochs"],
            "mae": record["metrics"]["mae"],
            "rmse": record["metrics"]["rmse"],
            "mape": record["metrics"]["mape"],
            "rmse_h3": horizons.get("3", {}).get("rmse"),
            "rmse_h6": horizons.get("6", {}).get("rmse"),
            "rmse_h12": horizons.get("12", {}).get("rmse"),
            "mape_h3": horizons.get("3", {}).get("mape"),
            "mape_h6": horizons.get("6", {}).get("mape"),
            "mape_h12": horizons.get("12", {}).get("mape"),
            "selection_time_sec": record["timing"].get("selection", 0.0),
            "probe_time_sec": record["timing"].get("probe", 0.0),
            "adapt_time_sec": record["timing"].get("adaptation", 0.0),
            "total_time_sec": record["timing"]["total_target"],
            "relative_gain": record.get("relative_gain"),
            "normalized_cost": record.get("normalized_cost"),
            "utility": record.get("utility"),
            "source_checkpoint_sha256": record["source_checkpoint_sha256"],
            "config_hash": record["config_hash"],
            "upstream_commit": record["upstream_commit"],
        }
        rows.append(row)
    return rows


def _table4(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_method_pair[(record["method"], record["pair"])].append(record)
    rows = []
    for method in DISPLAY_ORDER:
        row: dict[str, Any] = {"method": method}
        for pair in [f"P{index}" for index in range(1, 9)]:
            values = [float(item["metrics"]["rmse"]) for item in by_method_pair.get((method, pair), [])]
            mean, std = _mean_std(values)
            row[f"{pair}_mean"] = mean
            row[f"{pair}_std"] = std
        per_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for pair in [f"P{index}" for index in range(1, 7)]:
            for item in by_method_pair.get((method, pair), []):
                per_seed[int(item["seed"])].append(item)
        macro_values = []
        time_values = []
        cost_values = []
        utility_values = []
        for items in per_seed.values():
            if len({item["pair"] for item in items}) != 6:
                continue
            macro_values.append(float(np.mean([item["metrics"]["rmse"] for item in items])))
            time_values.append(float(np.mean([item["timing"]["total_target"] for item in items])))
            if all(item.get("normalized_cost") is not None for item in items):
                cost_values.append(float(np.mean([item["normalized_cost"] for item in items])))
            if all(item.get("utility") is not None for item in items):
                utility_values.append(float(np.mean([item["utility"] for item in items])))
        for prefix, values in (
            ("macro_rmse", macro_values),
            ("avg_time_sec", time_values),
            ("cost", cost_values),
            ("utility", utility_values),
        ):
            row[f"{prefix}_mean"], row[f"{prefix}_std"] = _mean_std(values)
        rows.append(row)
    return rows


def _format_value(mean: float, std: float, digits: int = 2) -> str:
    if math.isnan(mean):
        return "--"
    return f"{mean:.{digits}f}$\\pm${std:.{digits}f}"


def _write_table4_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = []
    for row in rows:
        values = [_format_value(row[f"P{index}_mean"], row[f"P{index}_std"]) for index in range(1, 9)]
        values.extend(
            [
                _format_value(row["macro_rmse_mean"], row["macro_rmse_std"]),
                _format_value(row["avg_time_sec_mean"], row["avg_time_sec_std"]),
                _format_value(row["cost_mean"], row["cost_std"], 3),
                _format_value(row["utility_mean"], row["utility_std"], 3),
            ]
        )
        label = "PAD-TF (Ours)" if row["method"] == "PAD-TF" else row["method"]
        lines.append(label + " & " + " & ".join(values) + r" \\")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _horizon_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in ["Direct", "FlashST-Full", "Retrain", "PAD-TF"]:
        selected = [record for record in records if record["method"] == method and record["pair"] in {f"P{i}" for i in range(1, 7)}]
        row: dict[str, Any] = {"method": method}
        for horizon in (3, 6, 12):
            for metric in ("rmse", "mape"):
                values = [float(record["metrics"]["horizons"][str(horizon)][metric]) for record in selected]
                row[f"{metric}_h{horizon}"] = float(np.mean(values)) if values else float("nan")
        rows.append(row)
    return rows


def _decision_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        if record["method"] != "PAD-TF":
            continue
        rows.append(
            {
                "pair": record["pair"],
                "seed": record["seed"],
                "strategy": record["selected_strategy"],
                "rho": record["ratio"],
                "epochs": record["epochs"],
                "phi_static": record["signals"]["phi_static"],
                "phi": record["signals"]["phi"],
                "probe_gain": record["signals"]["probe_gain"],
                "calibration": record["decision"]["calibration"]["source"],
                "rmse": record["metrics"]["rmse"],
                "total_time_sec": record["timing"]["total_target"],
                "utility": record.get("utility"),
            }
        )
    return rows


def _static_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for pair in [f"P{index}" for index in range(1, 9)]:
        candidates = [record for record in records if record["pair"] == pair and record["method"] == "PAD-TF"]
        if not candidates:
            continue
        signal = candidates[0]["signals"]
        rows.append(
            {
                "pair": pair,
                "source": candidates[0]["source"],
                "target": candidates[0]["target"],
                "delta_N": signal["delta_nodes"],
                "delta_mu": signal["delta_mean"],
                "delta_sigma": signal["delta_std"],
                "delta_g": signal["delta_graph"],
                "phi_static": signal["phi_static"],
                "level": "High" if signal["phi_static"] >= 0.45 else "Medium" if signal["phi_static"] >= 0.25 else "Low",
            }
        )
    return rows


def _sign_flip_pvalue(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    observed = abs(float(array.mean()))
    if len(array) == 0:
        return float("nan")
    if len(array) <= 16:
        statistics = []
        for signs in itertools.product((-1.0, 1.0), repeat=len(array)):
            statistics.append(abs(float(np.mean(array * np.asarray(signs)))))
        return float((sum(value >= observed - 1.0e-12 for value in statistics)) / len(statistics))
    rng = np.random.default_rng(2024)
    statistics = [abs(float(np.mean(array * rng.choice([-1.0, 1.0], len(array))))) for _ in range(100000)]
    return float((1 + sum(value >= observed for value in statistics)) / (len(statistics) + 1))


def _holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=np.float64)
    running = 0.0
    total = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def _significance_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pad = [record for record in records if record["method"] == "PAD-TF"]
    comparisons = ["Fixed-FT", "FlashST-Full", "Random-FT", "CoreSet-FT", "Proxy-FT", "ELFS-FT", "BADGE-FT", "UHerding-FT", "Retrain"]
    rows = []
    pvalues = []
    for method in comparisons:
        pair_differences = []
        for pair in [f"P{index}" for index in range(1, 9)]:
            pad_values = [record["metrics"]["rmse"] for record in pad if record["pair"] == pair]
            baseline_values = [record["metrics"]["rmse"] for record in records if record["pair"] == pair and record["method"] == method]
            if pad_values and baseline_values:
                pair_differences.append(float(np.mean(pad_values) - np.mean(baseline_values)))
        pvalue = _sign_flip_pvalue(pair_differences)
        pvalues.append(pvalue)
        rows.append(
            {
                "comparison": f"PAD-TF vs {method}",
                "pair_clusters": len(pair_differences),
                "mean_delta_rmse": float(np.mean(pair_differences)) if pair_differences else float("nan"),
                "sign_flip_p": pvalue,
            }
        )
    adjusted = _holm_adjust(pvalues)
    for row, value in zip(rows, adjusted):
        row["holm_p"] = value
    return rows


def _utility_sweep(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    selected_methods = ["PAD-TF", "BADGE-FT", "UHerding-FT"]
    for weight in (0.1, 0.2, 0.3, 0.5, 1.0):
        for method in selected_methods:
            selected = [record for record in records if record["method"] == method]
            values = [
                estimated_utility(record["relative_gain"], record["normalized_cost"], weight)
                for record in selected
                if record.get("relative_gain") is not None and record.get("normalized_cost") is not None
            ]
            mean, std = _mean_std(values)
            rows.append({"lambda": weight, "method": method, "utility_mean": mean, "utility_std": std, "runs": len(values)})
    return rows


def _correlations(x_values: list[float], y_values: list[float]) -> tuple[float, float]:
    if len(x_values) < 2 or len(y_values) < 2:
        return float("nan"), float("nan")
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
        return float("nan"), float("nan")
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    return pearson, spearman


def _probe_reliability(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {(record["pair"], int(record["seed"]), record["method"]): record for record in records}
    points = []
    for pair in [f"P{index}" for index in range(1, 9)]:
        probe_values = []
        final_values = []
        for seed in sorted({int(record["seed"]) for record in records if record["pair"] == pair}):
            direct = lookup.get((pair, seed, "Direct"))
            pad = lookup.get((pair, seed, "PAD-TF"))
            if direct is None or pad is None or "signals" not in pad:
                continue
            direct_mae = float(direct["metrics"]["mae"])
            final_gain = (direct_mae - float(pad["metrics"]["mae"])) / max(direct_mae, 1.0e-8)
            probe_values.append(float(pad["signals"]["probe_gain"]))
            final_values.append(final_gain)
        if probe_values:
            points.append(
                {
                    "pair": pair,
                    "runs": len(probe_values),
                    "probe_gain_mean": float(np.mean(probe_values)),
                    "probe_gain_std": _mean_std(probe_values)[1],
                    "final_relative_mae_gain_mean": float(np.mean(final_values)),
                    "final_relative_mae_gain_std": _mean_std(final_values)[1],
                }
            )
    pearson, spearman = _correlations(
        [row["probe_gain_mean"] for row in points],
        [row["final_relative_mae_gain_mean"] for row in points],
    )
    summary = [{"pairs": len(points), "pearson_r": pearson, "spearman_rho": spearman}]
    return points, summary


def _lopo_calibration_rows(config: ExperimentConfig, grid_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calibration_path = config.path_for("results") / "calibration" / "lopo.json"
    if not calibration_path.exists():
        return []
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    b_min = float(config.section("padtf")["b_min"])
    rows = []
    all_observed: list[float] = []
    all_predicted: list[float] = []
    for pair in sorted(calibration):
        parameters = calibration[pair]
        selected = [record for record in grid_records if record["pair"] == pair]
        observed = []
        predicted = []
        for record in selected:
            epoch_rate = max(float(record["probe_gain"]), b_min)
            estimate = float(parameters["k_max"]) * (1.0 - np.exp(-float(parameters["a"]) * float(record["ratio"])))
            estimate *= 1.0 - np.exp(-epoch_rate * int(record["epochs"]))
            predicted.append(float(estimate))
            observed.append(float(record["captured_fraction"]))
        if not observed:
            continue
        residual = np.asarray(predicted) - np.asarray(observed)
        pearson, spearman = _correlations(predicted, observed)
        rows.append(
            {
                "pair": pair,
                "grid_points": len(observed),
                "rmse": float(np.sqrt(np.mean(np.square(residual)))),
                "mae": float(np.mean(np.abs(residual))),
                "pearson_r": pearson,
                "spearman_rho": spearman,
                "calibration_pairs": parameters["calibration_pairs"],
            }
        )
        all_observed.extend(observed)
        all_predicted.extend(predicted)
    if all_observed:
        residual = np.asarray(all_predicted) - np.asarray(all_observed)
        pearson, spearman = _correlations(all_predicted, all_observed)
        rows.append(
            {
                "pair": "ALL",
                "grid_points": len(all_observed),
                "rmse": float(np.sqrt(np.mean(np.square(residual)))),
                "mae": float(np.mean(np.abs(residual))),
                "pearson_r": pearson,
                "spearman_rho": spearman,
                "calibration_pairs": "LOPO",
            }
        )
    return rows


def _oracle_rows(
    records: list[dict[str, Any]],
    grid_records: list[dict[str, Any]],
    weight: float,
) -> list[dict[str, Any]]:
    lookup = {(record["pair"], int(record["seed"]), record["method"]): record for record in records}
    rows = []
    for pair in [f"P{index}" for index in range(1, 9)]:
        grid = [record for record in grid_records if record["pair"] == pair and int(record["seed"]) == 2024]
        direct = lookup.get((pair, 2024, "Direct"))
        retrain = lookup.get((pair, 2024, "Retrain"))
        pad = lookup.get((pair, 2024, "PAD-TF"))
        if not grid or direct is None or retrain is None or pad is None:
            continue
        direct_rmse = float(direct["metrics"]["rmse"])
        retrain_time = float(retrain["timing"]["total_target"])
        probe_time = float(pad["timing"].get("probe", 0.0))

        def utility(rmse: float, total_time: float) -> float:
            return deployment_utility(direct_rmse, rmse, total_time, retrain_time, weight)

        candidates = [
            {
                "family": "Direct",
                "rho": 0.0,
                "epochs": 0,
                "rmse": direct_rmse,
                "time": probe_time,
                "utility": utility(direct_rmse, probe_time),
            },
            {
                "family": "Retrain",
                "rho": 1.0,
                "epochs": int(retrain["epochs"]),
                "rmse": float(retrain["metrics"]["rmse"]),
                "time": probe_time + retrain_time,
                "utility": utility(float(retrain["metrics"]["rmse"]), probe_time + retrain_time),
            },
        ]
        for record in grid:
            total_time = probe_time + float(record["selection_seconds"]) + float(record["adaptation_seconds"])
            candidates.append(
                {
                    "family": "Fine-tune",
                    "rho": float(record["ratio"]),
                    "epochs": int(record["epochs"]),
                    "rmse": float(record["rmse"]),
                    "time": total_time,
                    "utility": utility(float(record["rmse"]), total_time),
                }
            )
        oracle = max(candidates, key=lambda candidate: candidate["utility"])
        chosen_family = str(pad["selected_strategy"])
        chosen_utility = utility(float(pad["metrics"]["rmse"]), float(pad["timing"]["total_target"]))
        rows.append(
            {
                "pair": pair,
                "seed": 2024,
                "chosen_family": chosen_family,
                "oracle_family": oracle["family"],
                "family_correct": chosen_family == oracle["family"],
                "chosen_rho": float(pad["ratio"]),
                "oracle_rho": oracle["rho"],
                "rho_abs_error": abs(float(pad["ratio"]) - float(oracle["rho"])),
                "chosen_epochs": int(pad["epochs"]),
                "oracle_epochs": oracle["epochs"],
                "epoch_abs_error": abs(int(pad["epochs"]) - int(oracle["epochs"])),
                "chosen_rmse": float(pad["metrics"]["rmse"]),
                "oracle_rmse": oracle["rmse"],
                "chosen_utility": chosen_utility,
                "oracle_utility": oracle["utility"],
                "utility_regret": max(0.0, float(oracle["utility"]) - chosen_utility),
                "grid_candidates": len(grid),
                "oracle_time_sec": oracle["time"],
            }
        )
    return rows


def _enrich_policy_records(
    main_records: list[dict[str, Any]],
    policy_records: list[dict[str, Any]],
    evaluation_weight: float,
) -> list[dict[str, Any]]:
    lookup = {(record["pair"], int(record["seed"]), record["method"]): record for record in main_records}
    rows = []
    for record in policy_records:
        pair = record["pair"]
        seed = int(record["seed"])
        direct = lookup.get((pair, seed, "Direct"))
        retrain = lookup.get((pair, seed, "Retrain"))
        decision_weight = record.get("decision_arguments", {}).get("cost_weight", evaluation_weight)
        if direct is None or retrain is None:
            item = dict(record)
            item.update(
                {
                    "relative_gain": None,
                    "normalized_cost": None,
                    "utility": None,
                    "decision_weight": float(decision_weight),
                    "decision_weight_utility": None,
                }
            )
            rows.append(item)
            continue
        direct_rmse = float(direct["metrics"]["rmse"])
        retrain_time = float(retrain["timing"]["total_target"])
        rmse = float(record["metrics"]["rmse"])
        total_time = float(record["timing"]["total_target"])
        relative_gain = relative_error_reduction(direct_rmse, rmse)
        normalized_cost = normalized_target_cost(total_time, retrain_time)
        item = dict(record)
        item.update(
            {
                "relative_gain": relative_gain,
                "normalized_cost": normalized_cost,
                "utility": estimated_utility(relative_gain, normalized_cost, evaluation_weight),
                "decision_weight": float(decision_weight),
                "decision_weight_utility": estimated_utility(relative_gain, normalized_cost, float(decision_weight)),
            }
        )
        rows.append(item)
    return rows


def _macro_summary(records: list[dict[str, Any]]) -> dict[str, float]:
    per_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["pair"] in {f"P{index}" for index in range(1, 7)}:
            per_seed[int(record["seed"])].append(record)
    measurements: dict[str, list[float]] = defaultdict(list)
    for items in per_seed.values():
        if len({item["pair"] for item in items}) != 6:
            continue
        measurements["rmse"].append(float(np.mean([item["metrics"]["rmse"] for item in items])))
        if all(item.get("normalized_cost") is not None for item in items):
            measurements["cost"].append(float(np.mean([item["normalized_cost"] for item in items])))
        if all(item.get("utility") is not None for item in items):
            measurements["utility"].append(float(np.mean([item["utility"] for item in items])))
        measurements["ratio"].append(float(np.mean([item["ratio"] for item in items])))
        measurements["epochs"].append(float(np.mean([item["epochs"] for item in items])))
        measurements["time"].append(float(np.mean([item["timing"]["total_target"] for item in items])))
    summary: dict[str, float] = {"seeds": float(len(measurements["rmse"]))}
    for key in ("rmse", "cost", "utility", "ratio", "epochs", "time"):
        values = measurements[key]
        summary[f"{key}_mean"], summary[f"{key}_std"] = _mean_std(values)
    return summary


def _ablation_rows(
    main_records: list[dict[str, Any]],
    policy_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    base = [record for record in main_records if record["method"] == "PAD-TF"]
    rows.append({"variant": "PAD-TF", **_macro_summary(base)})
    labels = {
        "dynamic_only": "Dynamic signals only",
        "static_only": "Static signals only",
        "fixed_b": "Fixed response rate b",
        "no_cost": "No cost term",
    }
    for policy, label in labels.items():
        selected = [record for record in policy_records if record["policy"] == policy]
        rows.append({"variant": label, **_macro_summary(selected)})
    return rows


def _sensitivity_rows(
    main_records: list[dict[str, Any]],
    policy_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for weight in (0.1, 0.2, 0.3, 0.5, 1.0):
        if weight == 0.3:
            selected = [record for record in main_records if record["method"] == "PAD-TF"]
        else:
            policy = f"lambda_{weight:.1f}"
            selected = [record for record in policy_records if record["policy"] == policy]
        rows.append({"lambda": weight, **_macro_summary(selected)})
    return rows


def _control_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    labels = {"zero_shift": "Zero shift", "negative_transfer": "Negative-transfer stress"}
    for control, label in labels.items():
        selected = [record for record in records if record["control"] == control]
        if not selected:
            continue
        oracle_families = sorted({record["oracle"]["family"] for record in selected})
        pad_families = sorted({record["pad_tf"]["selected_strategy"] for record in selected})
        rows.append(
            {
                "control": label,
                "runs": len(selected),
                "direct_rmse_mean": float(np.mean([record["executed"]["Direct"]["metrics"]["rmse"] for record in selected])),
                "direct_rmse_std": _mean_std([record["executed"]["Direct"]["metrics"]["rmse"] for record in selected])[1],
                "full_rmse_mean": float(np.mean([record["executed"]["FlashST-Full"]["metrics"]["rmse"] for record in selected])),
                "full_rmse_std": _mean_std([record["executed"]["FlashST-Full"]["metrics"]["rmse"] for record in selected])[1],
                "retrain_rmse_mean": float(np.mean([record["executed"]["Retrain"]["metrics"]["rmse"] for record in selected])),
                "retrain_rmse_std": _mean_std([record["executed"]["Retrain"]["metrics"]["rmse"] for record in selected])[1],
                "oracle_family": "/".join(oracle_families),
                "pad_family": "/".join(pad_families),
                "correct": sum(bool(record["family_correct"]) for record in selected),
            }
        )
    return rows


def aggregate(config: ExperimentConfig) -> dict[str, Any]:
    raw_records = _read_records(config)
    grid_records = _read_grid_records(config)
    raw_policy_records = _read_policy_records(config)
    control_records = _read_control_records(config)
    weight = float(config.section("padtf")["lambda"])
    records = _enrich(raw_records, weight)
    policy_records = _enrich_policy_records(records, raw_policy_records, weight)
    table_dir = config.path_for("results") / "tables"
    long_rows = _long_rows(records)
    table4_rows = _table4(records)
    horizon_rows = _horizon_rows(records)
    decision_rows = _decision_rows(records)
    static_rows = _static_rows(records)
    significance_rows = _significance_rows(records)
    sweep_rows = _utility_sweep(records)
    probe_points, probe_summary = _probe_reliability(records)
    calibration_rows = _lopo_calibration_rows(config, grid_records)
    oracle_rows = _oracle_rows(records, grid_records, weight)
    ablation_rows = _ablation_rows(records, policy_records)
    sensitivity_rows = _sensitivity_rows(records, policy_records)
    control_rows = _control_rows(control_records)
    _write_csv(table_dir / "main_performance_long.csv", long_rows)
    _write_csv(table_dir / "table3_static_discrepancy.csv", static_rows)
    _write_csv(table_dir / "table4_main.csv", table4_rows)
    _write_table4_tex(table_dir / "table4_rows.tex", table4_rows)
    _write_csv(table_dir / "horizon_metrics.csv", horizon_rows)
    _write_csv(table_dir / "padtf_decisions.csv", decision_rows)
    _write_csv(table_dir / "pair_cluster_significance.csv", significance_rows)
    _write_csv(table_dir / "utility_weight_sweep.csv", sweep_rows)
    _write_csv(table_dir / "probe_reliability_points.csv", probe_points)
    _write_csv(table_dir / "probe_reliability_summary.csv", probe_summary)
    _write_csv(table_dir / "lopo_calibration_diagnostics.csv", calibration_rows)
    _write_csv(table_dir / "full_grid_oracle_diagnostics.csv", oracle_rows)
    _write_csv(table_dir / "policy_runs_long.csv", _long_rows(policy_records))
    _write_csv(table_dir / "ablation_results.csv", ablation_rows)
    _write_csv(table_dir / "cost_weight_sensitivity.csv", sensitivity_rows)
    _write_csv(table_dir / "strategy_controls.csv", control_rows)
    completed = {(record["pair"], int(record["seed"]), record["method"]) for record in records}
    expected = {
        (pair, int(seed), method)
        for pair in config.raw["pairs"]
        for seed in config.section("training")["seeds"]
        for method in DISPLAY_ORDER
    }
    paper_expected = {
        (pair, seed, method)
        for pair in config.raw["pairs"]
        for seed in (2024, 2025, 2026)
        for method in DISPLAY_ORDER
    }
    policy_active_expected = 6 * len(config.section("training")["seeds"]) * 8
    policy_paper_expected = 6 * 3 * 8
    bootstrap_decisions = sum(row["calibration"] == "bootstrap" for row in decision_rows)
    summary = {
        "runs_found": len(records),
        "runs_expected": len(expected),
        "runs_missing": len(expected - completed),
        "complete": completed >= expected,
        "paper_runs_expected": len(paper_expected),
        "paper_runs_missing": len(paper_expected - completed),
        "paper_main_complete": completed >= paper_expected,
        "bootstrap_calibration_decisions": bootstrap_decisions,
        "lopo_ready": bootstrap_decisions == 0 and bool(decision_rows),
        "grid_runs_found": len(grid_records),
        "policy_runs_found": len(policy_records),
        "policy_runs_active_expected": policy_active_expected,
        "policy_runs_expected": policy_paper_expected,
        "policy_runs_complete": len(policy_records) >= policy_paper_expected,
        "control_runs_found": len(control_records),
        "control_runs_expected": 4,
        "control_runs_complete": len(control_records) >= 4,
        "paper_publishable": (
            completed >= paper_expected
            and len(policy_records) >= policy_paper_expected
            and len(control_records) >= 4
        ),
        "oracle_pairs": len(oracle_rows),
        "oracle_family_accuracy": (
            float(np.mean([row["family_correct"] for row in oracle_rows])) if oracle_rows else None
        ),
        "oracle_mean_utility_regret": (
            float(np.mean([row["utility_regret"] for row in oracle_rows])) if oracle_rows else None
        ),
        "tables": [str(path) for path in sorted(table_dir.glob("*"))],
    }
    (table_dir / "aggregation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
