from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .runner import TIME_MATCHED_METHODS


PAIRS = [f"P{index}" for index in range(1, 7)]
METHODS = [*TIME_MATCHED_METHODS, "PAD-TF"]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _collect(config: ExperimentConfig) -> list[dict[str, Any]]:
    result_root = config.path_for("results")
    rows: list[dict[str, Any]] = []
    for pair in PAIRS:
        for seed in config.section("training")["seeds"]:
            direct_path = result_root / "runs" / pair / f"seed_{seed}" / "direct.json"
            pad_path = result_root / "runs" / pair / f"seed_{seed}" / "pad_tf.json"
            if not direct_path.exists() or not pad_path.exists():
                continue
            direct = _read_json(direct_path)
            pad = _read_json(pad_path)
            direct_rmse = float(direct["metrics"]["rmse"])
            cap = float(pad["timing"]["total_target"])
            pad_rmse = float(pad["metrics"]["rmse"])
            rows.append(
                {
                    "pair": pair,
                    "seed": int(seed),
                    "method": "PAD-TF",
                    "ratio": float(pad["ratio"]),
                    "epochs": int(pad["epochs"]),
                    "rmse": pad_rmse,
                    "direct_rmse": direct_rmse,
                    "relative_rmse_reduction": (direct_rmse - pad_rmse) / max(direct_rmse, 1.0e-8),
                    "time_cap_sec": cap,
                    "total_time_sec": cap,
                    "budget_utilization": 1.0,
                    "budget_feasible": True,
                }
            )
            for method in TIME_MATCHED_METHODS:
                method_slug = method.lower().replace("-", "_")
                path = result_root / "analysis" / "time_matched" / method_slug / pair / f"seed_{seed}.json"
                if not path.exists():
                    continue
                record = _read_json(path)
                rmse = float(record["metrics"]["rmse"])
                total = float(record["timing"]["total_target"])
                rows.append(
                    {
                        "pair": pair,
                        "seed": int(seed),
                        "method": method,
                        "ratio": float(record["ratio"]),
                        "epochs": int(record["epochs"]),
                        "rmse": rmse,
                        "direct_rmse": direct_rmse,
                        "relative_rmse_reduction": (direct_rmse - rmse) / max(direct_rmse, 1.0e-8),
                        "time_cap_sec": cap,
                        "total_time_sec": total,
                        "budget_utilization": total / max(cap, 1.0e-8),
                        "budget_feasible": total <= cap + 1.0e-6,
                    }
                )
    return rows


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pad_lookup = {
        (row["pair"], int(row["seed"])): float(row["rmse"])
        for row in rows
        if row["method"] == "PAD-TF"
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    output: list[dict[str, Any]] = []
    for method in METHODS:
        items = grouped.get(method, [])
        pair_reductions = []
        pair_rmses = []
        pair_times = []
        for pair in PAIRS:
            pair_items = [item for item in items if item["pair"] == pair]
            if not pair_items:
                continue
            pair_reductions.append(float(np.mean([item["relative_rmse_reduction"] for item in pair_items])))
            pair_rmses.append(float(np.mean([item["rmse"] for item in pair_items])))
            pair_times.append(float(np.mean([item["total_time_sec"] for item in pair_items])))
        reduction_mean, reduction_std = _mean_std(pair_reductions)
        rmse_mean, rmse_std = _mean_std(pair_rmses)
        time_mean, time_std = _mean_std(pair_times)
        utilization_mean, utilization_std = _mean_std([float(item["budget_utilization"]) for item in items])
        paired = [
            float(item["rmse"]) - pad_lookup[(item["pair"], int(item["seed"]))]
            for item in items
            if (item["pair"], int(item["seed"])) in pad_lookup
        ]
        output.append(
            {
                "method": method,
                "pair_seed_runs": len(items),
                "relative_rmse_reduction_mean": reduction_mean,
                "relative_rmse_reduction_std_across_pairs": reduction_std,
                "relative_rmse_reduction_ci95": 1.96 * reduction_std / math.sqrt(max(1, len(pair_reductions))),
                "macro_rmse_mean": rmse_mean,
                "macro_rmse_std_across_pairs": rmse_std,
                "avg_total_time_sec": time_mean,
                "total_time_std_across_pairs": time_std,
                "budget_utilization_mean": utilization_mean,
                "budget_utilization_std": utilization_std,
                "rmse_delta_vs_padtf_mean": float(np.mean(paired)) if paired else float("nan"),
                "wins_vs_padtf": sum(delta < 0.0 for delta in paired),
                "ties_vs_padtf": sum(abs(delta) <= 1.0e-12 for delta in paired),
                "losses_vs_padtf": sum(delta > 0.0 for delta in paired),
                "all_runs_within_cap": bool(items) and all(bool(item["budget_feasible"]) for item in items),
            }
        )
    return output


def _figure_tex(summary: list[dict[str, Any]]) -> str:
    labels = {
        "FlashST-Full": "Full",
        "Random-FT": "Random",
        "CoreSet-FT": "CoreSet",
        "Proxy-FT": "Proxy",
        "ELFS-FT": "ELFS",
        "BADGE-FT": "BADGE",
        "UHerding-FT": "UHerding",
        "PAD-TF": "PAD-TF",
    }
    lookup = {row["method"]: row for row in summary}
    baseline_coordinates = []
    pad_coordinate = ""
    for method in METHODS:
        row = lookup[method]
        value = 100.0 * float(row["relative_rmse_reduction_mean"])
        error = 100.0 * float(row["relative_rmse_reduction_ci95"])
        coordinate = f"({value:.3f},{labels[method]}) +- ({error:.3f},0)"
        if method == "PAD-TF":
            pad_coordinate = coordinate
        else:
            baseline_coordinates.append(coordinate)
    return "\n".join(
        [
            r"\begin{figure}[t]",
            r"\centering",
            r"\begin{tikzpicture}",
            r"\begin{axis}[",
            r"  width=\columnwidth,",
            r"  height=0.88\columnwidth,",
            r"  xbar,",
            r"  xlabel={Relative RMSE reduction (\%)},",
            r"  symbolic y coords={Full,Random,CoreSet,Proxy,ELFS,BADGE,UHerding,PAD-TF},",
            r"  ytick=data,",
            r"  y dir=reverse,",
            r"  xmin=0,",
            r"  grid=major,",
            r"  grid style={line width=.1pt, draw=gray!25},",
            r"  tick label style={font=\scriptsize},",
            r"  label style={font=\scriptsize},",
            r"  error bars/.cd, x dir=both, x explicit,",
            r"]",
            r"\addplot+[fill=blue!35, draw=blue!65!black] coordinates {",
            "  " + "\n  ".join(baseline_coordinates),
            r"};",
            r"\addplot+[fill=red!65, draw=red!80!black] coordinates {",
            f"  {pad_coordinate}",
            r"};",
            r"\end{axis}",
            r"\end{tikzpicture}",
            r"\caption{Time-matched target-domain adaptation on P1--P6. For every pair and seed, each method is capped by PAD-TF's measured end-to-end target-domain time; selection overhead counts against the cap, and the last complete feasible epoch is retained. Higher relative RMSE reduction is better. Error bars are 95\% intervals over transfer-pair means.}",
            r"\Description{Horizontal bar chart comparing relative RMSE reduction when all adaptation methods receive the same pair-specific target-domain time cap as PAD-TF.}",
            r"\label{fig:tradeoff}",
            r"\end{figure}",
            "",
        ]
    )


def aggregate_time_matched(config: ExperimentConfig) -> dict[str, Any]:
    rows = _collect(config)
    summary = _summarize(rows)
    table_dir = config.path_for("results") / "tables"
    long_path = table_dir / "time_matched_long.csv"
    summary_path = table_dir / "time_matched_summary.csv"
    figure_path = table_dir / "figure3_time_matched.tex"
    _write_csv(long_path, rows)
    _write_csv(summary_path, summary)
    complete = len(rows) == len(PAIRS) * len(config.section("training")["seeds"]) * len(METHODS)
    if complete:
        figure_path.write_text(_figure_tex(summary), encoding="utf-8")
    return {
        "complete": complete,
        "runs_found": len(rows),
        "runs_expected": len(PAIRS) * len(config.section("training")["seeds"]) * len(METHODS),
        "long_table": str(long_path),
        "summary_table": str(summary_path),
        "figure": str(figure_path) if complete else None,
    }
