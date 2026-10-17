from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .runner import ANALYSIS_POLICIES, PAPER_METHODS, TIME_MATCHED_METHODS
from .utility import estimated_utility
from .utils import atomic_write_json


PAPER_SEEDS = (2024, 2025, 2026)
PAD_TIMING_COMPONENTS = ("static", "zero_shot", "probe_training", "grid", "selection", "adaptation")


def _read_json_files(paths: list[Path]) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(paths)]


def pad_timing_consistent(record: dict[str, Any], tolerance: float = 1.0e-6) -> bool:
    timing = record.get("timing", {})
    if any(component not in timing for component in PAD_TIMING_COMPONENTS) or "total_target" not in timing:
        return False
    expected = sum(float(timing[component]) for component in PAD_TIMING_COMPONENTS)
    observed = float(timing["total_target"])
    return abs(expected - observed) <= tolerance * max(1.0, abs(expected), abs(observed))


def audit_protocol(config: ExperimentConfig) -> dict[str, Any]:
    result_root = config.path_for("results")
    records = _read_json_files(list((result_root / "runs").glob("P*/seed_*/*.json")))
    grid_records = _read_json_files(list((result_root / "calibration" / "grid").glob("*.json")))
    probe_records = _read_json_files(list((result_root / "calibration" / "probes").glob("*.json")))
    policy_records = _read_json_files(
        list((result_root / "analysis" / "policies").glob("*/P*/seed_*.json"))
    )
    control_records = _read_json_files(
        list((result_root / "analysis" / "controls").glob("*/seed_*.json"))
    )
    time_matched_records = _read_json_files(
        list((result_root / "analysis" / "time_matched").glob("*/P*/seed_*.json"))
    )
    calibration_path = result_root / "calibration" / "lopo.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.exists() else {}

    pairs = list(config.raw["pairs"])
    active_seeds = [int(seed) for seed in config.section("training")["seeds"]]
    completed = {
        (record["pair"], int(record["seed"]), record["method"])
        for record in records
    }
    active_expected = {
        (pair, seed, method)
        for pair in pairs
        for seed in active_seeds
        for method in PAPER_METHODS
    }
    paper_expected = {
        (pair, seed, method)
        for pair in pairs
        for seed in PAPER_SEEDS
        for method in PAPER_METHODS
    }

    candidate_checks = []
    argmax_checks = []
    pad_timing_checks = []
    probe_config_checks = []
    blocked_probe_checks = []
    calibration_sources = Counter()
    for record in records:
        if record["method"] != "PAD-TF" or "decision" not in record:
            continue
        decision = record["decision"]
        pad_timing_checks.append(pad_timing_consistent(record))
        probe = record.get("probe", {})
        probe_config_checks.append(
            float(probe.get("ratio", -1.0)) == float(config.section("padtf")["probe_ratio"])
            and int(probe.get("epochs", -1)) == int(config.section("padtf")["probe_epochs"])
        )
        if probe.get("split_protocol") == "temporally_blocked_stratified":
            blocked_probe_checks.append(
                int(probe.get("observed_start_gap", -1)) >= int(probe.get("minimum_start_gap", 0))
            )
        calibration_sources[str(decision["calibration"]["source"])] += 1
        for candidate in decision.get("candidates", []):
            replayed = estimated_utility(
                candidate["benefit"],
                candidate["cost"],
                float(config.section("padtf")["lambda"]),
            )
            candidate_checks.append(abs(replayed - float(candidate["utility"])) <= 1.0e-10)
        if decision.get("candidates") and record["selected_strategy"] == "Fine-tune":
            best = max(decision["candidates"], key=lambda candidate: candidate["utility"])
            argmax_checks.append(
                float(record["ratio"]) == float(best["ratio"])
                and int(record["epochs"]) == int(best["epochs"])
            )

    source_hashes: dict[str, set[str]] = defaultdict(set)
    for record in records:
        source_hash = record.get("source_checkpoint_sha256")
        if source_hash:
            source_hashes[f'{record["pair"]}/seed_{record["seed"]}'].add(str(source_hash))

    expected_grid = (
        len(pairs)
        * len(active_seeds)
        * len(config.section("padtf")["ratios"])
        * len(config.section("padtf")["epochs"])
    )
    expected_probes = len(pairs) * len(active_seeds)
    current_probe_records = [record for record in probe_records if record.get("config_hash") == config.hash]
    probe_gap_checks = [
        record.get("probe", {}).get("split_protocol") == "temporally_blocked_stratified"
        and int(record["probe"].get("observed_start_gap", -1)) >= int(record["probe"].get("minimum_start_gap", 0))
        for record in current_probe_records
    ]
    expected_time_matched = (
        6 * len(active_seeds) * len(TIME_MATCHED_METHODS)
    )
    expected_policies_active = 6 * len(active_seeds) * len(ANALYSIS_POLICIES)
    expected_policies_paper = 6 * len(PAPER_SEEDS) * len(ANALYSIS_POLICIES)
    lopo_disjoint_checks = []
    for held_out, parameters in calibration.items():
        training_pairs = parameters.get("training_pairs")
        if training_pairs is None:
            continue
        lopo_disjoint_checks.append(
            parameters.get("held_out_pair") == held_out
            and held_out not in training_pairs
            and len(training_pairs) == int(parameters["calibration_pairs"])
        )

    payload = {
        "active_seeds": active_seeds,
        "paper_seeds": list(PAPER_SEEDS),
        "main": {
            "found": len(records),
            "active_expected": len(active_expected),
            "active_missing": len(active_expected - completed),
            "paper_expected": len(paper_expected),
            "paper_missing": len(paper_expected - completed),
            "method_counts": dict(sorted(Counter(record["method"] for record in records).items())),
        },
        "grid": {
            "found": len(grid_records),
            "expected": expected_grid,
            "complete": len(grid_records) == expected_grid,
        },
        "probes": {
            "found": len(current_probe_records),
            "expected": expected_probes,
            "complete": len(current_probe_records) == expected_probes,
            "blocked_gaps_exact": bool(probe_gap_checks) and all(probe_gap_checks),
        },
        "time_matched": {
            "found": len(time_matched_records),
            "expected": expected_time_matched,
            "complete": len(time_matched_records) == expected_time_matched,
        },
        "policies": {
            "found": len(policy_records),
            "active_expected": expected_policies_active,
            "paper_expected": expected_policies_paper,
        },
        "controls": {"found": len(control_records), "paper_expected": 4},
        "decision": {
            "candidate_utility_checks": len(candidate_checks),
            "candidate_utilities_exact": bool(candidate_checks) and all(candidate_checks),
            "argmax_checks": len(argmax_checks),
            "argmax_exact": bool(argmax_checks) and all(argmax_checks),
            "calibration_sources": dict(sorted(calibration_sources.items())),
        },
        "padtf_protocol": {
            "timing_checks": len(pad_timing_checks),
            "timing_components_exact": bool(pad_timing_checks) and all(pad_timing_checks),
            "probe_config_checks": len(probe_config_checks),
            "probe_config_exact": bool(probe_config_checks) and all(probe_config_checks),
            "blocked_probe_records": len(blocked_probe_checks),
            "blocked_probe_gaps_exact": bool(blocked_probe_checks) and all(blocked_probe_checks),
        },
        "calibration": {
            "held_out_pairs": len(calibration),
            "disjoint_checks": len(lopo_disjoint_checks),
            "pair_lists_recorded": len(lopo_disjoint_checks) == len(calibration) and bool(calibration),
            "lopo_disjoint": bool(lopo_disjoint_checks) and all(lopo_disjoint_checks),
        },
        "source_checkpoint_hashes": {
            "pair_seed_groups": len(source_hashes),
            "consistent": all(len(hashes) == 1 for hashes in source_hashes.values()),
            "inconsistent_groups": sorted(key for key, hashes in source_hashes.items() if len(hashes) != 1),
        },
    }
    destination = result_root / "tables" / "protocol_audit.json"
    atomic_write_json(destination, payload)
    payload["output"] = str(destination)
    return payload
