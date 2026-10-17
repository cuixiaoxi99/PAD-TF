from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .data import DatasetBundle
from .graph import GraphBundle
from .utility import estimated_utility


@dataclass(frozen=True)
class Calibration:
    a: float
    k_max: float
    k_retrain: float
    source: str


@dataclass(frozen=True)
class DomainSignals:
    delta_nodes: float
    delta_mean: float
    delta_std: float
    delta_graph: float
    phi_static: float
    zero_shot_mae: float
    naive_mae: float
    degradation: float
    phi: float
    probe_gain: float


@dataclass(frozen=True)
class Decision:
    strategy: str
    ratio: float
    epochs: int
    utility: float
    direct_utility: float
    retrain_utility: float
    estimated_retrain_seconds: float
    candidates: list[dict[str, float]]
    calibration: Calibration

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["calibration"] = asdict(self.calibration)
        return payload


def _relative_difference(target: float, source: float, epsilon: float) -> float:
    return abs(target - source) / (abs(source) + epsilon)


def _graph_statistics(graph: GraphBundle) -> np.ndarray:
    adjacency = graph.adjacency
    binary = adjacency != 0
    degree = binary.sum(axis=1)
    density = binary.sum() / max(1, adjacency.shape[0] * (adjacency.shape[0] - 1))
    weights = adjacency[binary]
    weight_mean = float(np.mean(np.abs(weights))) if len(weights) else 0.0
    return np.asarray([float(degree.mean()), float(density), weight_mean], dtype=np.float64)


def static_discrepancy(
    source_data: DatasetBundle,
    target_data: DatasetBundle,
    source_graph: GraphBundle,
    target_graph: GraphBundle,
    epsilon: float,
) -> dict[str, float]:
    source_train = source_data.values[: source_data.train.split_end]
    target_train = target_data.values[: target_data.train.split_end]
    delta_nodes = _relative_difference(target_data.values.shape[1], source_data.values.shape[1], epsilon)
    delta_mean = _relative_difference(float(target_train.mean()), float(source_train.mean()), epsilon)
    delta_std = _relative_difference(float(target_train.std()), float(source_train.std()), epsilon)
    source_statistics = _graph_statistics(source_graph)
    target_statistics = _graph_statistics(target_graph)
    graph_terms = np.abs(target_statistics - source_statistics) / (np.abs(source_statistics) + epsilon)
    delta_graph = float(graph_terms.mean())
    bounded = np.tanh([delta_nodes, delta_mean, delta_std, delta_graph])
    return {
        "delta_nodes": float(delta_nodes),
        "delta_mean": float(delta_mean),
        "delta_std": float(delta_std),
        "delta_graph": delta_graph,
        "phi_static": float(np.mean(bounded)),
    }


def combine_signals(
    static: dict[str, float],
    zero_shot_mae: float,
    naive_mae: float,
    probe_gain: float,
    alpha: float,
    epsilon: float,
) -> DomainSignals:
    degradation = zero_shot_mae / (naive_mae + epsilon)
    phi = float(np.clip(alpha * static["phi_static"] + (1.0 - alpha) * degradation, 0.0, 1.0))
    return DomainSignals(
        **static,
        zero_shot_mae=float(zero_shot_mae),
        naive_mae=float(naive_mae),
        degradation=float(degradation),
        phi=phi,
        probe_gain=float(probe_gain),
    )


def load_calibration(config: ExperimentConfig, pair_id: str) -> Calibration:
    path = config.path_for("results") / "calibration" / "lopo.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if pair_id in payload:
            item = payload[pair_id]
            return Calibration(float(item["a"]), float(item["k_max"]), float(item["k_retrain"]), "lopo")
        if payload:
            values = list(payload.values())
            return Calibration(
                float(np.mean([item["a"] for item in values])),
                float(np.mean([item["k_max"] for item in values])),
                float(np.mean([item["k_retrain"] for item in values])),
                "lopo-global",
            )
    item = config.section("padtf")["calibration"]
    return Calibration(float(item["a"]), float(item["k_max"]), float(item["k_retrain"]), "bootstrap")


def choose_strategy(
    config: ExperimentConfig,
    pair_id: str,
    signals: DomainSignals,
    full_epoch_seconds: float,
    retrain_seconds: float | None,
    selector_seconds: float,
    signal_mode: str = "full",
    cost_weight: float | None = None,
    fixed_epoch_rate: float | None = None,
    epoch_candidates: list[int] | None = None,
    ratio_coverage: dict[float, float] | None = None,
    signed_log_nodes: float = 0.0,
    flow_target: bool = False,
) -> Decision:
    settings = config.section("padtf")
    training = config.section("training")
    calibration = load_calibration(config, pair_id)
    if signal_mode == "full":
        adaptation_need = signals.phi
    elif signal_mode == "dynamic":
        adaptation_need = float(np.clip(signals.degradation, 0.0, 1.0))
    elif signal_mode == "static":
        adaptation_need = float(np.clip(signals.phi_static, 0.0, 1.0))
    else:
        raise ValueError(f"Unknown signal mode: {signal_mode}")
    weight = float(settings["lambda"] if cost_weight is None else cost_weight)
    ratio_static_slope = float(settings.get("ratio_static_slope", 0.0))
    ratio_static_center = float(settings.get("ratio_static_center", 0.5))
    ratio_time_floor = float(settings.get("ratio_time_floor", 0.0))
    ratio_coverage_weight = float(settings.get("ratio_coverage_weight", 0.0))
    ratio_coverage_power = float(settings.get("ratio_coverage_power", 1.0))
    structural_risk_weight = float(settings.get("structural_risk_weight", 0.0))
    structural_epoch_cap = int(settings.get("structural_epoch_cap", max(settings["epochs"])))
    structural_probe_split = float(settings.get("structural_probe_split", 0.5))
    structural_high_ratio = float(settings.get("structural_high_ratio", 1.0))
    structural_default_ratio = float(settings.get("structural_default_ratio", 0.5))
    max_candidate_cost = float(settings.get("max_candidate_cost", 1.0))
    effective_ratio_rate = calibration.a * math.exp(
        ratio_static_slope * (signals.phi_static - ratio_static_center)
    )
    retrain_epochs = int(training["retrain_epochs"])
    estimated_retrain = float(retrain_seconds or max(full_epoch_seconds * retrain_epochs, 1.0e-6))
    direct_utility = 0.0
    retrain_utility = adaptation_need * calibration.k_retrain - weight
    ratio_values = [float(ratio) for ratio in settings["ratios"]]
    minimum_ratio = min(ratio_values)
    structural_breadth_score = 0.0
    structural_ratio_floor = minimum_ratio
    if ratio_coverage is not None and flow_target and signed_log_nodes > 0.0:
        minimum_coverage = float(np.clip(ratio_coverage.get(minimum_ratio, 1.0), 0.0, 1.0))
        structural_breadth_score = float(signed_log_nodes) * (1.0 - minimum_coverage)
        preferred_ratio = (
            structural_high_ratio
            if signals.probe_gain < structural_probe_split
            else structural_default_ratio
        )
        structural_ratio_floor = min(ratio_values, key=lambda ratio: abs(ratio - preferred_ratio))
    candidates: list[dict[str, float]] = []

    def build_candidate(ratio: float, epochs: int, structural_adjustment: bool = False) -> dict[str, float] | None:
        parametric_coverage = 1.0 - math.exp(-effective_ratio_rate * ratio)
        observed_coverage = parametric_coverage
        if ratio_coverage is not None:
            raw_coverage = float(ratio_coverage.get(float(ratio), parametric_coverage))
            observed_coverage = 1.0 - (1.0 - np.clip(raw_coverage, 0.0, 1.0)) ** ratio_coverage_power
        mixed_coverage = (
            (1.0 - ratio_coverage_weight) * parametric_coverage
            + ratio_coverage_weight * observed_coverage
        )
        benefit_fraction = calibration.k_max * mixed_coverage
        benefit_fraction *= 1.0 - math.exp(-epoch_rate * epochs)
        benefit = adaptation_need * benefit_fraction
        epoch_time_scale = ratio_time_floor + (1.0 - ratio_time_floor) * float(ratio)
        estimated_time = selector_seconds + full_epoch_seconds * epoch_time_scale * int(epochs)
        cost = min(estimated_time / estimated_retrain, 1.0)
        if cost > max_candidate_cost:
            return None
        base_utility = estimated_utility(benefit, cost, weight)
        structural_coverage = min(float(ratio) / max(structural_ratio_floor, minimum_ratio), 1.0)
        structural_bonus = (
            structural_risk_weight * structural_breadth_score * structural_coverage
            if structural_adjustment
            else 0.0
        )
        return {
            "ratio": float(ratio),
            "epochs": int(epochs),
            "effective_ratio_rate": float(effective_ratio_rate),
            "parametric_coverage": float(parametric_coverage),
            "observed_coverage": float(observed_coverage),
            "mixed_coverage": float(mixed_coverage),
            "epoch_time_scale": float(epoch_time_scale),
            "benefit": float(benefit),
            "estimated_seconds": float(estimated_time),
            "cost": float(cost),
            "base_utility": float(base_utility),
            "structural_breadth_score": float(structural_breadth_score),
            "structural_ratio_floor": float(structural_ratio_floor),
            "structural_coverage_bonus": float(structural_bonus),
            "structural_adjustment": float(structural_adjustment),
            "utility": float(base_utility + structural_bonus),
        }

    if signals.probe_gain > 0 or fixed_epoch_rate is not None:
        raw_epoch_rate = signals.probe_gain if fixed_epoch_rate is None else float(fixed_epoch_rate)
        epoch_rate = max(raw_epoch_rate, float(settings["b_min"]))
        candidate_epochs = settings["epochs"] if epoch_candidates is None else epoch_candidates
        for ratio in settings["ratios"]:
            for epochs in candidate_epochs:
                candidate = build_candidate(float(ratio), int(epochs))
                if candidate is not None:
                    candidates.append(candidate)
        baseline_fine = max(candidates, key=lambda item: item["base_utility"]) if candidates else None
        if baseline_fine is not None and structural_ratio_floor > baseline_fine["ratio"]:
            base_scale = ratio_time_floor + (1.0 - ratio_time_floor) * baseline_fine["ratio"]
            structural_scale = ratio_time_floor + (1.0 - ratio_time_floor) * structural_ratio_floor
            matched_epochs = baseline_fine["epochs"] * base_scale / structural_scale
            structural_epochs = min(
                [int(epoch) for epoch in settings["epochs"]],
                key=lambda epoch: (abs(epoch - matched_epochs), -epoch),
            )
            structural_epochs = min(structural_epochs, structural_epoch_cap)
            structural_candidate = build_candidate(
                structural_ratio_floor,
                structural_epochs,
                structural_adjustment=True,
            )
            if structural_candidate is not None:
                candidates.append(structural_candidate)
    best_fine = max(candidates, key=lambda item: item["utility"]) if candidates else None
    options = [("Direct", direct_utility, 0.0, 0)]
    if best_fine is not None:
        options.append(("Fine-tune", best_fine["utility"], best_fine["ratio"], best_fine["epochs"]))
    options.append(("Retrain", retrain_utility, 1.0, retrain_epochs))
    strategy, utility, ratio, epochs = max(options, key=lambda item: item[1])
    return Decision(
        strategy=strategy,
        ratio=float(ratio),
        epochs=int(epochs),
        utility=float(utility),
        direct_utility=float(direct_utility),
        retrain_utility=float(retrain_utility),
        estimated_retrain_seconds=estimated_retrain,
        candidates=candidates,
        calibration=calibration,
    )


def predict_finetune_epochs(
    config: ExperimentConfig,
    pair_id: str,
    signals: DomainSignals,
    probe_tau: float,
) -> tuple[int, dict[str, Any]]:
    settings = config.section("padtf")["epoch_predictor"]
    pair = config.pair(pair_id)
    source = config.dataset(pair["source"])
    target = config.dataset(pair["target"])
    feature_values = {
        "probe_gain": float(signals.probe_gain),
        "probe_tau": float(probe_tau),
        "phi_static": float(signals.phi_static),
        "signed_log_nodes": math.log(float(target["nodes"]) / float(source["nodes"])),
        "flow_target": float(target["feature"] == "flow"),
    }
    feature_names = list(settings["features"])
    means = np.asarray(settings["mean"], dtype=np.float64)
    scales = np.asarray(settings["scale"], dtype=np.float64)
    coefficients = np.asarray(settings["coefficients"], dtype=np.float64)
    values = np.asarray([feature_values[name] for name in feature_names], dtype=np.float64)
    standardized = (values - means) / scales
    raw_prediction = float(settings["intercept"]) + float(np.dot(coefficients, standardized))
    minimum = int(settings.get("minimum", 1))
    maximum = int(settings.get("maximum", 50))
    epochs = int(np.clip(np.rint(raw_prediction), minimum, maximum))
    return epochs, {
        "features": feature_values,
        "standardized_features": dict(zip(feature_names, standardized.tolist())),
        "raw_prediction": raw_prediction,
        "minimum": minimum,
        "maximum": maximum,
        "source": str(settings.get("source", "fixed")),
    }
