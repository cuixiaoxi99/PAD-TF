from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset

from .data import StandardScaler, make_loader
from .model import FlashSTTransferModel
from .utils import elapsed_timer


@dataclass
class SelectionResult:
    indices: list[int]
    elapsed_seconds: float
    metadata: dict[str, Any]


@dataclass
class CoverageProfile:
    values: dict[float, float]
    elapsed_seconds: float
    metadata: dict[str, Any]


@torch.no_grad()
def extract_embeddings(
    model: FlashSTTransferModel,
    dataset: torch.utils.data.Dataset,
    scaler: StandardScaler,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = make_loader(dataset, batch_size, False, seed)
    model.eval()
    features: list[np.ndarray] = []
    losses: list[np.ndarray] = []
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        prediction, representation = model(inputs)
        if representation is None:
            pooled = inputs[..., :1].mean(dim=(1, 2))
        else:
            pooled = representation.mean(dim=(1, 2))
        real_prediction = scaler.inverse_tensor(prediction)
        real_target = scaler.inverse_tensor(targets)
        mask = real_target > 0.0
        per_element = torch.abs(real_prediction - real_target) * mask
        denominator = mask.flatten(1).sum(dim=1).clamp_min(1)
        per_sample = per_element.flatten(1).sum(dim=1) / denominator
        features.append(pooled.detach().cpu().numpy())
        losses.append(per_sample.detach().cpu().numpy())
    return np.concatenate(features, axis=0), np.concatenate(losses, axis=0)


@torch.no_grad()
def mc_uncertainty(
    model: FlashSTTransferModel,
    dataset: torch.utils.data.Dataset,
    scaler: StandardScaler,
    device: torch.device,
    batch_size: int,
    seed: int,
    passes: int,
) -> np.ndarray:
    loader = make_loader(dataset, batch_size, False, seed)
    model.train()
    scores: list[np.ndarray] = []
    for inputs, _ in loader:
        inputs = inputs.to(device, non_blocking=True)
        predictions = []
        for _ in range(passes):
            prediction, _ = model(inputs)
            predictions.append(scaler.inverse_tensor(prediction))
        stack = torch.stack(predictions, dim=0)
        score = stack.var(dim=0, unbiased=False).mean(dim=(1, 2, 3))
        scores.append(score.detach().cpu().numpy())
    model.eval()
    return np.concatenate(scores, axis=0)


def _standardize(features: np.ndarray) -> np.ndarray:
    centered = features - features.mean(axis=0, keepdims=True)
    scale = features.std(axis=0, keepdims=True)
    return centered / np.maximum(scale, 1.0e-6)


def _project(features: np.ndarray, dimension: int, seed: int) -> np.ndarray:
    if features.shape[1] <= dimension:
        return _standardize(features.astype(np.float32, copy=False))
    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(features.shape[1], dimension)).astype(np.float32)
    projection /= math.sqrt(dimension)
    return _standardize(features.astype(np.float32, copy=False) @ projection)


def _farthest_rank(features: np.ndarray, budget: int, max_anchors: int) -> list[int]:
    count = len(features)
    if budget >= count:
        return list(range(count))
    anchors = min(budget, max_anchors)
    center = features.mean(axis=0)
    first = int(np.argmax(np.sum(np.square(features - center), axis=1)))
    selected = [first]
    minimum = np.sum(np.square(features - features[first]), axis=1)
    minimum[first] = -1.0
    for _ in range(1, anchors):
        next_index = int(np.argmax(minimum))
        selected.append(next_index)
        distance = np.sum(np.square(features - features[next_index]), axis=1)
        minimum = np.minimum(minimum, distance)
        minimum[selected] = -1.0
    if anchors < budget:
        remaining = np.argsort(-minimum)
        selected_set = set(selected)
        selected.extend(int(index) for index in remaining if int(index) not in selected_set)
    return selected[:budget]


def _nested_coreset_coverage(features: np.ndarray, ratios: list[float]) -> dict[float, float]:
    count = len(features)
    if count == 0:
        raise ValueError("Coverage profiling requires at least one feature vector")
    center = features.mean(axis=0)
    baseline = float(np.sqrt(np.sum(np.square(features - center), axis=1)).mean())
    if baseline <= 1.0e-12:
        return {float(ratio): 1.0 for ratio in ratios}
    budgets = {
        float(ratio): max(1, min(count, int(math.ceil(count * float(ratio)))))
        for ratio in ratios
    }
    checkpoints: dict[int, list[float]] = {}
    for ratio, budget in budgets.items():
        checkpoints.setdefault(budget, []).append(ratio)
    first = int(np.argmax(np.sum(np.square(features - center), axis=1)))
    minimum = np.sum(np.square(features - features[first]), axis=1)
    minimum[first] = 0.0
    selected = np.zeros(count, dtype=bool)
    selected[first] = True
    coverage: dict[float, float] = {}
    for selected_count in range(1, max(checkpoints) + 1):
        if selected_count in checkpoints:
            residual = float(np.sqrt(np.maximum(minimum, 0.0)).mean())
            value = float(np.clip(1.0 - residual / baseline, 0.0, 1.0))
            for ratio in checkpoints[selected_count]:
                coverage[ratio] = value
        if selected_count == max(checkpoints):
            break
        candidate = minimum.copy()
        candidate[selected] = -1.0
        next_index = int(np.argmax(candidate))
        selected[next_index] = True
        distance = np.sum(np.square(features - features[next_index]), axis=1)
        minimum = np.minimum(minimum, distance)
        minimum[selected] = 0.0
    return coverage


def estimate_coreset_coverage(
    model: FlashSTTransferModel,
    dataset: torch.utils.data.Dataset,
    scaler: StandardScaler,
    ratios: list[float],
    device: torch.device,
    seed: int,
    settings: dict[str, Any],
) -> CoverageProfile:
    sample_count = min(len(dataset), int(settings.get("coverage_profile_samples", 1024)))
    sample_indices = np.linspace(0, len(dataset) - 1, sample_count, dtype=np.int64)
    sampled = Subset(dataset, sample_indices.tolist())
    with elapsed_timer(device) as timer:
        features, _ = extract_embeddings(
            model,
            sampled,
            scaler,
            device,
            int(settings["embedding_batch_size"]),
            seed,
        )
        projected = _project(features, int(settings["random_projection_dim"]), seed)
        values = _nested_coreset_coverage(projected, ratios)
    return CoverageProfile(
        values=values,
        elapsed_seconds=timer["seconds"],
        metadata={
            "algorithm": "sampled_nested_k_center_coverage",
            "sample_count": sample_count,
            "embedding_dim": projected.shape[1],
        },
    )


def _kmeanspp_rank(features: np.ndarray, budget: int, max_anchors: int, seed: int) -> list[int]:
    count = len(features)
    if budget >= count:
        return list(range(count))
    rng = np.random.default_rng(seed)
    anchors = min(budget, max_anchors)
    selected = [int(rng.integers(0, count))]
    minimum = np.sum(np.square(features - features[selected[0]]), axis=1)
    minimum[selected[0]] = 0.0
    for _ in range(1, anchors):
        total = float(minimum.sum())
        next_index = int(rng.integers(0, count)) if total <= 0 else int(rng.choice(count, p=minimum / total))
        while next_index in selected:
            next_index = int(rng.integers(0, count))
        selected.append(next_index)
        distance = np.sum(np.square(features - features[next_index]), axis=1)
        minimum = np.minimum(minimum, distance)
        minimum[selected] = 0.0
    if anchors < budget:
        remaining = np.argsort(-minimum)
        selected_set = set(selected)
        selected.extend(int(index) for index in remaining if int(index) not in selected_set)
    return selected[:budget]


class ProxyRegressor(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        hidden = min(64, max(16, dimension))
        self.network = nn.Sequential(nn.Linear(dimension, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def _proxy_dynamics(
    features: np.ndarray,
    targets: np.ndarray,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    x = torch.from_numpy(features).float().to(device)
    y = torch.from_numpy(targets).float().to(device)
    model = ProxyRegressor(features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    trajectories = []
    for _ in range(epochs):
        permutation = torch.randperm(len(x), device=device)
        model.train()
        for start in range(0, len(x), 512):
            index = permutation[start : start + 512]
            prediction = model(x[index])
            loss = torch.nn.functional.smooth_l1_loss(prediction, y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            trajectories.append(torch.abs(model(x) - y).cpu().numpy())
    with torch.no_grad():
        final_prediction = model(x).cpu().numpy()
    return np.stack(trajectories, axis=1), final_prediction


def select_windows(
    method: str,
    model: FlashSTTransferModel,
    dataset: torch.utils.data.Dataset,
    scaler: StandardScaler,
    ratio: float,
    device: torch.device,
    seed: int,
    settings: dict[str, Any],
) -> SelectionResult:
    budget = max(1, min(len(dataset), int(math.ceil(len(dataset) * ratio))))
    method_key = method.lower().replace("-ft", "")
    with elapsed_timer(device) as timer:
        if method_key == "random":
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(dataset), size=budget, replace=False).tolist()
            metadata = {"algorithm": "uniform_without_replacement"}
        else:
            batch_size = int(settings["embedding_batch_size"])
            features, losses = extract_embeddings(model, dataset, scaler, device, batch_size, seed)
            projected = _project(features, int(settings["random_projection_dim"]), seed)
            max_anchors = int(settings["max_diversity_anchors"])
            if method_key == "coreset":
                indices = _farthest_rank(projected, budget, budget)
                metadata = {
                    "algorithm": "k_center_greedy",
                    "embedding_dim": projected.shape[1],
                    "exact_budget_iterations": budget,
                }
            elif method_key == "budget-coreset":
                indices = _farthest_rank(projected, budget, max_anchors)
                metadata = {
                    "algorithm": "anchor_approximated_k_center",
                    "embedding_dim": projected.shape[1],
                    "diversity_anchors": min(budget, max_anchors),
                }
            elif method_key == "badge":
                scale = losses / max(float(losses.mean()), 1.0e-6)
                gradient_embedding = projected * scale[:, None]
                indices = _kmeanspp_rank(gradient_embedding, budget, max_anchors, seed)
                metadata = {"algorithm": "gradient_embedding_kmeanspp", "embedding_dim": projected.shape[1]}
            elif method_key in {"proxy", "elfs"}:
                trajectories, prediction = _proxy_dynamics(
                    projected,
                    losses,
                    int(settings["proxy_epochs"]),
                    seed,
                    device,
                )
                if method_key == "proxy":
                    score = np.abs(prediction - losses)
                    augmented = np.concatenate([projected, _standardize(score[:, None])], axis=1)
                    indices = _farthest_rank(augmented, budget, max_anchors)
                    metadata = {"algorithm": "selection_via_proxy", "proxy_epochs": int(settings["proxy_epochs"])}
                else:
                    dynamics = np.column_stack(
                        [trajectories.mean(axis=1), trajectories.std(axis=1), trajectories[:, 0] - trajectories[:, -1]]
                    )
                    indices = _kmeanspp_rank(_standardize(dynamics), budget, max_anchors, seed)
                    metadata = {"algorithm": "proxy_training_dynamics", "proxy_epochs": int(settings["proxy_epochs"])}
            elif method_key in {"uherding", "uncertainty-herding"}:
                uncertainty = mc_uncertainty(
                    model,
                    dataset,
                    scaler,
                    device,
                    batch_size,
                    seed,
                    int(settings["mc_passes"]),
                )
                augmented = np.concatenate([projected, _standardize(uncertainty[:, None])], axis=1)
                ranked = _farthest_rank(augmented, max(1, len(dataset) - 1), max_anchors)
                if len(ranked) < len(dataset):
                    ranked.extend(index for index in range(len(dataset)) if index not in set(ranked))
                uncertainty_rank = np.empty(len(dataset), dtype=np.int64)
                uncertainty_rank[np.argsort(-uncertainty)] = np.arange(len(dataset))
                diversity_rank = np.empty(len(dataset), dtype=np.int64)
                diversity_rank[np.asarray(ranked)] = np.arange(len(dataset))
                indices = np.argsort(uncertainty_rank + diversity_rank)[:budget].tolist()
                metadata = {"algorithm": "uncertainty_coverage_herding", "mc_passes": int(settings["mc_passes"])}
            else:
                raise ValueError(f"Unknown selector: {method}")
    return SelectionResult(
        indices=sorted(int(index) for index in indices),
        elapsed_seconds=timer["seconds"],
        metadata={**metadata, "budget": budget, "ratio": ratio},
    )
