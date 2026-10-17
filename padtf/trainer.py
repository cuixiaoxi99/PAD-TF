from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import numpy as np
from torch.utils.data import DataLoader

from .data import StandardScaler
from .metrics import MetricAccumulator, info_nce_loss, masked_mae_loss
from .model import FlashSTTransferModel
from .utils import atomic_torch_save, elapsed_timer


@dataclass
class FitResult:
    elapsed_seconds: float
    best_epoch: int
    best_val_mae: float
    epochs_run: int
    history: list[dict[str, float]]


def fit_model_milestones(
    model: FlashSTTransferModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    scaler: StandardScaler,
    device: torch.device,
    milestones: Sequence[int],
    learning_rate: float,
    effective_batch_size: int,
    ssl_weight: float,
    grad_clip: float,
    checkpoint_paths: dict[int, str | Path],
    logger: Any,
) -> dict[int, FitResult]:
    requested = sorted({int(epoch) for epoch in milestones})
    if not requested or requested[0] <= 0:
        raise ValueError("Milestones must contain positive epoch numbers")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, eps=1.0e-8, weight_decay=0.0, amsgrad=False)
    physical_batch_size = int(train_loader.batch_size or effective_batch_size)
    accumulation = max(1, math.ceil(effective_batch_size / physical_batch_size))
    history: list[dict[str, float]] = []
    results: dict[int, FitResult] = {}
    excluded_seconds = 0.0
    model.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for epoch in range(1, requested[-1] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_forecast = 0.0
        batches = 0
        for batch_index, (inputs, targets) in enumerate(train_loader, start=1):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            loss, forecast = _forward_loss(model, inputs, targets, scaler, ssl_weight)
            (loss / accumulation).backward()
            if batch_index % accumulation == 0 or batch_index == len(train_loader):
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total_forecast += forecast.item()
            batches += 1
        val_mae = validation_mae(model, val_loader, scaler, device)
        train_mae = total_forecast / max(1, batches)
        history.append({"epoch": epoch, "train_mae": train_mae, "val_mae": val_mae})
        logger.info("epoch=%d/%d train_mae=%.6f val_mae=%.6f", epoch, requested[-1], train_mae, val_mae)
        if epoch not in requested:
            continue
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_seconds = time.perf_counter() - started - excluded_seconds
        save_started = time.perf_counter()
        checkpoint_path = Path(checkpoint_paths[epoch])
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = copy.deepcopy(model.state_dict())
        torch.save(
            {
                "state_dict": state_dict,
                "best_epoch": epoch,
                "best_val_mae": val_mae,
                "history": list(history),
                "trajectory_milestone": True,
            },
            checkpoint_path,
        )
        results[epoch] = FitResult(
            elapsed_seconds=elapsed_seconds,
            best_epoch=epoch,
            best_val_mae=val_mae,
            epochs_run=epoch,
            history=list(history),
        )
        excluded_seconds += time.perf_counter() - save_started
    return results


def _forward_loss(
    model: FlashSTTransferModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    scaler: StandardScaler,
    ssl_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction, representation = model(inputs)
    forecast = masked_mae_loss(prediction, targets, scaler)
    if representation is None or ssl_weight <= 0:
        return forecast, forecast
    total = forecast + ssl_weight * info_nce_loss(representation)
    return total, forecast


@torch.no_grad()
def validation_mae(
    model: FlashSTTransferModel,
    loader: DataLoader,
    scaler: StandardScaler,
    device: torch.device,
) -> float:
    return validation_scores(model, loader, scaler, device)["mae"]


@torch.no_grad()
def validation_scores(
    model: FlashSTTransferModel,
    loader: DataLoader,
    scaler: StandardScaler,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    absolute_sum = 0.0
    squared_sum = 0.0
    count = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        prediction, _ = model(inputs)
        prediction = scaler.inverse_tensor(prediction)
        targets = scaler.inverse_tensor(targets)
        valid = targets > 0.0
        difference = prediction[valid] - targets[valid]
        absolute_sum += torch.abs(difference).sum().item()
        squared_sum += torch.square(difference).sum().item()
        count += int(valid.sum().item())
    count = max(1, count)
    return {"mae": absolute_sum / count, "rmse": math.sqrt(squared_sum / count)}


def fit_model(
    model: FlashSTTransferModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    scaler: StandardScaler,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    effective_batch_size: int,
    ssl_weight: float,
    grad_clip: float,
    patience: int | None,
    checkpoint_path: str | Path,
    logger: Any,
    monitor: str = "mae",
    restore_best: bool = True,
    validate_each_epoch: bool = True,
) -> FitResult:
    if monitor not in {"mae", "rmse"}:
        raise ValueError(f"Unsupported validation monitor: {monitor}")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, eps=1.0e-8, weight_decay=0.0, amsgrad=False)
    physical_batch_size = int(train_loader.batch_size or effective_batch_size)
    accumulation = max(1, math.ceil(effective_batch_size / physical_batch_size))
    best_val = float("inf")
    best_val_mae = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float]] = []
    model.to(device)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    resume_suffix = "" if monitor == "mae" else f".{monitor}"
    resume_path = checkpoint_path.with_name(f"{checkpoint_path.stem}{resume_suffix}.resume.pt")
    start_epoch = 1
    prior_elapsed = 0.0
    if resume_path.exists():
        resume = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(resume["state_dict"])
        model.to(device)
        optimizer.load_state_dict(resume["optimizer_state"])
        best_val = float(resume.get("best_val_score", resume["best_val_mae"]))
        best_val_mae = float(resume["best_val_mae"])
        best_epoch = int(resume["best_epoch"])
        best_state = resume["best_state"]
        stale = int(resume["stale"])
        history = list(resume["history"])
        start_epoch = int(resume["epoch"]) + 1
        prior_elapsed = float(resume.get("elapsed_seconds", 0.0))
        random.setstate(resume["python_rng_state"])
        np.random.set_state(resume["numpy_rng_state"])
        torch.set_rng_state(resume["torch_rng_state"])
        if device.type == "cuda" and resume.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(resume["cuda_rng_state"])
        if train_loader.generator is not None and resume.get("loader_rng_state") is not None:
            train_loader.generator.set_state(resume["loader_rng_state"])
        logger.info("resume_training epoch=%d/%d checkpoint=%s", start_epoch - 1, epochs, resume_path)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    excluded_seconds = 0.0
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_forecast = 0.0
        batches = 0
        for batch_index, (inputs, targets) in enumerate(train_loader, start=1):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            loss, forecast = _forward_loss(model, inputs, targets, scaler, ssl_weight)
            (loss / accumulation).backward()
            if batch_index % accumulation == 0 or batch_index == len(train_loader):
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total_forecast += forecast.item()
            batches += 1
        should_validate = validate_each_epoch or epoch == epochs
        validation = validation_scores(model, val_loader, scaler, device) if should_validate else None
        val_mae = float("nan") if validation is None else validation["mae"]
        val_rmse = float("nan") if validation is None else validation["rmse"]
        val_score = float("inf") if validation is None else validation[monitor]
        train_mae = total_forecast / max(1, batches)
        history.append(
            {"epoch": epoch, "train_mae": train_mae, "val_mae": val_mae, "val_rmse": val_rmse}
        )
        logger.info(
            "epoch=%d/%d train_mae=%.6f val_mae=%.6f val_rmse=%.6f monitor=%s",
            epoch,
            epochs,
            train_mae,
            val_mae,
            val_rmse,
            monitor,
        )
        if validation is not None and val_score < best_val:
            best_val = val_score
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        elif validation is not None:
            stale += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_seconds = prior_elapsed + time.perf_counter() - started - excluded_seconds
        save_started = time.perf_counter()
        atomic_torch_save(
            resume_path,
            {
                "state_dict": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_state": best_state,
                "best_epoch": best_epoch,
                "best_val_mae": best_val_mae,
                "best_val_score": best_val,
                "monitor": monitor,
                "stale": stale,
                "history": history,
                "epoch": epoch,
                "elapsed_seconds": elapsed_seconds,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "loader_rng_state": train_loader.generator.get_state() if train_loader.generator is not None else None,
            },
        )
        excluded_seconds += time.perf_counter() - save_started
        if patience is not None and validation is not None and stale >= patience:
            logger.info("early_stop epoch=%d patience=%d", epoch, patience)
            break
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_elapsed = prior_elapsed + time.perf_counter() - started - excluded_seconds
    validation_best_epoch = best_epoch
    validation_best_val_mae = best_val_mae
    if restore_best:
        selected_state = best_state
        selected_epoch = best_epoch
        selected_val_mae = best_val_mae
    else:
        selected_state = copy.deepcopy(model.state_dict())
        selected_epoch = int(history[-1]["epoch"])
        selected_val_mae = float(history[-1]["val_mae"])
    model.load_state_dict(selected_state)
    atomic_torch_save(
        checkpoint_path,
        {
            "state_dict": selected_state,
            "best_epoch": selected_epoch,
            "best_val_mae": selected_val_mae,
            "validation_best_epoch": validation_best_epoch,
            "validation_best_val_mae": validation_best_val_mae,
            "best_val_score": best_val,
            "monitor": monitor,
            "restore_best": restore_best,
            "validate_each_epoch": validate_each_epoch,
            "history": history,
        },
    )
    resume_path.unlink(missing_ok=True)
    return FitResult(
        elapsed_seconds=total_elapsed,
        best_epoch=selected_epoch,
        best_val_mae=selected_val_mae,
        epochs_run=len(history),
        history=history,
    )


def fit_model_time_budget(
    model: FlashSTTransferModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    scaler: StandardScaler,
    device: torch.device,
    max_seconds: float,
    max_epochs: int,
    learning_rate: float,
    effective_batch_size: int,
    ssl_weight: float,
    grad_clip: float,
    checkpoint_path: str | Path,
    logger: Any,
) -> FitResult:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, eps=1.0e-8, weight_decay=0.0, amsgrad=False)
    physical_batch_size = int(train_loader.batch_size or effective_batch_size)
    accumulation = max(1, math.ceil(effective_batch_size / physical_batch_size))
    initial_state = copy.deepcopy(model.state_dict())
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = 0
    completed_elapsed = 0.0
    epoch_durations: list[float] = []
    history: list[dict[str, float]] = []
    model.to(device)
    if max_seconds <= 0.0:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(
            checkpoint_path,
            {
                "state_dict": initial_state,
                "best_epoch": 0,
                "best_val_mae": None,
                "history": history,
                "time_budget_seconds": max_seconds,
            },
        )
        return FitResult(
            elapsed_seconds=0.0,
            best_epoch=0,
            best_val_mae=float("nan"),
            epochs_run=0,
            history=history,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        if epoch_durations:
            estimated_next = float(np.median(epoch_durations[-3:])) * 1.05
            if completed_elapsed + estimated_next > max_seconds:
                logger.info(
                    "time_budget_stop_before_epoch epoch=%d elapsed=%.3f estimate=%.3f budget=%.3f",
                    epoch,
                    completed_elapsed,
                    estimated_next,
                    max_seconds,
                )
                break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_forecast = 0.0
        batches = 0
        for batch_index, (inputs, targets) in enumerate(train_loader, start=1):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            loss, forecast = _forward_loss(model, inputs, targets, scaler, ssl_weight)
            (loss / accumulation).backward()
            if batch_index % accumulation == 0 or batch_index == len(train_loader):
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total_forecast += forecast.item()
            batches += 1
        val_mae = validation_mae(model, val_loader, scaler, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        epoch_duration = elapsed - completed_elapsed
        if elapsed > max_seconds:
            logger.info(
                "time_budget_discard_epoch epoch=%d elapsed=%.3f budget=%.3f",
                epoch,
                elapsed,
                max_seconds,
            )
            break
        completed_elapsed = elapsed
        epoch_durations.append(epoch_duration)
        train_mae = total_forecast / max(1, batches)
        history.append({"epoch": epoch, "train_mae": train_mae, "val_mae": val_mae})
        logger.info(
            "epoch=%d/%d train_mae=%.6f val_mae=%.6f elapsed=%.3f budget=%.3f",
            epoch,
            max_epochs,
            train_mae,
            val_mae,
            completed_elapsed,
            max_seconds,
        )
        if val_mae < best_val:
            best_val = val_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    selected_state = initial_state if best_state is None else best_state
    model.load_state_dict(selected_state)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        checkpoint_path,
        {
            "state_dict": selected_state,
            "best_epoch": best_epoch,
            "best_val_mae": None if best_state is None else best_val,
            "history": history,
            "time_budget_seconds": max_seconds,
        },
    )
    return FitResult(
        elapsed_seconds=completed_elapsed,
        best_epoch=best_epoch,
        best_val_mae=float("nan") if best_state is None else best_val,
        epochs_run=len(history),
        history=history,
    )


@torch.no_grad()
def evaluate_model(
    model: FlashSTTransferModel,
    loader: DataLoader,
    scaler: StandardScaler,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    overall = MetricAccumulator()
    horizon_accumulators: list[MetricAccumulator] = []
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        prediction, _ = model(inputs)
        prediction = scaler.inverse_tensor(prediction)
        targets = scaler.inverse_tensor(targets)
        overall.update(prediction, targets)
        if not horizon_accumulators:
            horizon_accumulators = [MetricAccumulator() for _ in range(targets.shape[1])]
        for horizon, accumulator in enumerate(horizon_accumulators):
            accumulator.update(prediction[:, horizon], targets[:, horizon])
    result: dict[str, Any] = overall.result()
    result["horizons"] = {str(index + 1): item.result() for index, item in enumerate(horizon_accumulators)}
    return result


@torch.no_grad()
def persistence_mae(loader: DataLoader, scaler: StandardScaler, device: torch.device) -> float:
    absolute_sum = 0.0
    count = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        last_value = inputs[:, -1:, :, :1].expand(-1, targets.shape[1], -1, -1)
        prediction = scaler.inverse_tensor(last_value)
        real_targets = scaler.inverse_tensor(targets)
        mask = real_targets > 0.0
        absolute_sum += torch.abs(prediction[mask] - real_targets[mask]).sum().item()
        count += int(mask.sum().item())
    return absolute_sum / max(1, count)
