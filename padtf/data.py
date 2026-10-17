from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .config import ExperimentConfig


@dataclass(frozen=True)
class StandardScaler:
    mean: float
    std: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_tensor(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean


def load_values(path: str | Path) -> np.ndarray:
    source = Path(path)
    if source.suffix == ".npz":
        with np.load(source) as archive:
            values = archive["data"]
        if values.ndim == 3:
            values = values[..., 0]
    elif source.suffix in {".csv", ".txt"}:
        frame = pd.read_csv(source, index_col=0)
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        if numeric.isna().all(axis=None):
            numeric = pd.read_csv(source, header=None).apply(pd.to_numeric, errors="coerce")
        values = numeric.to_numpy()
    else:
        raise ValueError(f"Unsupported traffic data file: {source}")
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected [time, node] values at {source}, got {values.shape}")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return values


class WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        values: np.ndarray,
        scaler: StandardScaler,
        split_start: int,
        split_end: int,
        history: int,
        horizon: int,
        start_weekday: int,
        interval_minutes: int,
        normalized_values: np.ndarray | None = None,
        time_of_day: np.ndarray | None = None,
        day_of_week: np.ndarray | None = None,
    ) -> None:
        self.values = values
        self.scaler = scaler
        self.split_start = split_start
        self.split_end = split_end
        self.history = history
        self.horizon = horizon
        self.start_weekday = start_weekday
        self.slots_per_day = 24 * 60 // interval_minutes
        self.normalized_values = (
            scaler.transform(values).astype(np.float32, copy=False)
            if normalized_values is None
            else normalized_values
        )
        positions = np.arange(len(values), dtype=np.int64)
        self.time_of_day = positions % self.slots_per_day + 1 if time_of_day is None else time_of_day
        self.day_of_week = (
            (start_weekday - 1 + positions // self.slots_per_day) % 7 + 1
            if day_of_week is None
            else day_of_week
        )
        self.length = max(0, split_end - split_start - history - horizon + 1)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        if item < 0 or item >= self.length:
            raise IndexError(item)
        start = self.split_start + item
        middle = start + self.history
        end = middle + self.horizon
        normalized_x = self.normalized_values[start:middle]
        normalized_y = self.normalized_values[middle:end]
        time_of_day = self.time_of_day[start:middle]
        day_of_week = self.day_of_week[start:middle]
        nodes = self.values.shape[1]
        x = np.empty((self.history, nodes, 3), dtype=np.float32)
        x[..., 0] = normalized_x
        x[..., 1] = np.broadcast_to(time_of_day[:, None], (self.history, nodes))
        x[..., 2] = np.broadcast_to(day_of_week[:, None], (self.history, nodes))
        y = normalized_y[..., None].astype(np.float32, copy=False)
        return torch.from_numpy(x), torch.from_numpy(y)

    def absolute_start(self, item: int) -> int:
        return self.split_start + item


@dataclass
class DatasetBundle:
    name: str
    values: np.ndarray
    scaler: StandardScaler
    train: WindowDataset
    val: WindowDataset
    test: WindowDataset


def build_dataset_bundle(config: ExperimentConfig, name: str) -> DatasetBundle:
    spec = config.dataset(name)
    values = load_values(spec["values_path"])
    expected_nodes = int(spec["nodes"])
    if values.shape[1] != expected_nodes:
        raise ValueError(f"{name}: expected {expected_nodes} nodes, found {values.shape[1]}")
    data_config = config.section("data")
    total = values.shape[0]
    test_size = int(total * float(data_config["test_ratio"]))
    val_size = int(total * float(data_config["val_ratio"]))
    train_end = total - val_size - test_size
    val_end = total - test_size
    train_values = values[:train_end]
    scaler = StandardScaler(float(train_values.mean()), max(float(train_values.std()), 1.0e-6))
    normalized_values = scaler.transform(values).astype(np.float32, copy=False)
    positions = np.arange(total, dtype=np.int64)
    slots_per_day = 24 * 60 // int(data_config["interval_minutes"])
    time_of_day = positions % slots_per_day + 1
    day_of_week = (int(spec["start_weekday"]) - 1 + positions // slots_per_day) % 7 + 1
    common = {
        "values": values,
        "scaler": scaler,
        "history": int(data_config["history"]),
        "horizon": int(data_config["horizon"]),
        "start_weekday": int(spec["start_weekday"]),
        "interval_minutes": int(data_config["interval_minutes"]),
        "normalized_values": normalized_values,
        "time_of_day": time_of_day,
        "day_of_week": day_of_week,
    }
    train = WindowDataset(split_start=0, split_end=train_end, **common)
    val = WindowDataset(split_start=train_end, split_end=val_end, **common)
    test = WindowDataset(split_start=val_end, split_end=total, **common)
    return DatasetBundle(name=name, values=values, scaler=scaler, train=train, val=val, test=test)


def batch_size_for(config: ExperimentConfig, dataset_name: str) -> int:
    training = config.section("training")
    overrides = training.get("batch_size_overrides", {})
    return int(overrides.get(dataset_name, training["batch_size"]))


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    indices: Sequence[int] | None = None,
) -> DataLoader:
    selected: Dataset = Subset(dataset, list(indices)) if indices is not None else dataset
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        selected,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def stratified_indices(dataset: WindowDataset, ratio: float, seed: int, strata: int = 16) -> list[int]:
    count = max(1, min(len(dataset), int(math.ceil(len(dataset) * ratio))))
    rng = np.random.default_rng(seed)
    groups: list[list[int]] = [[] for _ in range(strata)]
    for index in range(len(dataset)):
        absolute = dataset.absolute_start(index)
        slot = absolute % dataset.slots_per_day
        weekday = (dataset.start_weekday - 1 + absolute // dataset.slots_per_day) % 7
        time_group = min(7, slot * 8 // dataset.slots_per_day)
        weekend_group = int(weekday >= 5)
        groups[weekend_group * 8 + time_group].append(index)
    selected: list[int] = []
    nonempty = [group for group in groups if group]
    for group in nonempty:
        allocation = max(1, round(count * len(group) / len(dataset)))
        allocation = min(allocation, len(group))
        selected.extend(rng.choice(group, size=allocation, replace=False).tolist())
    selected = list(dict.fromkeys(selected))
    if len(selected) < count:
        remaining = np.setdiff1d(np.arange(len(dataset)), np.asarray(selected), assume_unique=False)
        selected.extend(rng.choice(remaining, size=count - len(selected), replace=False).tolist())
    elif len(selected) > count:
        selected = rng.choice(selected, size=count, replace=False).tolist()
    return sorted(int(index) for index in selected)


def _stratified_sample_from_pool(
    dataset: WindowDataset,
    pool: Sequence[int],
    count: int,
    seed: int,
    strata: int = 16,
) -> list[int]:
    candidates = [int(index) for index in pool]
    if count < 1 or count > len(candidates):
        raise ValueError(f"Invalid sample count {count} for pool of size {len(candidates)}")
    rng = np.random.default_rng(seed)
    groups: list[list[int]] = [[] for _ in range(strata)]
    for index in candidates:
        absolute = dataset.absolute_start(index)
        slot = absolute % dataset.slots_per_day
        weekday = (dataset.start_weekday - 1 + absolute // dataset.slots_per_day) % 7
        time_group = min(7, slot * 8 // dataset.slots_per_day)
        weekend_group = int(weekday >= 5)
        groups[weekend_group * 8 + time_group].append(index)
    nonempty = [(group_index, group) for group_index, group in enumerate(groups) if group]
    exact = {
        group_index: count * len(group) / len(candidates)
        for group_index, group in nonempty
    }
    allocations = {
        group_index: min(len(group), int(math.floor(exact[group_index])))
        for group_index, group in nonempty
    }
    remaining = count - sum(allocations.values())
    tie_breakers = {group_index: float(rng.random()) for group_index, _ in nonempty}
    order = sorted(
        nonempty,
        key=lambda item: (exact[item[0]] - allocations[item[0]], tie_breakers[item[0]]),
        reverse=True,
    )
    while remaining:
        progressed = False
        for group_index, group in order:
            if allocations[group_index] >= len(group):
                continue
            allocations[group_index] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise RuntimeError("Unable to allocate the requested stratified sample")
    selected: list[int] = []
    for group_index, group in nonempty:
        allocation = allocations[group_index]
        if allocation:
            selected.extend(rng.choice(group, size=allocation, replace=False).tolist())
    return sorted(int(index) for index in selected)


def blocked_stratified_probe_split(
    dataset: WindowDataset,
    ratio: float,
    train_fraction: float,
    seed: int,
    minimum_gap: int,
) -> tuple[list[int], list[int]]:
    if not 0.0 < ratio <= 1.0:
        raise ValueError("Probe ratio must be in (0, 1]")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("Probe train fraction must be in (0, 1)")
    if minimum_gap < dataset.history + dataset.horizon:
        raise ValueError("Probe block gap must cover the complete input-target window")
    total_count = max(2, min(len(dataset), int(math.ceil(len(dataset) * ratio))))
    train_count = min(total_count - 1, max(1, int(round(total_count * train_fraction))))
    val_count = total_count - train_count
    available = len(dataset) - minimum_gap
    if available < total_count:
        raise ValueError("Dataset is too short for the requested blocked probe split")
    train_stop = int(round(available * train_fraction))
    train_stop = min(available - val_count, max(train_count, train_stop))
    val_start = train_stop + minimum_gap
    train = _stratified_sample_from_pool(dataset, range(train_stop), train_count, seed)
    validation = _stratified_sample_from_pool(dataset, range(val_start, len(dataset)), val_count, seed + 1)
    observed_gap = min(validation) - max(train)
    if observed_gap < minimum_gap:
        raise RuntimeError(f"Blocked probe split gap {observed_gap} is below {minimum_gap}")
    return train, validation


def split_indices(indices: Iterable[int], train_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    values = np.asarray(list(indices), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    boundary = min(len(values) - 1, max(1, int(round(len(values) * train_fraction))))
    return sorted(values[:boundary].tolist()), sorted(values[boundary:].tolist())
