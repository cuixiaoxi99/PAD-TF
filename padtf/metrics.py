from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import StandardScaler


@dataclass
class MetricAccumulator:
    absolute_sum: float = 0.0
    squared_sum: float = 0.0
    percentage_sum: float = 0.0
    valid_count: int = 0
    percentage_count: int = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        valid = target > 0.0
        difference = prediction - target
        self.absolute_sum += torch.abs(difference[valid]).sum().item()
        self.squared_sum += torch.square(difference[valid]).sum().item()
        self.valid_count += int(valid.sum().item())
        percentage_valid = target > 1.0e-3
        self.percentage_sum += (torch.abs(difference[percentage_valid]) / target[percentage_valid]).sum().item()
        self.percentage_count += int(percentage_valid.sum().item())

    def result(self) -> dict[str, float]:
        count = max(1, self.valid_count)
        percentage_count = max(1, self.percentage_count)
        return {
            "mae": self.absolute_sum / count,
            "rmse": float(np.sqrt(self.squared_sum / count)),
            "mape": 100.0 * self.percentage_sum / percentage_count,
        }


def masked_mae_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scaler: StandardScaler,
) -> torch.Tensor:
    real_prediction = scaler.inverse_tensor(prediction)
    real_target = scaler.inverse_tensor(target)
    mask = real_target > 0.0
    if not torch.any(mask):
        return torch.abs(real_prediction - real_target).mean()
    return torch.abs(real_prediction[mask] - real_target[mask]).mean()


def info_nce_loss(representation: torch.Tensor, temperature: float = 0.3) -> torch.Tensor:
    positive_similarity = torch.sum(representation * representation, dim=-1)
    negative_similarity = torch.matmul(representation, representation.transpose(-1, -2))
    positive = torch.exp(positive_similarity / temperature)
    negative = torch.sum(torch.exp(negative_similarity / temperature), dim=-1)
    return torch.mean(-torch.log(positive / (negative + positive + 1.0e-12)))
