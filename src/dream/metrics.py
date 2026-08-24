"""Primal residual metrics used to accept or reject continuation roots."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ResidualMetrics:
    pooled_rms: float
    maximum_sample_rms: float
    finite: bool


def residual_metrics(actual: torch.Tensor, target: torch.Tensor) -> ResidualMetrics:
    """Measure RMS globally and per first-axis sample."""

    if actual.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(actual.shape)} != {tuple(target.shape)}")
    residual = actual - target
    pooled = torch.sqrt(torch.mean(residual.square()))
    if residual.ndim == 0:
        per_sample = residual.abs().reshape(1)
    elif residual.ndim == 1:
        per_sample = residual.abs()
    else:
        per_sample = torch.linalg.vector_norm(residual.flatten(1), dim=1) / math.sqrt(
            residual[0].numel()
        )
    return ResidualMetrics(
        pooled_rms=float(pooled.detach().item()),
        maximum_sample_rms=float(per_sample.max().detach().item()),
        finite=bool(torch.isfinite(residual).all().detach().item()),
    )

