"""Adaptive fresh-J target-space continuation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import torch

from dream.metrics import ResidualMetrics, residual_metrics
from dream.operators import explicit_jacobian
from dream.solvers import live_refined_minimum_norm
from dream.targets import StraightTargetPath


TensorFunction = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class ContinuationConfig:
    initial_step: float = 0.25
    minimum_step: float = 1.0 / 4096.0
    maximum_step: float = 0.25
    growth: float = 2.0
    clean_stages_before_growth: int = 2
    maximum_attempts: int = 512
    correctors_per_attempt: int = 8
    pooled_rms_tolerance: float = 1e-9
    sample_rms_tolerance: float = 3e-9
    linear_relative_tolerance: float = 1e-10
    gram_rcond: float | None = None
    trust_ratio_minimum: float = 0.25
    backtracking_multipliers: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)


@dataclass(frozen=True)
class ContinuationResult:
    parameters: torch.Tensor
    reached_endpoint: bool
    final_t: float
    accepted_stages: int
    attempted_intervals: int
    jacobian_builds: int
    path_length: float
    endpoint_displacement: float
    final_metrics: ResidualMetrics
    history: tuple[dict, ...] = field(repr=False)


def _root_passed(metrics: ResidualMetrics, config: ContinuationConfig) -> bool:
    return bool(
        metrics.finite
        and metrics.pooled_rms <= config.pooled_rms_tolerance
        and metrics.maximum_sample_rms <= config.sample_rms_tolerance
    )


def _correct_to_target(
    output: TensorFunction,
    source: torch.Tensor,
    target: torch.Tensor,
    config: ContinuationConfig,
) -> tuple[torch.Tensor, bool, list[dict], int]:
    """Apply fresh-J Gauss--Newton corrections to one fixed target."""

    parameters = source.detach().clone()
    rows: list[dict] = []
    builds = 0
    for outer in range(config.correctors_per_attempt):
        actual = output(parameters)
        metrics = residual_metrics(actual, target)
        if _root_passed(metrics, config):
            return parameters, True, rows, builds
        residual = (actual - target).reshape(-1).detach()
        jacobian = explicit_jacobian(output, parameters)
        builds += 1
        solve = live_refined_minimum_norm(
            jacobian,
            -residual,
            rcond=config.gram_rcond,
            relative_tolerance=config.linear_relative_tolerance,
        )
        correction = solve.solution
        source_energy = float(torch.dot(residual, residual))
        linear_image = jacobian.forward(correction)
        candidates: list[dict] = []
        selected: tuple[torch.Tensor, ResidualMetrics, float, float] | None = None
        for multiplier in config.backtracking_multipliers:
            candidate = parameters + multiplier * correction
            candidate_metrics = residual_metrics(output(candidate), target)
            predicted = residual + multiplier * linear_image
            predicted_energy = float(torch.dot(predicted, predicted))
            actual_energy = candidate_metrics.pooled_rms**2 * residual.numel()
            denominator = source_energy - predicted_energy
            trust = (
                (source_energy - actual_energy) / denominator
                if denominator > 0.0
                else float("-inf")
            )
            eligible = bool(
                candidate_metrics.finite
                and candidate_metrics.pooled_rms < metrics.pooled_rms
                and trust >= config.trust_ratio_minimum
            )
            candidates.append(
                {
                    "multiplier": multiplier,
                    "eligible": eligible,
                    "trust_ratio": trust,
                    "metrics": asdict(candidate_metrics),
                }
            )
            if eligible:
                selected = candidate, candidate_metrics, trust, multiplier
                break
        row = {
            "outer": outer,
            "source_metrics": asdict(metrics),
            "linear_relative_residual": solve.relative_residual,
            "linear_absolute_rms": solve.absolute_rms,
            "retained_rank": solve.retained_rank,
            "jacobian_condition": solve.condition,
            "correction_l2": float(torch.linalg.vector_norm(correction)),
            "candidates": candidates,
        }
        rows.append(row)
        if selected is None:
            row["status"] = "no_trusted_step"
            return parameters, False, rows, builds
        parameters, accepted_metrics, trust, multiplier = selected
        row.update(
            {
                "status": "accepted",
                "selected_multiplier": multiplier,
                "trust_ratio": trust,
                "endpoint_metrics": asdict(accepted_metrics),
            }
        )
    passed = _root_passed(residual_metrics(output(parameters), target), config)
    return parameters, passed, rows, builds


def track_target_path(
    output: TensorFunction,
    initial_parameters: torch.Tensor,
    path: StraightTargetPath,
    config: ContinuationConfig | None = None,
) -> ContinuationResult:
    """Track exact local roots from ``path.at(0)`` to ``path.at(1)``.

    A failed interval restores the source parameters and halves the target-space
    step. A successful interval is accepted only after replaying the nonlinear
    model against the unchanged primal root gates.
    """

    config = config or ContinuationConfig()
    parameters = initial_parameters.detach().clone()
    initial_metrics = residual_metrics(output(parameters), path.at(0.0))
    if not _root_passed(initial_metrics, config):
        raise ValueError("initial_parameters are not an exact root at t=0")
    t = 0.0
    step = min(config.initial_step, config.maximum_step)
    clean_stages = 0
    accepted = 0
    builds = 0
    path_length = 0.0
    history: list[dict] = []
    while t < 1.0 and len(history) < config.maximum_attempts:
        dt = min(step, 1.0 - t)
        target_t = t + dt
        source = parameters.detach().clone()
        candidate, passed, correctors, used_builds = _correct_to_target(
            output, source, path.at(target_t), config
        )
        builds += used_builds
        attempt = {
            "source_t": t,
            "target_t": target_t,
            "dt": dt,
            "success": passed,
            "correctors": correctors,
        }
        history.append(attempt)
        if passed:
            displacement = float(torch.linalg.vector_norm(candidate - source))
            parameters = candidate.detach()
            t = target_t
            accepted += 1
            path_length += displacement
            attempt["inter_root_l2"] = displacement
            clean_stages += 1
            if clean_stages >= config.clean_stages_before_growth:
                step = min(step * config.growth, config.maximum_step)
                clean_stages = 0
        else:
            parameters = source
            clean_stages = 0
            step *= 0.5
            if step < config.minimum_step:
                break
    final_metrics = residual_metrics(output(parameters), path.at(t))
    reached = bool(t >= 1.0 - 1e-15 and _root_passed(final_metrics, config))
    return ContinuationResult(
        parameters=parameters,
        reached_endpoint=reached,
        final_t=t,
        accepted_stages=accepted,
        attempted_intervals=len(history),
        jacobian_builds=builds,
        path_length=path_length,
        endpoint_displacement=float(torch.linalg.vector_norm(parameters - initial_parameters)),
        final_metrics=final_metrics,
        history=tuple(history),
    )

