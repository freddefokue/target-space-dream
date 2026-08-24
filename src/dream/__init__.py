"""DREAM: target-space continuation for high-fidelity distillation."""

from dream.continuation import (
    ContinuationConfig,
    ContinuationResult,
    track_target_path,
)
from dream.metrics import ResidualMetrics, residual_metrics
from dream.operators import DenseJacobian, MatrixFreeJacobian, explicit_jacobian
from dream.parameters import ParameterSpec
from dream.solvers import DenseSolveResult, LSQRResult, live_refined_minimum_norm, lsqr
from dream.targets import StraightTargetPath, centered_logits, helmert_contrast

__all__ = [
    "ContinuationConfig",
    "ContinuationResult",
    "DenseJacobian",
    "DenseSolveResult",
    "LSQRResult",
    "MatrixFreeJacobian",
    "ParameterSpec",
    "ResidualMetrics",
    "StraightTargetPath",
    "centered_logits",
    "explicit_jacobian",
    "helmert_contrast",
    "live_refined_minimum_norm",
    "lsqr",
    "residual_metrics",
    "track_target_path",
]

