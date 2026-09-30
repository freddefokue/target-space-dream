"""Generalized Gauss--Newton products, damped conjugate gradient and Levenberg--Marquardt damping.

The generalized Gauss--Newton (GGN) matrix of a loss ``L(f(theta))`` is ``J.T H J`` with ``J`` the
Jacobian of the model outputs and ``H`` the Hessian of the loss with respect to the outputs. For
the mean softmax KL over a batch, ``H`` is block diagonal with blocks
``(diag(p) - p p.T) / batch``. The products never materialize ``J``.

Three exact ways to apply ``J`` are provided because their costs differ a lot between models:

``jvp_vjp``
    forward-mode ``torch.func.jvp`` for ``J v`` (recomputes the primal every call) and a
    reverse-mode pullback built once per point for ``J.T u``.
``linearize``
    ``torch.func.linearize`` once per point, then its linear JVP function for every ``J v``.
``double_vjp``
    reverse-over-reverse: ``J v`` is the VJP of the linear map ``u -> J.T u`` with cotangent
    ``v``. One forward and one differentiable backward per point, then two backward-like passes
    per product.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch.func import jvp, linearize, vjp

TensorFunction = Callable[[torch.Tensor], torch.Tensor]
OutputMetric = Callable[[torch.Tensor], torch.Tensor]
METHODS = ("jvp_vjp", "linearize", "double_vjp")


def softmax_output_metric(logits: torch.Tensor) -> OutputMetric:
    """Return ``u -> H u`` for the mean softmax KL, ``H = blockdiag(diag(p) - p p.T) / batch``."""

    probs = torch.softmax(logits.detach(), dim=-1)
    batch = logits.shape[0]

    def apply(direction: torch.Tensor) -> torch.Tensor:
        weighted = probs * direction
        return (weighted - probs * weighted.sum(-1, keepdim=True)) / batch

    return apply


def identity_output_metric(direction: torch.Tensor) -> torch.Tensor:
    return direction


class JacobianProducts:
    """``J v`` and ``J.T u`` of ``function`` at a fixed point, by one of :data:`METHODS`."""

    def __init__(self, function: TensorFunction, point: torch.Tensor, method: str) -> None:
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
        self.method = method
        self.function = function
        self.point = point.detach()
        if method == "double_vjp":
            self._leaf = self.point.clone().requires_grad_(True)
            with torch.enable_grad():
                output = function(self._leaf)
                self._dummy = torch.zeros_like(output, requires_grad=True)
                (self._transpose_dummy,) = torch.autograd.grad(
                    output, self._leaf, self._dummy, create_graph=True)
            self.output = output.detach()
            self._output_graph = output
        else:
            if method == "linearize":
                _, self._jvp = linearize(function, self.point)
            output, self._pullback = vjp(function, self.point)
            self.output = output.detach()

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        """``J v``."""

        if self.method == "jvp_vjp":
            return jvp(self.function, (self.point,), (vector,))[1]
        if self.method == "linearize":
            return self._jvp(vector)
        (image,) = torch.autograd.grad(self._transpose_dummy, self._dummy, vector,
                                       retain_graph=True)
        return image

    def adjoint(self, cotangent: torch.Tensor) -> torch.Tensor:
        """``J.T u``."""

        if self.method == "double_vjp":
            (pulled,) = torch.autograd.grad(self._output_graph, self._leaf, cotangent,
                                            retain_graph=True)
            return pulled
        return self._pullback(cotangent)[0]


class GaussNewtonOperator:
    """``v -> J.T H J v`` with ``H`` given by an output metric."""

    def __init__(self, products: JacobianProducts, output_metric: OutputMetric) -> None:
        self.products = products
        self.output_metric = output_metric
        self.products_applied = 0

    def __call__(self, vector: torch.Tensor) -> torch.Tensor:
        self.products_applied += 1
        return self.products.adjoint(self.output_metric(self.products.forward(vector)))


@dataclass(frozen=True)
class CGResult:
    solution: torch.Tensor
    residual: torch.Tensor
    iterations: int
    products: int
    relative_residual: float
    stopped: str


def damped_cg(
    operator: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    damping: float,
    *,
    iterations: int,
    initial: torch.Tensor | None = None,
    relative_tolerance: float = 0.0,
) -> CGResult:
    """Conjugate gradient on ``(A + damping I) x = rhs`` for a symmetric PSD ``A``.

    ``initial`` warm-starts the iteration (one extra operator product). The returned residual is
    the recursively updated ``rhs - (A + damping I) x``; callers that need ``A x`` can recover it
    as ``rhs - residual - damping * x`` without another product.
    """

    def system(vector: torch.Tensor) -> torch.Tensor:
        return operator(vector) + damping * vector

    products = 0
    rhs_norm = float(torch.linalg.vector_norm(rhs))
    if initial is None or not bool(initial.any()):
        x = torch.zeros_like(rhs)
        residual = rhs.clone()
    else:
        x = initial.clone()
        residual = rhs - system(x)
        products += 1
    direction = residual.clone()
    rr = torch.dot(residual, residual)
    stopped = "iterations"
    completed = 0
    for completed in range(1, iterations + 1):
        if rhs_norm == 0.0 or float(rr) ** 0.5 <= relative_tolerance * rhs_norm:
            completed -= 1
            stopped = "tolerance"
            break
        image = system(direction)
        products += 1
        curvature = torch.dot(direction, image)
        if not bool(torch.isfinite(curvature)) or float(curvature) <= 0.0:
            completed -= 1
            stopped = "curvature"
            break
        step = rr / curvature
        x = x + step * direction
        residual = residual - step * image
        rr_next = torch.dot(residual, residual)
        direction = residual + (rr_next / rr) * direction
        rr = rr_next
    else:
        if relative_tolerance > 0.0 and float(rr) ** 0.5 <= relative_tolerance * rhs_norm:
            stopped = "tolerance"
    relative = float(rr) ** 0.5 / rhs_norm if rhs_norm > 0.0 else 0.0
    return CGResult(x, residual, completed, products, relative, stopped)


@dataclass
class LevenbergMarquardt:
    """Trust-ratio damping control (brief §6): shrink when the model is good, grow when bad."""

    damping: float
    lower: float = 0.25
    upper: float = 0.75
    increase: float = 1.5
    decrease: float = 2.0 / 3.0
    minimum: float = 1e-8
    maximum: float = 1e8

    def update(self, ratio: float) -> float:
        if not ratio == ratio or ratio < self.lower:  # NaN counts as a bad step
            self.damping = min(self.damping * self.increase, self.maximum)
        elif ratio > self.upper:
            self.damping = max(self.damping * self.decrease, self.minimum)
        return self.damping


def reduction_ratio(actual_change: float, gradient_dot_step: float,
                    step_curvature: float) -> float:
    """``rho = [L(theta + delta) - L(theta)] / [g.T delta + 0.5 delta.T G delta]``."""

    predicted = gradient_dot_step + 0.5 * step_curvature
    if predicted >= 0.0:
        return float("-inf")
    return actual_change / predicted
