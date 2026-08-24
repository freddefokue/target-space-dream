"""Dense and matrix-free access to a model's parameter Jacobian."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch
from torch.func import jacrev, jvp, vjp


TensorFunction = Callable[[torch.Tensor], torch.Tensor]


class LinearOperator(Protocol):
    rows: int
    columns: int

    def forward(self, vector: torch.Tensor) -> torch.Tensor: ...

    def adjoint(self, vector: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class DenseJacobian:
    matrix: torch.Tensor

    @property
    def rows(self) -> int:
        return self.matrix.shape[0]

    @property
    def columns(self) -> int:
        return self.matrix.shape[1]

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        return self.matrix @ vector

    def adjoint(self, vector: torch.Tensor) -> torch.Tensor:
        return self.matrix.transpose(0, 1) @ vector


def explicit_jacobian(function: TensorFunction, point: torch.Tensor) -> DenseJacobian:
    """Materialize ``d vec(function(point)) / d point``.

    This is the reference solver path for small residual spaces. It is exact up
    to floating-point arithmetic but scales as outputs times parameters.
    """

    def flattened(candidate: torch.Tensor) -> torch.Tensor:
        return function(candidate).reshape(-1)

    matrix = jacrev(flattened)(point)
    return DenseJacobian(matrix.reshape(-1, point.numel()).detach())


class MatrixFreeJacobian:
    """JVP/VJP access to a Jacobian without materializing it."""

    def __init__(self, function: TensorFunction, point: torch.Tensor) -> None:
        self.function = function
        self.point = point.detach()
        with torch.no_grad():
            output = function(self.point)
        self.output_shape = output.shape
        self.rows = output.numel()
        self.columns = point.numel()

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        _, image = jvp(self.function, (self.point,), (vector,))
        return image.reshape(-1)

    def adjoint(self, vector: torch.Tensor) -> torch.Tensor:
        _, pullback = vjp(self.function, self.point)
        return pullback(vector.view(self.output_shape))[0].reshape(-1)

    def adjoint_error(self, seed: int = 0) -> float:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.randn(self.columns, generator=generator).to(self.point)
        y = torch.randn(self.rows, generator=generator).to(self.point)
        lhs = torch.dot(self.forward(x), y)
        rhs = torch.dot(x, self.adjoint(y))
        denominator = max(float(lhs.abs()), float(rhs.abs()), torch.finfo(lhs.dtype).tiny)
        return float((lhs - rhs).abs()) / denominator

