"""Linear solvers used by DREAM's predictor and corrector steps."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dream.operators import DenseJacobian, LinearOperator


@dataclass(frozen=True)
class DenseSolveResult:
    solution: torch.Tensor
    relative_residual: float
    absolute_rms: float
    retained_rank: int
    condition: float
    refinement_passes: int


def live_refined_minimum_norm(
    jacobian: DenseJacobian | torch.Tensor,
    rhs: torch.Tensor,
    *,
    rcond: float | None = None,
    refinement_passes: int = 3,
    relative_tolerance: float = 1e-10,
) -> DenseSolveResult:
    """Solve ``J delta = rhs`` through the residual-space Gram matrix.

    The solution is ``J.T (J J.T)^+ rhs``. Every refinement pass recomputes
    ``rhs - J delta`` with the live rectangular operator; a small residual of
    the squared Gram system alone is never treated as sufficient evidence.
    """

    matrix = jacobian.matrix if isinstance(jacobian, DenseJacobian) else jacobian
    if matrix.ndim != 2 or rhs.numel() != matrix.shape[0]:
        raise ValueError("incompatible Jacobian and right-hand side")
    rhs = rhs.reshape(-1)
    gram = matrix @ matrix.transpose(0, 1)
    gram = 0.5 * (gram + gram.transpose(0, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    largest = float(eigenvalues[-1]) if eigenvalues.numel() else 0.0
    if rcond is None:
        rcond = torch.finfo(matrix.dtype).eps * max(matrix.shape)
    cutoff = largest * float(rcond)
    keep = eigenvalues > cutoff
    if not bool(keep.any()):
        raise RuntimeError("the Jacobian has no numerically retained direction")
    basis = eigenvectors[:, keep]
    values = eigenvalues[keep]

    def gram_solve(vector: torch.Tensor) -> torch.Tensor:
        return basis @ ((basis.transpose(0, 1) @ vector) / values)

    solution = matrix.transpose(0, 1) @ gram_solve(rhs)
    rhs_norm = max(float(torch.linalg.vector_norm(rhs)), torch.finfo(rhs.dtype).tiny)
    used = 0
    for index in range(refinement_passes + 1):
        live_residual = rhs - matrix @ solution
        relative = float(torch.linalg.vector_norm(live_residual)) / rhs_norm
        used = index
        if relative <= relative_tolerance or index == refinement_passes:
            break
        solution = solution + matrix.transpose(0, 1) @ gram_solve(live_residual)
    live_residual = rhs - matrix @ solution
    smallest = float(values.min())
    return DenseSolveResult(
        solution=solution,
        relative_residual=float(torch.linalg.vector_norm(live_residual)) / rhs_norm,
        absolute_rms=float(torch.sqrt(torch.mean(live_residual.square()))),
        retained_rank=int(keep.sum()),
        condition=(largest / smallest) ** 0.5,
        refinement_passes=used,
    )


@dataclass(frozen=True)
class LSQRResult:
    solution: torch.Tensor
    relative_residual: float
    iterations: int
    converged: bool
    residual_history: tuple[float, ...]


def lsqr(
    operator: LinearOperator,
    rhs: torch.Tensor,
    *,
    iterations: int = 128,
    relative_tolerance: float = 1e-8,
) -> LSQRResult:
    """Rectangular Golub--Kahan LSQR using only JVPs and VJPs.

    This deliberately reports the independently applied live residual instead
    of trusting only the scalar recurrence estimate.
    """

    rhs = rhs.reshape(-1)
    rhs_norm = float(torch.linalg.vector_norm(rhs))
    if rhs_norm == 0.0:
        zero = torch.zeros(operator.columns, device=rhs.device, dtype=rhs.dtype)
        return LSQRResult(zero, 0.0, 0, True, (0.0,))
    u = rhs / rhs_norm
    v_raw = operator.adjoint(u)
    alpha = torch.linalg.vector_norm(v_raw)
    if float(alpha) == 0.0:
        zero = torch.zeros(operator.columns, device=rhs.device, dtype=rhs.dtype)
        return LSQRResult(zero, 1.0, 0, False, (1.0,))
    v = v_raw / alpha
    w = v.clone()
    x = torch.zeros_like(v)
    phi_bar = torch.as_tensor(rhs_norm, device=rhs.device, dtype=rhs.dtype)
    rho_bar = alpha
    history: list[float] = []
    converged = False
    completed = 0
    for completed in range(1, int(iterations) + 1):
        u_raw = operator.forward(v) - alpha * u
        beta = torch.linalg.vector_norm(u_raw)
        u = u_raw / beta if float(beta) > 0.0 else torch.zeros_like(u_raw)
        v_raw = operator.adjoint(u) - beta * v
        alpha = torch.linalg.vector_norm(v_raw)
        v = v_raw / alpha if float(alpha) > 0.0 else torch.zeros_like(v_raw)

        rho = torch.sqrt(rho_bar.square() + beta.square())
        c = rho_bar / rho
        s = beta / rho
        theta = s * alpha
        rho_bar = -c * alpha
        phi = c * phi_bar
        phi_bar = s * phi_bar
        x = x + (phi / rho) * w
        w = v - (theta / rho) * w

        live = rhs - operator.forward(x)
        relative = float(torch.linalg.vector_norm(live)) / rhs_norm
        history.append(relative)
        if relative <= relative_tolerance:
            converged = True
            break
    return LSQRResult(x, history[-1], completed, converged, tuple(history))

