import torch

from dream.operators import MatrixFreeJacobian, explicit_jacobian
from dream.solvers import live_refined_minimum_norm, lsqr


def test_dense_minimum_norm_solve() -> None:
    torch.manual_seed(0)
    matrix = torch.randn(5, 9, dtype=torch.float64)
    rhs = torch.randn(5, dtype=torch.float64)
    result = live_refined_minimum_norm(matrix, rhs)
    assert result.relative_residual < 1e-11
    reference = torch.linalg.lstsq(matrix, rhs).solution
    torch.testing.assert_close(result.solution, reference, rtol=1e-9, atol=1e-9)


def test_explicit_and_matrix_free_actions_agree() -> None:
    torch.manual_seed(1)
    design = torch.randn(6, 4, dtype=torch.float64)

    def function(vector: torch.Tensor) -> torch.Tensor:
        return torch.sin(design @ vector)

    point = torch.randn(4, dtype=torch.float64)
    vector = torch.randn(4, dtype=torch.float64)
    cotangent = torch.randn(6, dtype=torch.float64)
    dense = explicit_jacobian(function, point)
    implicit = MatrixFreeJacobian(function, point)
    torch.testing.assert_close(implicit.forward(vector), dense.forward(vector))
    torch.testing.assert_close(implicit.adjoint(cotangent), dense.adjoint(cotangent))
    assert implicit.adjoint_error() < 1e-12


def test_lsqr_uses_rectangular_operator() -> None:
    torch.manual_seed(2)
    design = torch.randn(8, 13, dtype=torch.float64)
    rhs = torch.randn(8, dtype=torch.float64)

    def function(vector: torch.Tensor) -> torch.Tensor:
        return design @ vector

    operator = MatrixFreeJacobian(function, torch.zeros(13, dtype=torch.float64))
    result = lsqr(operator, rhs, iterations=20, relative_tolerance=1e-10)
    assert result.converged
    assert result.relative_residual < 1e-10

