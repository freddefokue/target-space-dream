import pytest
import torch
from torch.func import jacrev

from dream.ggn import (METHODS, GaussNewtonOperator, JacobianProducts, LevenbergMarquardt,
                       damped_cg, identity_output_metric, reduction_ratio,
                       softmax_output_metric)
from dream.solvers import live_refined_minimum_norm


def tiny_network(batch: int = 4, inputs: int = 3, hidden: int = 5, classes: int = 6):
    torch.manual_seed(0)
    x = torch.randn(batch, inputs, dtype=torch.float64)
    shapes = [(inputs, hidden), (hidden,), (hidden, classes)]
    sizes = [torch.Size(shape).numel() for shape in shapes]

    def function(theta: torch.Tensor) -> torch.Tensor:
        w1, b1, w2 = (part.view(shape) for part, shape in zip(theta.split(sizes), shapes))
        return torch.tanh(x @ w1 + b1) @ w2

    theta = torch.randn(sum(sizes), dtype=torch.float64)
    return function, theta


@pytest.mark.parametrize("method", METHODS)
def test_ggn_product_matches_explicit_jt_f_j(method: str) -> None:
    function, theta = tiny_network()
    logits = function(theta)
    batch, classes = logits.shape
    jacobian = jacrev(lambda p: function(p).reshape(-1))(theta)
    probs = torch.softmax(logits, dim=-1)
    blocks = [torch.diag(p) - torch.outer(p, p) for p in probs]
    output_hessian = torch.block_diag(*blocks) / batch
    explicit = jacobian.T @ output_hessian @ jacobian
    operator = GaussNewtonOperator(JacobianProducts(function, theta, method),
                                   softmax_output_metric(logits))
    vector = torch.randn_like(theta)
    torch.testing.assert_close(operator(vector), explicit @ vector, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(operator.products.output, logits)


def test_cg_solves_damped_spd_system() -> None:
    torch.manual_seed(1)
    factor = torch.randn(40, 30, dtype=torch.float64)
    matrix = factor.T @ factor
    rhs = torch.randn(30, dtype=torch.float64)
    damping = 0.1
    result = damped_cg(lambda v: matrix @ v, rhs, damping, iterations=200,
                       relative_tolerance=1e-10)
    live = rhs - (matrix + damping * torch.eye(30, dtype=torch.float64)) @ result.solution
    assert float(torch.linalg.vector_norm(live) / torch.linalg.vector_norm(rhs)) < 1e-8
    # The recursive residual lets callers recover A x without another product.
    recovered = rhs - result.residual - damping * result.solution
    torch.testing.assert_close(recovered, matrix @ result.solution, rtol=1e-6, atol=1e-8)


def test_cg_warm_start_reaches_same_solution() -> None:
    torch.manual_seed(2)
    factor = torch.randn(25, 20, dtype=torch.float64)
    matrix = factor.T @ factor
    rhs = torch.randn(20, dtype=torch.float64)
    cold = damped_cg(lambda v: matrix @ v, rhs, 0.5, iterations=100, relative_tolerance=1e-12)
    warm = damped_cg(lambda v: matrix @ v, rhs, 0.5, iterations=100, relative_tolerance=1e-12,
                     initial=0.9 * cold.solution)
    torch.testing.assert_close(warm.solution, cold.solution, rtol=1e-9, atol=1e-10)
    # With a small iteration budget, starting near the solution leaves a smaller residual.
    short_cold = damped_cg(lambda v: matrix @ v, rhs, 0.5, iterations=3)
    short_warm = damped_cg(lambda v: matrix @ v, rhs, 0.5, iterations=3,
                           initial=0.9 * cold.solution)
    assert short_warm.relative_residual < short_cold.relative_residual
    assert short_warm.products == 4


@pytest.mark.parametrize("method", METHODS)
def test_damped_gauss_newton_limit_is_minimum_norm_step(method: str) -> None:
    torch.manual_seed(3)
    design = torch.randn(5, 12, dtype=torch.float64)

    def function(theta: torch.Tensor) -> torch.Tensor:
        return torch.sin(design @ theta) + 0.1 * (design @ theta) ** 2

    theta = 0.3 * torch.randn(12, dtype=torch.float64)
    target = torch.randn(5, dtype=torch.float64)
    products = JacobianProducts(function, theta, method)
    residual = products.output - target
    gradient = products.adjoint(residual)
    operator = GaussNewtonOperator(products, identity_output_metric)
    step = damped_cg(operator, -gradient, 1e-13, iterations=200, relative_tolerance=1e-15)
    jacobian = jacrev(function)(theta)
    reference = live_refined_minimum_norm(jacobian, -residual).solution
    torch.testing.assert_close(step.solution, reference, rtol=1e-7, atol=1e-9)


def test_levenberg_marquardt_directions() -> None:
    controller = LevenbergMarquardt(1.0)
    assert controller.update(0.1) == pytest.approx(1.5)
    assert controller.update(0.5) == pytest.approx(1.5)
    assert controller.update(0.9) == pytest.approx(1.0)
    assert controller.update(float("nan")) == pytest.approx(1.5)


def test_reduction_ratio_is_one_on_a_quadratic() -> None:
    torch.manual_seed(4)
    factor = torch.randn(8, 6, dtype=torch.float64)
    matrix = factor.T @ factor + torch.eye(6, dtype=torch.float64)
    linear = torch.randn(6, dtype=torch.float64)

    def loss(x: torch.Tensor) -> float:
        return float(0.5 * x @ matrix @ x + linear @ x)

    x = torch.randn(6, dtype=torch.float64)
    gradient = matrix @ x + linear
    step = -0.5 * torch.linalg.solve(matrix, gradient)
    ratio = reduction_ratio(loss(x + step) - loss(x), float(gradient @ step),
                            float(step @ matrix @ step))
    assert ratio == pytest.approx(1.0, rel=1e-10)
