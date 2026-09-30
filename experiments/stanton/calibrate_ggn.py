#!/usr/bin/env python3
"""Time the three exact GGN-product implementations and add them to the calibration.

For each method and curvature batch size this measures, relative to a no-grad forward at the same
batch size: the per-point setup (``ggn_setup_<method>``) and one GGN-vector product
(``ggn_product_<method>``). Timing uses CUDA-graph replay where capture works (Amendment 2) and
falls back to eager timing otherwise; the mode used is recorded. Every method is first checked
against the others for agreement. The cheapest exact method (setup + K products at the
preregistered K values) is recorded as ``ggn_method``. ``B_full`` and existing ratios are left
unchanged.
"""

from __future__ import annotations

import argparse
import time

import torch
from torch.func import functional_call

from calibrate import graphed, median_ms
from common import CALIBRATION, git_commit, load_calibration, set_numerics, write_json
from dream.ggn import METHODS, GaussNewtonOperator, JacobianProducts, softmax_output_metric
from dream.models import ResNet20GN1
from dream.parameters import ParameterSpec


def build(batch: int):
    torch.manual_seed(0)
    model = ResNet20GN1().cuda()
    params = {name: value.detach() for name, value in model.named_parameters()}
    spec = ParameterSpec.from_params(params)
    theta = spec.flatten(params)
    images = torch.randn(batch, 3, 32, 32, device="cuda")

    def function(flat: torch.Tensor) -> torch.Tensor:
        return functional_call(model, spec.unflatten(flat), (images,))

    return model, function, theta, images


def timed(function, repeats: int) -> tuple[float, str]:
    try:
        return median_ms(graphed(function), repeats), "cuda_graph"
    except Exception as error:  # capture unsupported for this op sequence
        torch.cuda.synchronize()
        print(f"    graph capture failed ({type(error).__name__}: {str(error)[:80]}); eager",
              flush=True)
        return median_ms(function, repeats), "eager"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[64, 128, 512, 1024])
    parser.add_argument("--repeats", type=int, default=30)
    arguments = parser.parse_args()
    set_numerics()
    calibration = load_calibration()
    ratios = calibration["ratios"]
    modes: dict[str, dict[str, str]] = {}
    agreement: dict[str, float] = {}

    for batch in arguments.batches:
        model, function, theta, images = build(batch)

        def forward_nograd():
            with torch.no_grad():
                model(images)

        forward_ms, _ = timed(forward_nograd, arguments.repeats)
        ratios.setdefault("fwd", {})[str(batch)] = 1.0
        vector = torch.randn_like(theta)
        results = {}
        for method in METHODS:
            products = JacobianProducts(function, theta, method)
            operator = GaussNewtonOperator(products, softmax_output_metric(products.output))
            results[method] = operator(vector).detach().clone()

            def setup():
                JacobianProducts(function, theta, method)

            def product():
                operator(vector)

            setup_ms, setup_mode = timed(setup, arguments.repeats)
            product_ms, product_mode = timed(product, arguments.repeats)
            ratios.setdefault(f"ggn_setup_{method}", {})[str(batch)] = setup_ms / forward_ms
            ratios.setdefault(f"ggn_product_{method}", {})[str(batch)] = product_ms / forward_ms
            modes[f"{method}@{batch}"] = f"setup {setup_mode}, product {product_mode}"
            print(f"batch {batch} {method:10s} setup {setup_ms / forward_ms:6.2f} FE  "
                  f"product {product_ms / forward_ms:6.2f} FE  ({setup_mode}/{product_mode})",
                  flush=True)
        reference = results["jvp_vjp"]
        for method in METHODS:
            error = float(torch.linalg.vector_norm(results[method] - reference)
                          / torch.linalg.vector_norm(reference))
            agreement[f"{method}@{batch}"] = error
        print(f"batch {batch} relative disagreement vs jvp_vjp:",
              {m: f"{agreement[f'{m}@{batch}']:.1e}" for m in METHODS}, flush=True)

    # Cheapest method at the preregistered (curvature batch, K) pairs of the A3 grid.
    grid = [(64, 5), (128, 10), (128, 25), (512, 10), (512, 25), (512, 10)]

    def cost(method: str) -> float:
        total = 0.0
        for batch, k in grid:
            size = str(min(arguments.batches, key=lambda b: abs(b - batch)))
            total += batch * (ratios[f"ggn_setup_{method}"][size]
                              + (k + 1) * ratios[f"ggn_product_{method}"][size])
        return total

    costs = {method: cost(method) for method in METHODS}
    chosen = min(costs, key=costs.get)
    calibration.update({
        "ratios": ratios,
        "ggn_timing_modes": modes,
        "ggn_relative_disagreement_vs_jvp_vjp": agreement,
        "ggn_grid_cost_fe": costs,
        "ggn_method": chosen,
        "ggn_measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ggn_git": git_commit(),
        "ggn_selection_rule": "lowest total FE of setup + (K + 1) products over the A3 grid's "
                              "(|C|, K) pairs; the +1 is the warm-start product",
    })
    write_json(CALIBRATION, calibration)
    print("costs over the A3 grid (FE):", {m: f"{c:.3g}" for m, c in costs.items()},
          "-> chosen:", chosen, flush=True)


if __name__ == "__main__":
    main()
