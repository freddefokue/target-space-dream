#!/usr/bin/env python3
"""Measure pass costs of ResNet20GN1 relative to one no-grad forward pass (FE per image).

Writes ``compute_calibration.json``. Each ratio is the median GPU time of a pass divided by the
median GPU time of a no-grad forward at the same batch size, float32, TF32 off. Passes are timed as
CUDA-graph replays, so the ratios measure GPU work and exclude CPU launch overhead, which otherwise
inflates small-batch backward passes (PREREGISTRATION.md, Amendment 2). Eager timings are recorded
alongside for reference. ``B_full`` is the FE of 200
epochs of arm A1 at batch 128: per image one student forward+backward and one teacher forward.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import time

import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp

from common import CALIBRATION, git_commit, set_numerics, write_json
from dream.models import ResNet20GN1


def median_ms(function, repeats: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def graphed(function):
    """Capture ``function`` in a CUDA graph and return its replay."""

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            function()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        function()
    return graph.replay


def measure(batch: int, repeats: int) -> tuple[dict[str, float], dict[str, float]]:
    torch.manual_seed(0)
    model = ResNet20GN1().cuda().train()
    images = torch.randn(batch, 3, 32, 32, device="cuda")
    target = torch.softmax(torch.randn(batch, 100, device="cuda"), dim=-1)
    params = {name: value.detach() for name, value in model.named_parameters()}
    tangent = {name: torch.randn_like(value) for name, value in params.items()}

    def forward_nograd():
        with torch.no_grad():
            model(images)

    def forward_grad():
        model(images)

    def forward_backward():
        model.zero_grad(set_to_none=False)
        loss = -(target * F.log_softmax(model(images), dim=-1)).sum(-1).mean()
        loss.backward()

    def forward_mode_jvp():
        jvp(lambda p: functional_call(model, p, (images,)), (params,), (tangent,))

    passes = {"fwd": forward_nograd, "fwd_grad": forward_grad, "fwd_bwd": forward_backward,
              "jvp": forward_mode_jvp}
    eager = {kind: median_ms(function, repeats) for kind, function in passes.items()}
    gpu = {kind: median_ms(graphed(function), repeats) for kind, function in passes.items()}
    return gpu, eager


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--repeats", type=int, default=50)
    arguments = parser.parse_args()
    set_numerics()

    milliseconds: dict[str, dict[str, float]] = {}
    eager_ms: dict[str, dict[str, float]] = {}
    ratios: dict[str, dict[str, float]] = {}
    for batch in arguments.batches:
        timings, eager = measure(batch, arguments.repeats)
        milliseconds[str(batch)] = timings
        eager_ms[str(batch)] = eager
        for kind, value in timings.items():
            ratios.setdefault(kind, {})[str(batch)] = value / timings["fwd"]
        ratios.setdefault("vjp", {})[str(batch)] = (timings["fwd_bwd"] - timings["fwd_grad"]) / timings["fwd"]
        print(batch, {k: round(v / timings["fwd"], 3) for k, v in timings.items()},
              f"fwd {timings['fwd']:.2f} ms | eager fwd_bwd ratio "
              f"{eager['fwd_bwd'] / eager['fwd']:.3f}", flush=True)

    per_image_a1 = ratios["fwd_bwd"]["128"] + ratios["fwd"]["128"]
    payload = {
        "description": "Pass cost in forward-equivalents per image: CUDA-graph GPU time of the "
                       "pass / CUDA-graph GPU time of a no-grad forward at the same batch size. "
                       "vjp = fwd_bwd - fwd_grad (backward only). eager_milliseconds are for "
                       "reference only (they include CPU launch overhead).",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "precision": "float32, TF32 disabled",
        "repeats": arguments.repeats,
        "git": git_commit(),
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "milliseconds": milliseconds,
        "eager_milliseconds": eager_ms,
        "ratios": ratios,
        "A1_fe_per_image": per_image_a1,
        "B_full": 200 * 50_000 * per_image_a1,
        "B_full_definition": "200 epochs x 50,000 images x (student fwd_bwd@128 + teacher fwd@128)",
    }
    write_json(CALIBRATION, payload)
    print("B_full =", f"{payload['B_full']:.4e}", "FE")


if __name__ == "__main__":
    main()
