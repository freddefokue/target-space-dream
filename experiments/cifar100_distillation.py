#!/usr/bin/env python3
"""Exact small-D CIFAR-100 target-space continuation.

This compact driver consumes the frozen cache and teacher checkpoint produced
by the thesis experiment. It intentionally contains no validation-driven
selection logic and evaluates only the fixed transfer set. See
``docs/reproduction.md`` for artifact formats and the report-only evaluation.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path

import torch
from torch.func import functional_call

from dream.continuation import ContinuationConfig, track_target_path
from dream.models import ResNet20GN1
from dream.parameters import ParameterSpec
from dream.targets import StraightTargetPath, helmert_contrast


def named_parameters(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, value.detach().clone()) for name, value in model.named_parameters())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/cifar100_dream"))
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--teacher-fraction", type=float, default=0.25)
    parser.add_argument("--student-seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--initial-step", type=float, default=0.25)
    arguments = parser.parse_args()

    device = torch.device(arguments.device)
    dtype = torch.float64
    teacher_model = ResNet20GN1().to(device=device, dtype=dtype).eval()
    teacher_model.load_state_dict(
        torch.load(arguments.teacher, map_location=device, weights_only=True)
    )
    torch.manual_seed(arguments.student_seed)
    random_model = ResNet20GN1().to(device=device, dtype=dtype).eval()
    teacher_params = named_parameters(teacher_model)
    random_params = named_parameters(random_model)
    init_params = OrderedDict(
        (
            name,
            arguments.teacher_fraction * teacher_params[name]
            + (1.0 - arguments.teacher_fraction) * random_params[name],
        )
        for name in teacher_params
    )
    spec = ParameterSpec.from_params(init_params)
    initial = spec.flatten(init_params)

    cache = torch.load(arguments.cache, map_location="cpu", weights_only=False)
    indices = cache["balanced_order"][: arguments.d]
    images = cache["train_images_u8"][indices].to(device=device, dtype=dtype) / 127.5 - 1.0
    contrast = helmert_contrast(100, device=device, dtype=dtype)

    def output(flat: torch.Tensor) -> torch.Tensor:
        params = spec.unflatten(flat)
        logits = functional_call(teacher_model, params, (images,), strict=False)
        return logits @ contrast

    with torch.no_grad():
        initial_output = output(initial).detach()
        teacher_output = functional_call(teacher_model, teacher_params, (images,), strict=False)
        teacher_output = (teacher_output @ contrast).detach()
    path = StraightTargetPath(initial_output, teacher_output)
    config = ContinuationConfig(initial_step=arguments.initial_step)
    result = track_target_path(output, initial, path, config)

    arguments.output.mkdir(parents=True, exist_ok=True)
    torch.save(spec.unflatten(result.parameters.detach().cpu()), arguments.output / "endpoint.pt")
    summary = {
        "d": arguments.d,
        "teacher_fraction": arguments.teacher_fraction,
        "reached_endpoint": result.reached_endpoint,
        "final_t": result.final_t,
        "accepted_stages": result.accepted_stages,
        "attempted_intervals": result.attempted_intervals,
        "jacobian_builds": result.jacobian_builds,
        "path_length": result.path_length,
        "endpoint_displacement": result.endpoint_displacement,
        "final_metrics": asdict(result.final_metrics),
        "history": result.history,
    }
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()

