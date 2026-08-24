"""A tiny, dependency-free demonstration of target-space DREAM."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch

from dream.continuation import ContinuationConfig, track_target_path
from dream.targets import StraightTargetPath


def run_toy(seed: int = 0) -> dict:
    """Distill a linear teacher on an underdetermined transfer set."""

    generator = torch.Generator().manual_seed(seed)
    rows, parameters = 12, 24
    design = torch.randn(rows, parameters, generator=generator, dtype=torch.float64)
    initial = torch.zeros(parameters, dtype=torch.float64)
    teacher = torch.randn(parameters, generator=generator, dtype=torch.float64)

    def output(vector: torch.Tensor) -> torch.Tensor:
        return (design @ vector).view(4, 3)

    path = StraightTargetPath(output(initial).detach(), output(teacher).detach())
    result = track_target_path(
        output,
        initial,
        path,
        ContinuationConfig(initial_step=0.5, maximum_step=0.5),
    )
    return {
        "reached_endpoint": result.reached_endpoint,
        "final_t": result.final_t,
        "accepted_stages": result.accepted_stages,
        "attempted_intervals": result.attempted_intervals,
        "jacobian_builds": result.jacobian_builds,
        "endpoint_displacement": result.endpoint_displacement,
        "final_metrics": asdict(result.final_metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    print(json.dumps(run_toy(arguments.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

