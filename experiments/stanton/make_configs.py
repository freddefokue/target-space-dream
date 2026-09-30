#!/usr/bin/env python3
"""Write the preregistered tuning configurations to ``configs/stanton/``.

A1 and A3 grids are fixed. A2 and A5 inherit the optimizer settings of the selected A1 config,
A4 inherits the numerics of the selected A3 config (PREREGISTRATION.md §6), so they are written
only when ``--a1`` / ``--a3`` name the selected configs (or a ``--prefix`` for smoke tests).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "stanton"
OPTIMIZER_KEYS = ("optimizer", "lr", "weight_decay", "momentum", "nesterov", "betas", "lr_min",
                  "batch", "warmup")
SCHEDULES = [{"kind": "fixed", "phi": 0.25}, {"kind": "fixed", "phi": 0.5},
             {"kind": "fixed", "phi": 0.75}, {"kind": "adaptive", "threshold": 0.05},
             {"kind": "adaptive", "threshold": 0.15}, {"kind": "adaptive", "threshold": 0.4}]
TEMPERATURES = [{"kind": "geometric", "T0": 4.0, "phi": 0.25},
                {"kind": "geometric", "T0": 4.0, "phi": 0.5},
                {"kind": "geometric", "T0": 4.0, "phi": 0.75},
                {"kind": "geometric", "T0": 16.0, "phi": 0.25},
                {"kind": "geometric", "T0": 16.0, "phi": 0.5},
                {"kind": "annealing_kd", "tau_max": 10, "phi": 0.5,
                 "note": "Jafari et al. 2021: MSE(f_S, Phi(T) f_T), T = 10..1 in ten stages over "
                         "50% of the budget, then T = 1 (Amendment 3)"}]
A3_GRID = [(256, 64, 5), (512, 128, 10), (512, 128, 25), (2048, 512, 10), (2048, 512, 25),
           (512, 512, 10)]


def write(name: str, config: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(
        json.dumps({"name": name, **config}, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a1", help="selected A1 config name, e.g. A1-1")
    parser.add_argument("--a3", help="selected A3 config name, e.g. A3-2")
    parser.add_argument("--prefix", default="", help="name prefix (e.g. smoke-)")
    parser.add_argument("--out", type=Path, default=CONFIGS)
    arguments = parser.parse_args()
    p = arguments.prefix

    a1_7 = json.loads((CONFIGS / "A1-1.json").read_text())
    a1_7.pop("name")
    a1_7.update({"warmup": 0.1, "note": "A1-1 plus linear lr warmup over the first 10% "
                                        "(Amendment 3)"})
    write(f"{p}A1-7", a1_7, arguments.out)
    for index, (batch, curvature, iterations) in enumerate(A3_GRID, start=1):
        write(f"{p}A3-{index}", {"arm": "A3", "gn": {
            "batch": batch, "curvature_batch": curvature, "cg_iterations": iterations,
            "damping": 1.0, "warm_start": 0.9}}, arguments.out)

    if arguments.a1:
        selected = json.loads((CONFIGS / f"{arguments.a1}.json").read_text())
        optimizer = {k: selected[k] for k in OPTIMIZER_KEYS if k in selected}
        for index, schedule in enumerate(SCHEDULES, start=1):
            write(f"{p}A2-{index}", {"arm": "A2", "inherits": arguments.a1, **optimizer,
                                     "schedule": schedule}, arguments.out)
        for index, schedule in enumerate(TEMPERATURES, start=1):
            write(f"{p}A5-{index}", {"arm": "A5", "inherits": arguments.a1, **optimizer,
                                     "schedule": schedule}, arguments.out)
    if arguments.a3:
        selected = json.loads((CONFIGS / f"{arguments.a3}.json").read_text())
        for index, schedule in enumerate(SCHEDULES, start=1):
            write(f"{p}A4-{index}", {"arm": "A4", "inherits": arguments.a3,
                                     "gn": selected["gn"], "schedule": schedule}, arguments.out)


if __name__ == "__main__":
    main()
