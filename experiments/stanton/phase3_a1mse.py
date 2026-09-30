#!/usr/bin/env python3
"""A1-MSE diagnostic, Phase 3 part (Amendment 5): tune lr, select by the preregistered rule, sweep."""

from __future__ import annotations

import json
import subprocess
import sys

from phase3_orchestrate import FO, HERE, LAMBDAS, job, log, run_queue


def main() -> None:
    run_queue("phase3_tune_A1MSE", [job(f"A1MSE-{i}", "0.25", "tune", FO) for i in (1, 2, 3)],
              3, 6)
    subprocess.run([sys.executable, "analyze.py", "tuning"], cwd=HERE, check=True,
                   stdout=subprocess.DEVNULL)
    selected = json.loads((HERE / "tuning_selection.json").read_text())["A1-MSE"]["selected"]
    log(f"A1-MSE selected: {selected}")
    run_queue("phase3_sweep_A1MSE", [job(selected, lam, "full", FO) for lam in LAMBDAS], 5, 8)
    log("A1-MSE phase 3 done")


if __name__ == "__main__":
    main()
