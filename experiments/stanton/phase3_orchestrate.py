#!/usr/bin/env python3
"""Unattended remainder of Phase 3 (preregistered steps only; validation only).

1. Wait for the A4 tuning runs; select A4 by the preregistered rule (``analyze.py tuning``).
2. Run the A4 λ sweep.
3. Re-run every Phase 3 sweep queue until no job fails (crashed runs resume from checkpoints;
   finished runs are skipped), at most three passes.
4. Run the hand-off diagnostic (PREREGISTRATION.md §7): endpoints of A2, A4, A6 and the A1
   control at λ ∈ {0, 0.25}, seed 0, primary and secondary variants, at B_full/4.
Writes ``/workspace/runs/phase3_orchestrate.done`` at the end.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = Path("/workspace/runs")
QUEUES = RUNS / "queues"
CONFIGS = "../../configs/stanton"
GN, FO = 9.5, 1.2
LAMBDAS = ("0", "0.1", "0.25", "0.4", "0.5")


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def running_queues() -> int:
    output = subprocess.run(["pgrep", "-f", "python run_queue.py"], capture_output=True,
                            text=True).stdout
    return len(output.split())


def run_queue(name: str, jobs: list[dict], concurrency: int, mem_gb: float) -> None:
    path = QUEUES / f"{name}.json"
    path.write_text(json.dumps(jobs, indent=1))
    for attempt in range(1, 4):
        log(f"queue {name} pass {attempt}")
        with open(RUNS / "logs" / f"queue_{name}.log", "a") as handle:
            subprocess.run([sys.executable, "run_queue.py", str(path), "--concurrency",
                            str(concurrency), "--mem-gb", str(mem_gb)], cwd=HERE, stdout=handle,
                           stderr=subprocess.STDOUT, check=False)
        missing = [job for job in jobs if not finished(job)]
        if not missing:
            return
        log(f"queue {name}: {len(missing)} job(s) unfinished after pass {attempt}")
    log(f"queue {name}: giving up on {len(missing)} job(s); needs attention")


def run_dir(job: dict) -> Path:
    args = job["args"]
    config = json.loads((HERE / args[args.index("--config") + 1]).read_text())
    lam, seed, budget = (args[args.index(flag) + 1] for flag in ("--lam", "--seed", "--budget"))
    name = f"{config['name']}_lam{float(lam):g}_s{seed}_{budget}"
    if "--handoff" in args:
        name += f"_ho-{args[args.index('--handoff') + 1]}"
    if "--init-from" in args:
        name += f"_from-{Path(args[args.index('--init-from') + 1]).name}"
    return RUNS / name


def finished(job: dict) -> bool:
    return (run_dir(job) / "final.json").exists()


def job(config: str, lam: str, budget: str, mem: float, *extra: str) -> dict:
    return {"args": ["--config", f"{CONFIGS}/{config}.json", "--lam", lam, "--seed", "0",
                     "--budget", budget, *extra], "mem_gb": mem}


def main() -> None:
    # 1. A4 selection.
    while not all((RUNS / f"A4-{i}_lam0.25_s0_tune" / "final.json").exists() for i in range(1, 7)):
        time.sleep(60)
    subprocess.run([sys.executable, "analyze.py", "tuning"], cwd=HERE, check=True,
                   stdout=subprocess.DEVNULL)
    selection = json.loads((HERE / "tuning_selection.json").read_text())
    a4 = selection["A4"]["selected"]
    log(f"A4 selected: {a4}; full selection { {k: v['selected'] for k, v in selection.items()} }")

    # 2./3. A4 sweep, then make sure every sweep job has finished.
    run_queue("phase3_sweep_A4", [job(a4, lam, "full", GN) for lam in LAMBDAS], 5, 30)
    while running_queues() > 0:  # let the other Phase 3 queues drain first
        time.sleep(60)
    main_jobs = json.loads((QUEUES / "phase3_main.json").read_text())
    a5_jobs = json.loads((QUEUES / "phase3_sweep_A5.json").read_text())
    run_queue("phase3_main", main_jobs + a5_jobs, 10, 40)

    # 4. Hand-off diagnostic.
    a1, a2, a6 = (selection[arm]["selected"] for arm in ("A1", "A2", "A6"))
    handoff = []
    for source in (a2, a4, a6, a1):
        for lam in ("0", "0.25"):
            source_dir = RUNS / f"{source}_lam{float(lam):g}_s0_full"
            for variant in ("primary", "secondary"):
                handoff.append(job(a1, lam, "tune", FO, "--handoff",
                                   variant, "--init-from", str(source_dir)))
    run_queue("phase3_handoff", handoff, 10, 40)
    (RUNS / "phase3_orchestrate.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    log("phase 3 orchestration done")


if __name__ == "__main__":
    main()
