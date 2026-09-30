#!/usr/bin/env python3
"""Run a list of distill.py jobs, one log file per job, within a concurrency and memory limit.

Usage (inside tmux): ``python run_queue.py jobs.json --concurrency 8``. ``jobs.json`` holds a list
of jobs for ``distill.py`` (without ``--concurrency``, which the queue adds): either an argument
list or ``{"args": [...], "mem_gb": 7.7}``. A job starts only while fewer than ``--concurrency``
jobs run, the memory estimates of the running jobs plus its own stay within ``--mem-gb``, and the
GPU reports enough free memory. Finished runs are skipped by distill.py itself; interrupted runs
resume from their checkpoint, so rerunning the same queue after a crash is safe.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = Path("/workspace/runs/logs")


def free_gpu_gb() -> float:
    output = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return float(output.split()[0]) / 1024.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", type=Path)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--mem-gb", type=float, default=40.0,
                        help="total GPU memory the queue's jobs may use (estimates)")
    arguments = parser.parse_args()
    jobs = json.loads(arguments.jobs.read_text())
    LOGS.mkdir(parents=True, exist_ok=True)
    pending = [(i, job if isinstance(job, dict) else {"args": job, "mem_gb": 1.5})
               for i, job in enumerate(jobs)]
    running: list[tuple[int, subprocess.Popen, float, float]] = []
    failures = 0
    while pending or running:
        while pending and len(running) < arguments.concurrency:
            used = sum(entry[3] for entry in running)
            free = free_gpu_gb()
            fitting = [position for position, (_, spec) in enumerate(pending)
                       if not running or (used + spec["mem_gb"] <= arguments.mem_gb
                                          and free >= spec["mem_gb"] + 1.0)]
            if not fitting:  # the first pending job that fits starts; order is kept otherwise
                break
            index, spec = pending.pop(fitting[0])
            job = spec["args"]
            name = "_".join(str(part).replace("/", "-").lstrip("-") for part in job)[-120:]
            log = open(LOGS / f"queue_{arguments.jobs.stem}_{index:03d}_{name}.log", "a")
            command = [sys.executable, str(HERE / "distill.py"), *map(str, job),
                       "--concurrency", str(arguments.concurrency)]
            print(time.strftime("%H:%M:%S"), "start", index, " ".join(map(str, job)), flush=True)
            running.append((index, subprocess.Popen(command, cwd=HERE, stdout=log,
                                                    stderr=subprocess.STDOUT), time.time(),
                            spec["mem_gb"]))
            time.sleep(30)  # let the job allocate before the next free-memory check
        time.sleep(5)
        for entry in list(running):
            index, process, started, _ = entry
            if process.poll() is not None:
                running.remove(entry)
                failures += process.returncode != 0
                print(time.strftime("%H:%M:%S"), "end", index, "rc", process.returncode,
                      f"{(time.time() - started) / 60:.1f} min", flush=True)
    print("queue finished,", failures, "failed job(s)", flush=True)


if __name__ == "__main__":
    main()
