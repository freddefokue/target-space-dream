#!/usr/bin/env python3
"""Run a list of distill.py jobs with fixed concurrency, one log file per job.

Usage (inside tmux): ``python run_queue.py jobs.json --concurrency 2``. ``jobs.json`` holds a list of
argument lists for ``distill.py`` (without ``--concurrency``, which the queue adds). Finished runs
are skipped by distill.py itself; interrupted runs resume from their checkpoint, so rerunning the
same queue after a crash is safe.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", type=Path)
    parser.add_argument("--concurrency", type=int, default=2)
    arguments = parser.parse_args()
    jobs = json.loads(arguments.jobs.read_text())
    LOGS.mkdir(parents=True, exist_ok=True)
    pending = list(enumerate(jobs))
    running: list[tuple[int, subprocess.Popen, float]] = []
    failures = 0
    while pending or running:
        while pending and len(running) < arguments.concurrency:
            index, job = pending.pop(0)
            name = "_".join(str(part).replace("/", "-").lstrip("-") for part in job)[-120:]
            log = open(LOGS / f"queue_{arguments.jobs.stem}_{index:03d}_{name}.log", "a")
            command = [sys.executable, str(HERE / "distill.py"), *map(str, job),
                       "--concurrency", str(arguments.concurrency)]
            print(time.strftime("%H:%M:%S"), "start", index, " ".join(map(str, job)), flush=True)
            running.append((index, subprocess.Popen(command, cwd=HERE, stdout=log,
                                                    stderr=subprocess.STDOUT), time.time()))
        time.sleep(5)
        for entry in list(running):
            index, process, started = entry
            if process.poll() is not None:
                running.remove(entry)
                failures += process.returncode != 0
                print(time.strftime("%H:%M:%S"), "end", index, "rc", process.returncode,
                      f"{(time.time() - started) / 60:.1f} min", flush=True)
    print("queue finished,", failures, "failed job(s)", flush=True)


if __name__ == "__main__":
    main()
