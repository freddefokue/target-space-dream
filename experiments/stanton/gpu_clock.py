#!/usr/bin/env python3
"""GPU busy-clock: pod wall-clock during which any process of this study holds the GPU.

This is the GPU-hour figure checked against the cap. Per-run ledger entries (wall / concurrency)
remain as an attribution of that time to runs, but they undercount when a queue's tail runs with
fewer jobs than its concurrency. Run inside tmux: ``python gpu_clock.py --offset-hours X``, where
X is the GPU time used before the clock started.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time

from common import RUNS, write_json

CLOCK = RUNS / "gpu_clock.json"
INTERVAL = 30.0


def gpu_busy() -> bool:
    output = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                            capture_output=True, text=True).stdout
    return bool(output.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offset-hours", type=float, default=None)
    arguments = parser.parse_args()
    state = json.loads(CLOCK.read_text()) if CLOCK.exists() else None
    if state is None:
        if arguments.offset_hours is None:
            raise SystemExit("first start needs --offset-hours")
        state = {"busy_hours": arguments.offset_hours, "offset_hours": arguments.offset_hours,
                 "started": time.time(), "interval_s": INTERVAL}
    last = time.time()
    while True:
        time.sleep(INTERVAL)
        now = time.time()
        if gpu_busy():
            state["busy_hours"] += (now - last) / 3600.0
        state["updated"] = now
        last = now
        write_json(CLOCK, state)


if __name__ == "__main__":
    main()
