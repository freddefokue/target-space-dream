"""Shared run infrastructure: numerics, compute accounting, run directories, ledger, index."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = Path("/workspace/runs")
CALIBRATION = HERE / "compute_calibration.json"
LOCK = Path("/tmp/stanton_runs.lock")


def set_numerics() -> None:
    """float32 everywhere, TF32 off (identical numerics across arms)."""

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = True


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ---------------------------------------------------------------------------------------------
# Compute accounting


def load_calibration() -> dict:
    return json.loads(CALIBRATION.read_text())


class ComputeCounter:
    """Tally of passes in forward-equivalents (FE) per image.

    Every pass is recorded by kind and image count. The primary FE total uses the measured cost
    ratios from ``compute_calibration.json``; the secondary total charges a JVP as 2 forward
    passes (PREREGISTRATION.md §5) and every other pass at its measured ratio.
    """

    def __init__(self, calibration: dict) -> None:
        self.ratios = calibration["ratios"]
        self.images: dict[str, int] = {}
        self.fe = 0.0
        self.fe_secondary = 0.0

    def ratio(self, kind: str, batch: int) -> float:
        table = self.ratios[kind]
        sizes = sorted(int(size) for size in table)
        nearest = min(sizes, key=lambda size: abs(size - batch))
        return float(table[str(nearest)])

    def cost(self, kind: str, batch: int) -> float:
        return self.ratio(kind, batch) * batch

    def secondary_cost(self, kind: str, batch: int) -> float:
        """Cost with every JVP charged as 2 forward passes.

        A GGN product is one JVP plus one backward pass, whichever method computes the JVP.
        """

        if kind == "jvp":
            return 2.0 * batch
        if kind.startswith("ggn_product_"):
            return (2.0 + self.ratio("vjp", batch)) * batch
        return self.cost(kind, batch)

    def charge(self, kind: str, batch: int, times: int = 1) -> None:
        if times == 0:
            return
        primary = self.cost(kind, batch) * times
        secondary = self.secondary_cost(kind, batch) * times
        batch *= times
        self.images[kind] = self.images.get(kind, 0) + batch
        self.fe += primary
        self.fe_secondary += secondary

    def state_dict(self) -> dict:
        return {"images": dict(self.images), "fe": self.fe, "fe_secondary": self.fe_secondary}

    def load_state_dict(self, state: dict) -> None:
        self.images = dict(state["images"])
        self.fe = float(state["fe"])
        self.fe_secondary = float(state["fe_secondary"])


def budget_fe(name: str | float, calibration: dict) -> float:
    full = float(calibration["B_full"])
    if name == "full":
        return full
    if name == "tune":
        return full / 4.0
    return full * float(name)


# ---------------------------------------------------------------------------------------------
# Run bookkeeping


@contextmanager
def _locked():
    LOCK.touch(exist_ok=True)
    with open(LOCK, "r+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def index_event(run_id: str, event: str, **fields) -> None:
    with _locked():
        append_jsonl(RUNS / "index.jsonl",
                     {"run_id": run_id, "event": event, "time": time.time(), **fields})


def ledger_add(run_id: str, wall_hours: float, concurrency: int, note: str = "") -> float:
    """Charge ``wall_hours / concurrency`` GPU-hours and return the new cumulative total.

    GPU-hours are pod wall-clock while the GPU is occupied by this project; runs that share the
    GPU share the hour.
    """

    path = RUNS / "budget.json"
    with _locked():
        ledger = json.loads(path.read_text()) if path.exists() else {
            "cap_gpu_hours": 40.0, "total_gpu_hours": 0.0, "entries": []}
        charged = wall_hours / max(1, concurrency)
        ledger["entries"].append({"run_id": run_id, "wall_hours": wall_hours,
                                  "concurrency": concurrency, "gpu_hours": charged,
                                  "time": time.time(), "note": note})
        ledger["total_gpu_hours"] = sum(entry["gpu_hours"] for entry in ledger["entries"])
        write_json(path, ledger)
        return ledger["total_gpu_hours"]


def save_checkpoint(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def truncate_metrics(path: Path, max_step: int) -> None:
    """Drop metric rows logged after the checkpoint a resumed run restarts from."""

    if not path.exists():
        return
    rows = [line for line in path.read_text().splitlines()
            if line.strip() and json.loads(line).get("step", 0) <= max_step]
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(row + "\n" for row in rows))
    os.replace(tmp, path)
