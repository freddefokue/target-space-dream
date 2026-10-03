#!/usr/bin/env python3
"""POST-HOC test evaluation for follow-up Amendment F2 (fixed tau = 8 at 1x B_full; exploratory).

Evaluates, once, the three fixed-τ = 8 runs at λ = 0, ``1·B_full`` (seeds 0, 1, 2).
Writes ``/workspace/runs/test_results_posthoc_tau8_1x.json`` and refuses to run if it exists.
"""

from __future__ import annotations

import time

import torch

from common import RUNS, git_commit, ledger_add, set_numerics, write_json
from data import CIFAR100GPU, split_digest
from dream.models import ResNet20GN1
from evaluate import fidelity
from teacher import load_teacher, logits_on

OUT = RUNS / "test_results_posthoc_tau8_1x.json"
RUNS_4X = ["A1-7-tau8_lam0_s0_full", "A1-7-tau8_lam0_s1_full", "A1-7-tau8_lam0_s2_full"]


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: the post-hoc evaluation has already been run.")
    missing = [name for name in RUNS_4X if not (RUNS / name / "final_model.pt").exists()]
    if missing:
        raise SystemExit(f"missing final models: {missing}")
    set_numerics()
    start = time.time()
    data = CIFAR100GPU(open_test=True)
    teacher_test = logits_on(load_teacher(), data.test)
    student = ResNet20GN1().cuda()
    results = []
    for name in RUNS_4X:
        student.load_state_dict(torch.load(RUNS / name / "final_model.pt", map_location="cuda",
                                           weights_only=True))
        metrics = fidelity(student.eval(), data.test, teacher_test)
        results.append({"run": name, **{f"test_{k}": v for k, v in metrics.items()}})
        print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})
    write_json(OUT, {"label": "POST-HOC, exploratory (follow-up Amendment F2); no decision rule",
                     "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_commit(),
                     "splits": split_digest(), "results": results})
    ledger_add("test_eval_posthoc", (time.time() - start) / 3600, 1, "Amendment F2 post-hoc test")


if __name__ == "__main__":
    main()
