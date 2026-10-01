#!/usr/bin/env python3
"""POST-HOC test evaluation for Amendment 11 (exploratory; the decision rule stays closed).

Evaluates, once, the λ = 0 ``4·B_full`` runs: the existing seed-0 runs of A1-7, A2-3 and A5-6
(Amendment 7) and the Amendment 11 runs (A5-6 seeds 1 and 2, A1MSE-1 seed 0, A1-7-tau4 seed 0, A1-7 seed 1).
Writes ``/workspace/runs/test_results_posthoc_amendment11.json`` and refuses to run if it exists.
"""

from __future__ import annotations

import time

import torch

from common import RUNS, git_commit, ledger_add, set_numerics, write_json
from data import CIFAR100GPU, split_digest
from dream.models import ResNet20GN1
from evaluate import fidelity
from teacher import load_teacher, logits_on

OUT = RUNS / "test_results_posthoc_amendment11.json"
RUNS_4X = ["A1-7_lam0_s0_4", "A2-3_lam0_s0_4", "A5-6_lam0_s0_4",
           "A5-6_lam0_s1_4", "A5-6_lam0_s2_4", "A1MSE-1_lam0_s0_4", "A1-7-tau4_lam0_s0_4",
           "A1-7_lam0_s1_4"]


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
    write_json(OUT, {"label": "POST-HOC, exploratory (Amendment 11); not part of the decision rule",
                     "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_commit(),
                     "splits": split_digest(), "results": results})
    ledger_add("test_eval_posthoc", (time.time() - start) / 3600, 1, "Amendment 11 post-hoc test")


if __name__ == "__main__":
    main()
