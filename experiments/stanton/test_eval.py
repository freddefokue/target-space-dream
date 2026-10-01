#!/usr/bin/env python3
"""Phase 4: evaluate the preregistered final models on the test split, exactly once.

Models (PREREGISTRATION.md §7 step 5, Amendments 9 and 10): seeds 0, 1, 2 at λ ∈ {0, 0.25} for
A1-7, A2-3, A5-6, A6-3 (decision rule) and A1MSE-1, A1-anchor-mu0.01 (diagnostics); seed 0 at
λ ∈ {0, 0.25} for A3-6 and A4-6. Writes ``/workspace/runs/test_results.json`` and refuses to run
if it already exists. Metrics as for validation: top-1 agreement with the teacher, mean
KL(p_T || p_S) at temperature 1, label accuracy. The GPU time is charged to the ledger.
"""

from __future__ import annotations

import time

import torch

from common import RUNS, git_commit, ledger_add, set_numerics, write_json
from data import CIFAR100GPU, split_digest
from dream.models import ResNet20GN1
from evaluate import fidelity
from teacher import load_teacher, logits_on

OUT = RUNS / "test_results.json"
RULE = ["A1-7", "A2-3", "A5-6", "A6-3"]
DIAGNOSTICS = ["A1MSE-1", "A1-anchor-mu0.01"]
SEED0_ONLY = ["A3-6", "A4-6"]


def models() -> list[tuple[str, float, int]]:
    selected = [(c, lam, s) for c in RULE + DIAGNOSTICS for lam in (0.0, 0.25) for s in (0, 1, 2)]
    return selected + [(c, lam, 0) for c in SEED0_ONLY for lam in (0.0, 0.25)]


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: the test split has already been evaluated once.")
    missing = [f"{c}_lam{lam:g}_s{s}_full" for c, lam, s in models()
               if not (RUNS / f"{c}_lam{lam:g}_s{s}_full" / "final_model.pt").exists()]
    if missing:
        raise SystemExit(f"missing final models: {missing}")
    set_numerics()
    start = time.time()
    data = CIFAR100GPU(open_test=True)
    teacher_test = logits_on(load_teacher(), data.test)
    teacher_accuracy = float((teacher_test.argmax(-1) == data.test.labels).float().mean())
    student = ResNet20GN1().cuda()
    results = []
    for config, lam, seed in models():
        run = RUNS / f"{config}_lam{lam:g}_s{seed}_full"
        student.load_state_dict(torch.load(run / "final_model.pt", map_location="cuda",
                                           weights_only=True))
        metrics = fidelity(student.eval(), data.test, teacher_test)
        results.append({"config": config, "lam": lam, "seed": seed, "run": run.name,
                        **{f"test_{k}": v for k, v in metrics.items()}})
        print(run.name, {k: round(v, 4) if isinstance(v, float) else v
                         for k, v in metrics.items()}, flush=True)
    hours = (time.time() - start) / 3600
    write_json(OUT, {"evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_commit(),
                     "splits": split_digest(), "test_images": len(data.test),
                     "teacher_test_accuracy": teacher_accuracy, "results": results})
    total = ledger_add("test_eval", hours, 1, "single test-split evaluation")
    print(f"teacher test accuracy {teacher_accuracy:.4f}; {len(results)} models; "
          f"{hours * 3600:.0f} s; ledger {total:.2f}")


if __name__ == "__main__":
    main()
