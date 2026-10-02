#!/usr/bin/env python3
"""Run the follow-up of FOLLOWUP_PREREG.md end to end (λ = 0, 4·B_full).

Stage A (parallel): A1-7 seed 2, A1-7-tau2 and A1-7-tau8 seed 0, A5-short seeds 0-2. Then τ is
selected on validation only, stage B runs the selected τ's seeds 1 and 2, every new run gets the
50k clean-train evaluation, the test split is evaluated once for all λ = 0 4× runs, and the
decision rule is applied. Writes ``/workspace/runs/followup.done`` at the end.
"""

from __future__ import annotations

import json
import math
import statistics
import subprocess
import sys
import time

import torch

from common import RUNS, git_commit, ledger_add, set_numerics, write_json
from phase3_orchestrate import HERE, log, run_queue

CONFIGS = "../../configs/stanton"
TEST_OUT = RUNS / "test_results_followup.json"
DECISION_OUT = HERE / "followup_decision.json"


def job(config: str, seed: int) -> dict:
    return {"args": ["--config", f"{CONFIGS}/{config}.json", "--lam", "0", "--seed", str(seed),
                     "--budget", "4", "--full-train-eval"], "mem_gb": 1.5}


def name(config: str, seed: int) -> str:
    return f"{config}_lam0_s{seed}_4"


def final(config: str, seed: int) -> dict:
    return json.loads((RUNS / name(config, seed) / "final.json").read_text())


def select_tau() -> str:
    """Highest seed-0 validation agreement; ties within 0.1 pp go to the lower validation KL."""

    candidates = [(c, final(c, 0)) for c in ("A1-7-tau2", "A1-7-tau4", "A1-7-tau8")]
    ordered = sorted(candidates, key=lambda item: -item[1]["val_agreement"])
    best = ordered[0][1]["val_agreement"]
    tied = [item for item in ordered if best - item[1]["val_agreement"] <= 0.001]
    chosen = min(tied, key=lambda item: item[1]["val_kl"])[0]
    table = {c: {"val_agreement": f["val_agreement"], "val_kl": f["val_kl"]}
             for c, f in candidates}
    write_json(HERE / "followup_tau_selection.json", {"selected": chosen, "candidates": table,
                                                      "rule": "max seed-0 val agreement; ties "
                                                              "within 0.1 pp -> lower val KL"})
    return chosen


def test_models(tau_config: str) -> list[str]:
    runs = [name("A1-7", s) for s in (0, 1, 2)] + [name("A2-3", 0)]
    runs += [name("A5-6", s) for s in (0, 1, 2)] + [name("A1MSE-1", 0)]
    runs += [name(c, 0) for c in ("A1-7-tau2", "A1-7-tau4", "A1-7-tau8")]
    runs += [name(tau_config, s) for s in (1, 2)]
    runs += [name("A5-short", s) for s in (0, 1, 2)]
    return runs


def evaluate_test(runs: list[str]) -> dict:
    if TEST_OUT.exists():
        raise SystemExit(f"{TEST_OUT} exists: the follow-up test split was already evaluated.")
    from data import CIFAR100GPU, split_digest
    from dream.models import ResNet20GN1
    from evaluate import fidelity
    from teacher import load_teacher, logits_on

    set_numerics()
    start = time.time()
    data = CIFAR100GPU(open_test=True)
    teacher_test = logits_on(load_teacher(), data.test)
    student = ResNet20GN1().cuda()
    results = []
    for run in runs:
        student.load_state_dict(torch.load(RUNS / run / "final_model.pt", map_location="cuda",
                                           weights_only=True))
        metrics = fidelity(student.eval(), data.test, teacher_test)
        results.append({"run": run, **{f"test_{k}": v for k, v in metrics.items()}})
    payload = {"label": "follow-up (FOLLOWUP_PREREG.md), single test evaluation",
               "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_commit(),
               "splits": split_digest(), "results": results}
    write_json(TEST_OUT, payload)
    ledger_add("test_eval_followup", (time.time() - start) / 3600, 1, "follow-up test")
    return payload


def summary(values: list[float]) -> dict:
    return {"values": values, "mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else None}


def compare(x: dict, y: dict) -> dict:
    diff = 100 * (x["mean"] - y["mean"])
    pooled = 100 * math.sqrt((x["sd"] ** 2 + y["sd"] ** 2) / 2)
    return {"diff_pp": diff, "sd_pool_pp": pooled, "two_sd_pool_pp": 2 * pooled,
            "beats": diff >= 1.0 and diff > 2 * pooled, "within_2sd": abs(diff) <= 2 * pooled}


def decide(test: dict, tau_config: str) -> dict:
    by_run = {r["run"]: r for r in test["results"]}
    arms = {"A5-6": [name("A5-6", s) for s in (0, 1, 2)],
            tau_config: [name(tau_config, s) for s in (0, 1, 2)],
            "A5-short": [name("A5-short", s) for s in (0, 1, 2)],
            "A1-7 (tau = 1)": [name("A1-7", s) for s in (0, 1, 2)]}
    stats = {arm: {"test_agreement": summary([by_run[r]["test_agreement"] for r in runs]),
                   "test_kl": summary([by_run[r]["test_kl"] for r in runs]), "runs": runs}
             for arm, runs in arms.items()}
    primary = compare(stats["A5-6"]["test_agreement"], stats[tau_config]["test_agreement"])
    verdict = ("annealing beats fixed temperature" if primary["beats"]
               else "soft targets plus patience suffice")
    secondary = {
        "A5-short vs A5-6": compare(stats["A5-short"]["test_agreement"],
                                    stats["A5-6"]["test_agreement"]),
        "A5-6 vs A1-7 (tau = 1)": compare(stats["A5-6"]["test_agreement"],
                                          stats["A1-7 (tau = 1)"]["test_agreement"]),
        f"{tau_config} vs A1-7 (tau = 1)": compare(stats[tau_config]["test_agreement"],
                                                   stats["A1-7 (tau = 1)"]["test_agreement"]),
        "A5-short vs A1-7 (tau = 1)": compare(stats["A5-short"]["test_agreement"],
                                              stats["A1-7 (tau = 1)"]["test_agreement"]),
    }
    decision = {"selected_fixed_tau": tau_config, "stats": stats,
                "primary": {"comparison": f"A5-6 vs {tau_config}", **primary,
                            "verdict": verdict},
                "secondary_descriptive": secondary,
                "sd": "sample standard deviation over seeds 0, 1, 2 (n - 1 = 2)"}
    write_json(DECISION_OUT, decision)
    return decision


def main() -> None:
    log("follow-up stage A")
    run_queue("followup_stageA", [job("A1-7", 2), job("A1-7-tau2", 0), job("A1-7-tau8", 0)]
              + [job("A5-short", s) for s in (0, 1, 2)], 6, 12)
    tau_config = select_tau()
    log(f"selected fixed temperature: {tau_config}")
    run_queue("followup_stageB", [job(tau_config, s) for s in (1, 2)], 2, 4)
    new_runs = [name("A1-7", 2), name("A1-7-tau2", 0), name("A1-7-tau8", 0)]
    new_runs += [name("A5-short", s) for s in (0, 1, 2)] + [name(tau_config, s) for s in (1, 2)]
    missing = [r for r in new_runs if not (RUNS / r / "final.json").exists()]
    if missing:
        log(f"needs attention: unfinished runs {missing}; stopping before the test evaluation")
        return
    subprocess.run([sys.executable, "train_eval.py", *[str(RUNS / r) for r in new_runs]],
                   cwd=HERE, check=True)
    test = evaluate_test(test_models(tau_config))
    decision = decide(test, tau_config)
    log(f"decision: {decision['primary']['verdict']} "
        f"(diff {decision['primary']['diff_pp']:+.2f} pp, 2*sd_pool "
        f"{decision['primary']['two_sd_pool_pp']:.2f} pp)")
    (RUNS / "followup.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    log("follow-up done")


if __name__ == "__main__":
    main()
