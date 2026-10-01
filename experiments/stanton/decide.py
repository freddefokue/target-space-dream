#!/usr/bin/env python3
"""Apply the preregistered decision rule (PREREGISTRATION.md §8, Amendment 9) to the test results.

Reads ``/workspace/runs/test_results.json`` (never re-evaluates) and writes
``decision.json`` next to this file. "sd" is the sample standard deviation (n − 1 = 2) over seeds
0, 1, 2; ``sd_pool(X, Y) = sqrt((s_X² + s_Y²) / 2)``. "X beats Y by the GO margin": mean
difference ≥ 1.0 pp and > 2·sd_pool. "Within two SD": |difference| ≤ 2·sd_pool. A3 and A4 have
seed 0 only (Amendment 9): an outcome requiring A3/A4 to beat or match another arm is "not met" if
the seed-0 test agreement of that run is more than 10 pp below A1's 3-seed mean, otherwise
"undecidable".
"""

from __future__ import annotations

import json
import math
import statistics

from common import HERE, RUNS, write_json

ARMS = {"A1": "A1-7", "A2": "A2-3", "A3": "A3-6", "A4": "A4-6", "A5": "A5-6", "A6": "A6-3",
        "A1-MSE": "A1MSE-1", "anchor": "A1-anchor-mu0.01"}


def stats(rows: list[dict], config: str, lam: float, key: str) -> dict:
    values = [r[key] for r in sorted(rows, key=lambda r: r["seed"])
              if r["config"] == config and r["lam"] == lam]
    return {"values": values, "mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values)}


def compare(x: dict, y: dict) -> dict:
    diff = 100 * (x["mean"] - y["mean"])
    pooled = 100 * math.sqrt(((x["sd"] or 0) ** 2 + (y["sd"] or 0) ** 2) / 2)
    return {"diff_pp": diff, "sd_pool_pp": pooled,
            "beats": diff >= 1.0 and diff > 2 * pooled, "within_2sd": abs(diff) <= 2 * pooled}


def decide(rows: list[dict], lam: float) -> dict:
    agree = {arm: stats(rows, cfg, lam, "test_agreement") for arm, cfg in ARMS.items()}
    kl = {arm: stats(rows, cfg, lam, "test_kl") for arm, cfg in ARMS.items()}
    a1 = agree["A1"]["mean"]
    seed0_far = {arm: 100 * (a1 - agree[arm]["mean"]) > 10.0 for arm in ("A3", "A4")}

    def gn_outcome(arms: tuple[str, ...]) -> str:
        return "not met" if all(seed0_far[a] for a in arms) else "undecidable (flag to Fred)"

    three = ("A1", "A2", "A5", "A6")
    vs_a1 = {arm: compare(agree[arm], agree["A1"]) for arm in ("A2", "A5", "A6")}
    for arm in ("A3", "A4"):  # seed 0 only: margin arithmetic without a seed spread
        diff = 100 * (agree[arm]["mean"] - a1)
        vs_a1[arm] = {"diff_pp": diff, "sd_pool_pp": None, "beats": diff >= 1.0,
                      "within_2sd": None}
    outcomes = {
        "GO": gn_outcome(("A4",)),
        "second order is the lever": gn_outcome(("A3", "A4")),
        "the path is the lever": gn_outcome(("A4",)),
        "teacher trajectory suffices": gn_outcome(("A4",)),
    }
    a2_beats = vs_a1["A2"]["beats"]
    a5_vs_a2 = compare(agree["A5"], agree["A2"])
    outcomes["temperature suffices"] = (
        ("met" if a5_vs_a2["within_2sd"] else "not met") if a2_beats
        else "not applicable (neither A2 nor A4 beats A1 by the GO margin)")
    outcomes["NO-GO"] = "met" if not any(v["beats"] for v in vs_a1.values()) else "not met"
    descriptive = {f"{x} vs {y}": compare(agree[x], agree[y])
                   for x, y in (("A2", "A1"), ("A5", "A1"), ("A6", "A1"), ("A5", "A2"),
                                ("A2", "A6"), ("A5", "A6"), ("A1-MSE", "A1"),
                                ("A1-MSE", "A5"), ("anchor", "A1"), ("anchor", "A5"),
                                ("anchor", "A2"))}
    return {"lam": lam, "agreement": agree, "kl": kl, "vs_A1": vs_a1,
            "A3_A4_more_than_10pp_below_A1": seed0_far, "outcomes": outcomes,
            "descriptive": descriptive, "decision_arms": list(three) + ["A3", "A4"]}


def main() -> None:
    test = json.loads((RUNS / "test_results.json").read_text())
    decision = {"source": str(RUNS / "test_results.json"), "evaluated_at": test["evaluated_at"],
                "teacher_test_accuracy": test["teacher_test_accuracy"],
                "sd": "sample standard deviation over seeds 0, 1, 2 (n - 1 = 2)",
                "per_lambda": [decide(test["results"], lam) for lam in (0.0, 0.25)]}
    write_json(HERE / "decision.json", decision)
    for block in decision["per_lambda"]:
        print(f"\nλ = {block['lam']:g}")
        for arm, s in block["agreement"].items():
            sd = f"{100 * s['sd']:.2f}" if s["sd"] is not None else "n/a"
            print(f"  {arm:7s} {100 * s['mean']:6.2f} ± {sd:>5s}  KL {block['kl'][arm]['mean']:.4f}"
                  f"  seeds {[round(100 * v, 2) for v in s['values']]}")
        for arm, c in block["vs_A1"].items():
            print(f"  {arm} − A1: {c['diff_pp']:+.2f} pp, sd_pool "
                  f"{c['sd_pool_pp'] if c['sd_pool_pp'] is None else round(c['sd_pool_pp'], 2)}, "
                  f"beats: {c['beats']}")
        for name, verdict in block["outcomes"].items():
            print(f"  {name}: {verdict}")


if __name__ == "__main__":
    main()
