#!/usr/bin/env python3
"""Tuning selection, summary tables and figures for Phase 3 (validation only).

``python analyze.py tuning`` ranks every tuning run per arm by final validation agreement (ties
within 0.1 pp broken by lower validation KL, PREREGISTRATION.md §6), writes
``tuning_selection.json`` and prints markdown tables. ``python analyze.py figures`` draws the
figures into ``experiments/stanton/figures/``: distance to the teacher for every λ > 0 run of every
arm, and (after the sweep) agreement and KL against compute and final agreement against λ.
The test split is never read here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import HERE, RUNS, write_json  # noqa: E402

FIGURES = HERE / "figures"
SELECTION = HERE / "tuning_selection.json"
ARMS = ["A1", "A2", "A3", "A4", "A5", "A6", "A1-MSE"]
# Reference categorical palette (dataviz skill, light mode), fixed order, never cycled.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
               "#e34948"]
# Ordinal blue ramp for λ (steps 250, 400, 550, 700).
LAMBDA_COLORS = {0.1: "#86b6ef", 0.25: "#3987e5", 0.4: "#1c5cab", 0.5: "#0d366b"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2,
        "ytick.color": INK_2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
    })


def load_run(directory: Path) -> dict | None:
    final = directory / "final.json"
    if not final.exists():
        return None
    rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()
            if line.strip()]
    return {"final": json.loads(final.read_text()), "rows": rows, "dir": directory}


def runs(budget: str, seed: int | None = 0) -> list[dict]:
    found = []
    for directory in sorted(RUNS.iterdir()):
        name = directory.name
        if (not directory.is_dir() or f"_{budget}" not in name or "_from-" in name
                or not name.startswith("A")):
            continue
        run = load_run(directory)
        if run and (seed is None or run["final"]["seed"] == seed) \
                and run["final"]["budget_name"] == budget:
            found.append(run)
    return found


def arm_of(run: dict) -> str:
    """The arm, or a diagnostic group such as A1-MSE that shares an arm's code path."""

    config = run["final"]["config"]
    return config.get("group", config["arm"])


def config_of(run: dict) -> str:
    return run["final"]["config"]["name"]


# ---------------------------------------------------------------------------------------------
# Tuning


def rank(candidates: list[dict]) -> list[dict]:
    """Sort by validation agreement; within 0.1 pp of the best, lower KL wins."""

    ordered = sorted(candidates, key=lambda r: -r["final"]["val_agreement"])
    if not ordered:
        return ordered
    best = ordered[0]["final"]["val_agreement"]
    tied = [r for r in ordered if best - r["final"]["val_agreement"] <= 0.001]
    rest = [r for r in ordered if r not in tied]
    return sorted(tied, key=lambda r: r["final"]["val_kl"]) + rest


def tuning() -> None:
    tuned = [r for r in runs("tune") if r["final"]["lam"] == 0.25
             and "anchor" not in config_of(r)]
    selection = {}
    for arm in ARMS:
        ranked = rank([r for r in tuned if arm_of(r) == arm])
        if not ranked:
            continue
        print(f"\n### {arm} tuning (λ = 0.25, seed 0, B_full/4)\n")
        print("| rank | config | val agree | val KL | train-subset agree | ‖θ−θ_T‖ end | "
              "‖θ−θ_init‖ end | extra |")
        print("|---:|---|---:|---:|---:|---:|---:|---|")
        for position, run in enumerate(ranked, start=1):
            final = run["final"]
            extra = []
            if "mean_accepted" in final:
                extra.append(f"accepted {final['mean_accepted']:.2f}")
            summary = final.get("schedule_summary")
            if summary:
                extra.append(f"final t {summary['final_t']:.2f}, adv {summary['advances']}, "
                             f"rel {summary['forced_relative']}, "
                             f"deadline {'yes' if summary['deadline_triggered'] else 'no'}")
            print(f"| {position} | {config_of(run)} | {100 * final['val_agreement']:.2f}% | "
                  f"{final['val_kl']:.4f} | {100 * final['train_subset_agreement']:.2f}% | "
                  f"{final['teacher_distance']:.1f} | {final['displacement']:.1f} | "
                  f"{'; '.join(extra)} |")
        selection[arm] = {"selected": config_of(ranked[0]),
                          "ranking": [config_of(r) for r in ranked],
                          "val_agreement": [r["final"]["val_agreement"] for r in ranked],
                          "val_kl": [r["final"]["val_kl"] for r in ranked]}
    write_json(SELECTION, selection)
    print("\nselected:", {arm: entry["selected"] for arm, entry in selection.items()})


# ---------------------------------------------------------------------------------------------
# Figures


def fraction(rows: list[dict]) -> list[float]:
    return [row["fraction"] for row in rows]


def label_end(axis, run_rows, key, text, color) -> None:
    axis.annotate(text, (run_rows[-1]["fraction"], run_rows[-1][key]), xytext=(4, 0),
                  textcoords="offset points", va="center", fontsize=7, color=INK_2)


def tuning_distance_figure() -> Path:
    """‖θ − θ_T‖ against budget for every tuning run (all at λ = 0.25), one panel per arm."""

    tuned = [r for r in runs("tune") if r["final"]["lam"] == 0.25]
    arms = [arm for arm in ARMS if any(arm_of(r) == arm for r in tuned)]
    figure, axes = plt.subplots(1, len(arms), figsize=(3.2 * len(arms), 3.0), squeeze=False)
    for axis, arm in zip(axes[0], arms):
        members = sorted([r for r in tuned if arm_of(r) == arm], key=config_of)
        for index, run in enumerate(members):
            color = CATEGORICAL[index % len(CATEGORICAL)]
            axis.plot(fraction(run["rows"]), [row["teacher_distance"] for row in run["rows"]],
                      color=color, label=config_of(run).replace(f"{arm}-", ""))
        axis.set_title(f"{arm} (λ = 0.25)")
        axis.set_xlabel("fraction of tuning budget")
        axis.legend(fontsize=7, ncol=2)
    axes[0][0].set_ylabel("‖θ − θ_teacher‖₂")
    figure.suptitle("Distance to the teacher's weights, tuning runs (y-axes differ per arm)",
                    x=0.01, ha="left")
    figure.tight_layout()
    path = FIGURES / "tuning_teacher_distance.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def sweep_figures() -> list[Path]:
    sweep = [r for r in runs("full") if "anchor" not in config_of(r)]
    if not sweep:
        return []
    paths = []
    lambdas = sorted({r["final"]["lam"] for r in sweep})
    # 1. distance to teacher: one panel per arm, one line per λ > 0.
    arms = [arm for arm in ARMS if any(arm_of(r) == arm for r in sweep)]
    figure, axes = plt.subplots(1, len(arms), figsize=(3.2 * len(arms), 3.0), squeeze=False)
    for axis, arm in zip(axes[0], arms):
        for run in sorted([r for r in sweep if arm_of(r) == arm and r["final"]["lam"] > 0],
                          key=lambda r: r["final"]["lam"]):
            lam = run["final"]["lam"]
            axis.plot(fraction(run["rows"]), [row["teacher_distance"] for row in run["rows"]],
                      color=LAMBDA_COLORS.get(lam, INK_2), label=f"λ={lam:g}")
            label_end(axis, run["rows"], "teacher_distance", f"{lam:g}",
                      LAMBDA_COLORS.get(lam, INK_2))
        axis.set_title(arm)
        axis.set_xlabel("fraction of B_full")
    axes[0][0].set_ylabel("‖θ − θ_teacher‖₂")
    axes[0][-1].legend(fontsize=7)
    figure.suptitle("Distance to the teacher's weights, λ sweep (seed 0; y-axes differ per arm)",
                    x=0.01, ha="left")
    figure.tight_layout()
    paths.append(FIGURES / "sweep_teacher_distance.png")
    figure.savefig(paths[-1], dpi=150)
    plt.close(figure)
    # 2./3. validation agreement and KL against compute, one panel per λ, one line per arm.
    for key, label, name in (("val_agreement", "validation agreement", "sweep_agreement"),
                             ("val_kl", "validation KL(p_T ‖ p_S)", "sweep_kl")):
        figure, axes = plt.subplots(1, len(lambdas), figsize=(3.2 * len(lambdas), 3.0),
                                    sharey=True, squeeze=False)
        for axis, lam in zip(axes[0], lambdas):
            for index, arm in enumerate(ARMS):
                for run in [r for r in sweep if arm_of(r) == arm and r["final"]["lam"] == lam]:
                    axis.plot(fraction(run["rows"]), [row[key] for row in run["rows"]],
                              color=CATEGORICAL[index], label=arm)
                    label_end(axis, run["rows"], key, arm, CATEGORICAL[index])
            axis.set_title(f"λ = {lam:g}")
            axis.set_xlabel("fraction of B_full (FE)")
            if key == "val_kl":
                axis.set_yscale("log")
        axes[0][0].set_ylabel(label)
        axes[0][-1].legend(fontsize=7)
        figure.tight_layout()
        paths.append(FIGURES / f"{name}.png")
        figure.savefig(paths[-1], dpi=150)
        plt.close(figure)
    # 4. final agreement against λ.
    figure, axis = plt.subplots(figsize=(4.8, 3.2))
    for index, arm in enumerate(ARMS):
        members = sorted([r for r in sweep if arm_of(r) == arm], key=lambda r: r["final"]["lam"])
        if members:
            xs = [r["final"]["lam"] for r in members]
            ys = [r["final"]["val_agreement"] for r in members]
            axis.plot(xs, ys, color=CATEGORICAL[index], marker="o", markersize=5, label=arm)
    axis.set_xlabel("λ (teacher fraction in the initialization)")
    axis.set_ylabel("final validation agreement")
    axis.legend(fontsize=7)
    figure.tight_layout()
    paths.append(FIGURES / "sweep_agreement_vs_lambda.png")
    figure.savefig(paths[-1], dpi=150)
    plt.close(figure)
    return paths


RESTORATION = {
    "first_order": "model, optimizer state (momentum/Adam moments), data-sampler CUDA generator "
                   "state with epoch order and cursor, FE counter (hence the lr schedule, a "
                   "function of the FE fraction), t/T schedule state: restored exactly",
    "gauss_newton": "parameters, previous CG step (warm start), LM damping, data-sampler "
                    "generator state, FE counter, t schedule state: restored exactly",
}


def resumes() -> None:
    """Every run that was resumed from a checkpoint or restarted after a crash or kill."""

    events: dict[str, list[dict]] = {}
    for line in (RUNS / "index.jsonl").read_text().splitlines():
        row = json.loads(line)
        events.setdefault(row["run_id"], []).append(row)
    print("| run | resumed from checkpoint (step) | fresh restarts | finished | restored |")
    print("|---|---|---:|---|---|")
    for run_id, rows in sorted(events.items()):
        config = RUNS / run_id / "config.json"
        if not config.exists() or "config" not in json.loads(config.read_text()):
            continue  # smoke tests live under runs/smoke; the teacher has its own record
        archived = [i for i, row in enumerate(rows) if row["event"] == "archived"]
        rows = rows[archived[-1] + 1:] if archived else rows  # a new run under a reused id
        kinds = [row["event"] for row in rows]
        resumed = [str(row.get("step")) for row in rows if row["event"] == "resume"]
        restarts = max(0, kinds.count("start") - 1)
        if not resumed and not restarts:
            continue
        arm = json.loads(config.read_text())["config"]["arm"]
        kind = "gauss_newton" if arm in ("A3", "A4") else "first_order"
        print(f"| {run_id} | {', '.join(resumed) or '-'} | {restarts} | "
              f"{'yes' if 'end' in kinds else 'no'} | {RESTORATION[kind]} |")
    print("\nNot restored: work done after the last checkpoint is recomputed, and because cuDNN "
          "kernels are not bitwise deterministic the recomputed steps are not bit-identical to "
          "the lost ones. A6's cache of snapshot monitor logits is rebuilt (and charged again). "
          "A fresh restart (crash before the first checkpoint) repeats the run from its start "
          "with the same seeds.")


def train_table(budget: str = "full") -> None:
    """50k clean-train agreement and KL (train_eval.py) for every arm at every λ."""

    members = [r for r in runs(budget) if (r["dir"] / "train_full_eval.json").exists()]
    lambdas = sorted({r["final"]["lam"] for r in members})
    print("| arm | " + " | ".join(f"λ={lam:g} agree / KL" for lam in lambdas) + " |")
    print("|---|" + "---:|" * len(lambdas))
    for arm in ARMS:
        cells = []
        for lam in lambdas:
            found = [r for r in members if arm_of(r) == arm and r["final"]["lam"] == lam
                     and "anchor" not in config_of(r)]
            if found:
                result = json.loads((found[0]["dir"] / "train_full_eval.json").read_text())
                cells.append(f"{100 * result['train_agreement']:.2f}% / {result['train_kl']:.3f}")
            else:
                cells.append("–")
        if any(cell != "–" for cell in cells):
            print(f"| {arm} | " + " | ".join(cells) + " |")


def train_agreement_figure() -> Path | None:
    """Train agreement (50k, no augmentation) against λ per arm, cf. Stanton et al. Fig. 6(b)."""

    members = [r for r in runs("full") if (r["dir"] / "train_full_eval.json").exists()
               and "anchor" not in config_of(r)]
    if not members:
        return None
    figure, axis = plt.subplots(figsize=(5.2, 3.4))
    for index, arm in enumerate(ARMS):
        points = sorted((r["final"]["lam"], json.loads(
            (r["dir"] / "train_full_eval.json").read_text())["train_agreement"])
            for r in members if arm_of(r) == arm)
        if points:
            color = CATEGORICAL[index]
            axis.plot([p[0] for p in points], [p[1] for p in points], color=color, marker="o",
                      markersize=5, label=arm, linestyle="--" if arm == "A1-MSE" else "-")
    axis.set_xlabel("λ (θ_init = λ θ_T + (1 − λ) θ_R)")
    axis.set_ylabel("train agreement (50k, no augmentation)")
    axis.set_title("Train agreement vs λ (cf. Stanton et al. Fig. 6b; 200-epoch-equivalent "
                   "budget, λ ∈ {0, .1, .25, .4, .5})", fontsize=8)
    axis.legend(fontsize=7)
    figure.tight_layout()
    path = FIGURES / "sweep_train_agreement_vs_lambda.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def figures() -> None:
    FIGURES.mkdir(exist_ok=True)
    style()
    paths = [tuning_distance_figure(), *sweep_figures(), train_agreement_figure()]
    print("\n".join(str(path) for path in paths if path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("what", choices=["tuning", "figures", "resumes", "train"])
    arguments = parser.parse_args()
    {"tuning": tuning, "figures": figures, "resumes": resumes,
     "train": train_table}[arguments.what]()


if __name__ == "__main__":
    main()
