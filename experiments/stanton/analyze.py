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
    axis.set_title("Train agreement vs λ, seed 0 (cf. Stanton et al. Fig. 6b)", fontsize=9)
    axis.legend(fontsize=7, loc="lower right", ncol=2)
    figure.tight_layout()
    path = FIGURES / "sweep_train_agreement_vs_lambda.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def patience_figure() -> Path | None:
    """4×-budget runs (Amendments 6, 7) against compute, with the 1× sweep endpoints marked."""

    long_runs = [load_run(d) for d in sorted(RUNS.glob("*_s0_4"))
                 if d.name.split("_lam")[0] in ("A1-7", "A2-3", "A5-6")]  # Amendments 6, 7
    long_runs = [r for r in long_runs if r]
    if not long_runs:
        return None
    lambdas = sorted({r["final"]["lam"] for r in long_runs})
    keys = (("val_agreement", "validation agreement"), ("train50k_agreement", "train agreement (50k)"),
            ("train50k_kl", "train KL(p_T ‖ p_S) (50k)"))
    figure, axes = plt.subplots(len(lambdas), 3, figsize=(10.5, 3.0 * len(lambdas)), squeeze=False)
    for row_axes, lam in zip(axes, lambdas):
        for axis, (key, label) in zip(row_axes, keys):
            for run in [r for r in long_runs if r["final"]["lam"] == lam]:
                arm = arm_of(run)
                color = CATEGORICAL[ARMS.index(arm)]
                xs = [4 * row["fraction"] for row in run["rows"] if key in row]
                ys = [row[key] for row in run["rows"] if key in row]
                axis.plot(xs, ys, color=color, label=f"{config_of(run)} 4×")
                one = RUNS / f"{config_of(run)}_lam{lam:g}_s0_full"
                if (one / "train_full_eval.json").exists():
                    final = json.loads((one / "final.json").read_text())
                    train = json.loads((one / "train_full_eval.json").read_text())
                    value = {"val_agreement": final["val_agreement"],
                             "train50k_agreement": train["train_agreement"],
                             "train50k_kl": train["train_kl"]}[key]
                    axis.plot([1.0], [value], marker="D", markersize=7, color=color,
                              markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none",
                              label=f"{config_of(run)} 1× end")
            if key == "train50k_kl":
                axis.set_yscale("log")
            axis.set_title(f"λ = {lam:g}: {label}")
            axis.set_xlabel("compute (multiples of B_full)")
        row_axes[0].legend(fontsize=7)
    figure.tight_layout()
    path = FIGURES / "patience_4x.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def amendment11_figure() -> Path | None:
    """POST-HOC λ = 0, 4× runs (Amendments 7 and 11) against compute."""

    names = [("A1-7_lam0_s0_4", "A1-7 s0", 0), ("A1-7_lam0_s1_4", "A1-7 s1", 0),
             ("A2-3_lam0_s0_4", "A2-3 s0", 1), ("A5-6_lam0_s0_4", "A5-6 s0", 4),
             ("A5-6_lam0_s1_4", "A5-6 s1", 4), ("A5-6_lam0_s2_4", "A5-6 s2", 4),
             ("A1MSE-1_lam0_s0_4", "A1-MSE s0", 6), ("A1-7-tau4_lam0_s0_4", "A1-7 τ=4 s0", 7)]
    runs_found = [(load_run(RUNS / n), label, c) for n, label, c in names]
    if any(r is None for r, _, _ in runs_found):
        return None
    keys = (("val_agreement", "validation agreement"),
            ("train50k_agreement", "train agreement (50k)"),
            ("train50k_kl", "train KL(p_T ‖ p_S) (50k)"))
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    styles = {}
    for run, label, color_index in runs_found:
        seen = styles.setdefault(color_index, 0)
        styles[color_index] += 1
        for axis, (key, title) in zip(axes, keys):
            rows = [row for row in run["rows"] if key in row]
            axis.plot([4 * row["fraction"] for row in rows], [row[key] for row in rows],
                      color=CATEGORICAL[color_index], label=label,
                      linestyle=("-", "--", ":")[seen % 3])
            axis.set_title(f"λ = 0, 4× budget: {title}", fontsize=9)
            axis.set_xlabel("compute (multiples of B_full)")
    axes[2].set_yscale("log")
    axes[0].legend(fontsize=7, ncol=2)
    figure.text(0.01, 0.01, "POST-HOC (Amendment 11); A1-7 and A5-6 seed 0 are Amendment 7 runs",
                fontsize=7, color=INK_2)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    path = FIGURES / "posthoc_lambda0_4x.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def followup_figure() -> Path | None:
    """Follow-up (FOLLOWUP_PREREG.md): λ = 0, 4×; main arms by seed, and the τ sweep at seed 0."""

    main = [("A1-7", "A1-7 τ=1", 0), ("A1-7-tau8", "fixed τ=8", 7), ("A5-6", "A5-6", 4),
            ("A5-short", "A5-short", 6)]
    sweep = [("A1-7-tau2", "τ=2", "#86b6ef"), ("A1-7-tau4", "τ=4", "#3987e5"),
             ("A1-7-tau8", "τ=8", "#1c5cab"), ("A1-7-tau16", "τ=16", "#0d366b")]
    keys = (("val_agreement", "validation agreement"), ("train50k_agreement", "train agreement (50k)"),
            ("train50k_kl", "train KL(p_T ‖ p_S) (50k)"))
    if not all((RUNS / f"{c}_lam0_s{s}_4" / "final.json").exists()
               for c, _, _ in main for s in (0, 1, 2)):
        return None
    figure, axes = plt.subplots(2, 3, figsize=(12, 6.6))
    for row, entries in ((0, main), (1, sweep)):
        for config, label, colour in entries:
            seeds = (0, 1, 2) if row == 0 else (0,)
            for seed in seeds:
                run = load_run(RUNS / f"{config}_lam0_s{seed}_4")
                color = CATEGORICAL[colour] if row == 0 else colour
                for axis, (key, title) in zip(axes[row], keys):
                    rows = [r for r in run["rows"] if key in r]
                    axis.plot([4 * r["fraction"] for r in rows], [r[key] for r in rows],
                              color=color, linestyle=("-", "--", ":")[seed],
                              label=f"{label} s{seed}" if row == 0 else label)
                    axis.set_title(title + (" (seeds 0–2)" if row == 0 else " (τ sweep, seed 0)"),
                                   fontsize=9)
                    axis.set_xlabel("compute (multiples of B_full)")
        axes[row][2].set_yscale("log")
        axes[row][0].legend(fontsize=6.5, ncol=2 if row == 0 else 1)
    figure.suptitle("Follow-up: λ = 0, 4× budget (FOLLOWUP_PREREG.md)", x=0.01, ha="left")
    figure.tight_layout()
    path = FIGURES / "followup_lambda0_4x.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def handoff_table() -> None:
    """Hand-off diagnostic: endpoint before, and after both lr variants (validation, 50k train)."""

    print("| source endpoint | λ | before: val / train50k | primary: val / train50k | "
          "secondary: val / train50k |")
    print("|---|---:|---|---|---|")
    for directory in sorted(RUNS.glob("*_s0_full")):
        if "_from-" in directory.name or not (directory / "final.json").exists():
            continue
        variants = {}
        for variant in ("primary", "secondary"):
            hits = list(RUNS.glob(f"*_tune_ho-{variant}_from-{directory.name}"))
            if hits and (hits[0] / "final.json").exists():
                variants[variant] = hits[0]
        if not variants:
            continue

        def cell(path: Path) -> str:
            final = json.loads((path / "final.json").read_text())
            train_path = path / "train_full_eval.json"
            train = (f"{100 * json.loads(train_path.read_text())['train_agreement']:.2f}%"
                     if train_path.exists() else "–")
            return f"{100 * final['val_agreement']:.2f}% / {train}"

        lam = json.loads((directory / "final.json").read_text())["lam"]
        print(f"| {directory.name.split('_lam')[0]} | {lam:g} | {cell(directory)} | "
              + " | ".join(cell(variants[v]) if v in variants else "–"
                           for v in ("primary", "secondary")) + " |")


def test_figure() -> Path | None:
    """Test agreement per seed (dots) and 3-seed mean (bar mark) per arm, one panel per λ."""

    path_in = RUNS / "test_results.json"
    if not path_in.exists():
        return None
    results = json.loads(path_in.read_text())["results"]
    order = [("A1-7", "A1"), ("A2-3", "A2"), ("A3-6", "A3"), ("A4-6", "A4"), ("A5-6", "A5"),
             ("A6-3", "A6"), ("A1MSE-1", "A1-MSE*"), ("A1-anchor-mu0.01", "anchor*")]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for axis, lam in zip(axes, (0.0, 0.25)):
        for index, (config, label) in enumerate(order):
            values = [r["test_agreement"] for r in results
                      if r["config"] == config and r["lam"] == lam]
            color = CATEGORICAL[index % len(CATEGORICAL)]
            axis.scatter([index] * len(values), values, s=36, color=color, zorder=3,
                         edgecolors=SURFACE, linewidths=1.5)
            mean = sum(values) / len(values)
            axis.plot([index - 0.3, index + 0.3], [mean, mean], color=INK_2, linewidth=2)
        axis.set_xticks(range(len(order)), [label for _, label in order], fontsize=8)
        axis.set_title(f"λ = {lam:g}: test agreement (dots: seeds; bar: mean)")
    axes[0].set_ylabel("test agreement with the teacher")
    figure.text(0.01, 0.01, "* diagnostics, not part of the decision rule; A3 and A4: seed 0 only",
                fontsize=7, color=INK_2)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    path = FIGURES / "test_agreement_per_seed.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def figures() -> None:
    FIGURES.mkdir(exist_ok=True)
    style()
    paths = [tuning_distance_figure(), *sweep_figures(), train_agreement_figure(),
             patience_figure(), test_figure(), amendment11_figure(), followup_figure()]
    print("\n".join(str(path) for path in paths if path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("what", choices=["tuning", "figures", "resumes", "train", "handoff"])
    arguments = parser.parse_args()
    {"tuning": tuning, "figures": figures, "resumes": resumes, "train": train_table,
     "handoff": handoff_table}[arguments.what]()


if __name__ == "__main__":
    main()
