# Agent brief: soft Dream on Stanton's CIFAR-100 fidelity benchmark

Save this file as `CLAUDE.md` at the root of the `target-space-dream` repository on the pod.
It is loaded at the start of every Claude Code session, so re-read it whenever you resume work.

Two values to confirm with Fred in Phase 0: the GPU-hour cap (default **30 GPU-hours**) and
whether the thesis teacher checkpoint is available at `/workspace/inputs/teacher_state.pt`.

> **Status note (2026-09-30):** decisions and amendments that override this brief (cap now
> 40 GPU-hours, no thesis teacher, (1−λ) lr scaling, λ = 0.5 control, schedule changes) are in
> `experiments/stanton/PLAN.md` §6 and `PREREGISTRATION.md` §9. Those files take precedence.

---

## 0. How to work

- You are working for Fred Defokue, author of the master's thesis *High-Fidelity Distillation as
  Root Tracking* (ETH Zürich, 2026). This repository is its curated implementation.
- You run inside tmux on a RunPod GPU pod (RTX 4090, 24 GB). Persistent storage is `/workspace`.
  Anything outside `/workspace` can disappear when the pod restarts.
- Work in the phases of section 9. At the end of each phase, append a short report to
  `experiments/stanton/REPORT.md`, print a summary, then **stop and wait for Fred's go-ahead**.
  Never start the next phase on your own.
- Honesty over impressiveness. A clean negative result is a successful outcome. Never tune,
  filter, re-run selectively or reword results in a way that favours soft Dream.

## 1. The scientific question

In Stanton et al.'s identical-architecture CIFAR-100 self-distillation setting, does a *soft*
version of Dream (Gauss–Newton steps along a moving target) reach higher teacher fidelity than
well-tuned first-order distillation, at matched compute?

Background you must know:

- **Stanton et al. (NeurIPS 2021), "Does Knowledge Distillation Really Work?"**
  (https://arxiv.org/abs/2106.05945, code: https://github.com/samuelstanton/gnosis).
  Students often fail to match the teacher even on the distillation data; training longer or
  switching optimizers barely helps. When the student is initialized as
  `λ·θ_teacher + (1−λ)·θ_random`, students with λ > 0.25 reach near-perfect agreement and
  students with λ ≤ 0.25 do not. Read section 6 (optimization) and the gnosis distillation
  configs; mirror their baseline recipe where feasible and document every deviation (for example,
  this repo's `ResNet20GN1` uses GroupNorm(1, C) rather than their exact LayerNorm ResNet).
- **Thesis lessons.** Exact root-finding on tiny transfer sets fits them perfectly but does not
  generalize. Dense exact solves do not scale. The continuation path and step size select which
  solution is reached: finer steps selected closer, better-generalizing roots. A direct
  Gauss–Newton jump to the endpoint failed where continuation succeeded.
- **Hypothesis under test.** Stanton's failure at small λ is a basin-finding (global) problem.
  Moving the target gradually from the student's own function to the teacher's keeps the student
  near a solution path and may reach the teacher's basin from far away. Gauss–Newton steps may
  additionally help fit large, augmented distillation sets where first-order methods plateau.
- **Known confound.** From a random initialization with small logits, the path
  `y_t ≈ t·f_teacher`, i.e. the teacher at temperature 1/t. The continuation arms may therefore
  reduce to temperature annealing. Arm A5 exists to test this.

## 2. Repository orientation

- `src/dream/continuation.py`: exact fresh-Jacobian continuation. Note that the first
  Gauss–Newton correction from a root at `t` toward `y(t+dt)` is exactly the tangent predictor.
- `src/dream/solvers.py`: dense minimum-norm solve with live refinement, matrix-free LSQR.
- `src/dream/operators.py`: explicit and JVP/VJP Jacobian access (`torch.func`).
- `src/dream/targets.py`: straight target path, centered logits, Helmert contrast.
- `src/dream/models/resnet.py`: `ResNet20GN1` (~278k parameters).
- `experiments/cifar100_distillation.py`: the exact small-D driver. It needs private thesis
  artifacts; do not run it.
- Conventions (`CONTRIBUTING.md`): model-independent math in `src/dream`, orchestration in
  `experiments/`, no datasets, checkpoints or run directories in git, every new numerical
  component comes with a test.
- Start with `pip install -e '.[test,vision]'` and `pytest`. All tests must pass before and after
  your changes.

## 3. Fixed experimental setup

- **Data.** CIFAR-100 via torchvision into `/workspace/data`. Preprocessing identical to the
  thesis: uint8 → `x / 127.5 − 1` (no mean/std normalization). Augmentation: random crop 32 with
  padding 4 plus horizontal flip, performed on the GPU with the dataset resident in GPU memory.
- **Splits.**
  - Distillation set: all 50,000 training images (the teacher's own training data, as in
    Stanton), with augmentation.
  - Validation: a fixed, class-balanced 5,000 of the 10,000 official test images (seed 0). Used
    for all tuning and all Phase 1–3 reporting.
  - Test: the other 5,000 test images. **Opened only in Phase 4, after Fred approves.**
- **Teacher.** If `/workspace/inputs/teacher_state.pt` exists, load it into `ResNet20GN1` (the
  thesis teacher: 200 epochs, 63.74% test accuracy) and report its validation accuracy.
  Otherwise train one: `ResNet20GN1`, SGD momentum 0.9 (Nesterov), lr 0.1 with cosine decay,
  weight decay 5e-4, batch 128, 200 epochs, same augmentation, all 50k training images, seed 0.
  Record the recipe and its validation accuracy.
- **Student initialization.** `θ_init(λ, seed) = λ·θ_T + (1−λ)·θ_R(seed)`, with `θ_R` a fresh
  default initialization of `ResNet20GN1`. λ ∈ {0.0, 0.1, 0.25, 0.4}. λ = 0.4 is the positive
  control: first-order distillation should succeed there. If it does not, stop and report.
- **Precision.** float32 everywhere, TF32 disabled for all arms (identical numerics across arms).

## 4. Metrics

Log every run against compute, about every 2% of its budget:

- validation top-1 agreement with the teacher (primary);
- validation `KL(p_teacher ‖ p_student)` at temperature 1, mean over images, in nats (primary);
- validation student accuracy against labels;
- agreement and KL on a fixed 5,000-image subset of the training images, without augmentation
  (tells optimization failure apart from identification failure);
- parameter displacement `‖θ − θ_init‖₂`;
- compute consumed (section 5), wall-clock time and GPU-hours.

## 5. Compute accounting

- **Unit: forward-equivalents (FE) per image.** In Phase 1, measure on this GPU the cost of a
  VJP (backward) and a JVP relative to one forward pass of `ResNet20GN1` at the batch sizes used,
  and record it in `experiments/stanton/compute_calibration.json`.
- Count every pass in every arm: student forwards and backwards, JVPs and VJPs inside CG,
  line-search and trust-ratio evaluations, teacher forwards, and forwards of the frozen initial
  student used by the continuation targets.
- **Full budget** `B_full` = the FE consumed by 200 epochs of arm A1 on the 50k distillation set.
  Every arm gets exactly `B_full` in full runs. **Tuning budget** = `B_full / 4`.
- Keep a run index in `/workspace/runs/index.jsonl` and a budget ledger in
  `/workspace/runs/budget.json` (cumulative GPU-hours).

## 6. Arms

| Arm | Optimizer | Target |
|---|---|---|
| **A1** first-order direct | best of Adam / SGD+momentum | teacher, t = 1 throughout (Stanton baseline) |
| **A2** first-order continuation | A1's best optimizer | moving target `y_t` |
| **A3** Gauss–Newton direct | damped GGN-CG | teacher, t = 1 throughout |
| **A4** soft Dream | damped GGN-CG | moving target `y_t` |
| **A5** temperature annealing | A1's best optimizer | teacher at temperature `T_k → 1` |

**Moving target (A2, A4).** For each augmented image,
`y_t(x) = (1−t)·f_init(x) + t·f_teacher(x)` in logits, with `f_init` a frozen copy of the initial
student. Objective on a batch: mean `KL(softmax(y_t(x)) ‖ softmax(f_θ(x)))`.

**Soft Dream step (A3, A4).**
- Gradient `g = (1/|B|) Σ J_xᵀ (p_θ(x) − q_t(x))` on a batch B.
- Generalized Gauss–Newton product on a curvature batch C ⊆ B:
  `G v = (1/|C|) Σ J_xᵀ [(diag(p) − p pᵀ)(J_x v)]`, computed with `torch.func.jvp` then `vjp`
  (use `torch.func.linearize` if it is faster).
- Solve `(G + λ_d I) δ = −g` with K iterations of conjugate gradient, warm-started from
  `β·δ_prev` (β ≈ 0.9).
- Levenberg–Marquardt damping: `ρ = [L(θ+δ) − L(θ)] / [gᵀδ + ½ δᵀGδ]` on B.
  If ρ < 0.25, multiply λ_d by 1.5; if ρ > 0.75, multiply it by 2/3. Reject a step that increases
  the loss.
- A3 is identical with t fixed at 1.

**t schedules (shared by A2 and A4).**
- Fixed: t rises linearly from 0 to 1 over a fraction φ of the budget, then stays at 1.
- Adaptive (Dream-style step control): every M steps, compute `KL(q_t ‖ p_θ)` on a fixed monitor
  subset of 2,000 training images (not validation). If it is below a threshold τ, advance
  `t ← min(1, t + Δt)`, doubling Δt (up to a cap) after two consecutive advances. If it stays
  above τ for S consecutive checks, halve Δt. Once t = 1, continue at t = 1 until the budget ends.

**Temperature annealing (A5).** Target `softmax(f_teacher / T_k)`, with T_k decreasing
geometrically from T0 ∈ {4, 16} to 1 over a fraction φ of the budget; student at temperature 1.

## 7. Tuning protocol (fairness)

- Tune only at λ = 0.25, seed 0, at the tuning budget. Select by validation agreement, with KL as
  tie-breaker.
- At most 6 configurations per arm. Write every grid into `PREREGISTRATION.md` before running any
  tuning run. A4's grid may reuse A3's best numerics (K, λ_d init, curvature batch size) combined
  with schedule settings; document this.
- Then run each arm's selected configuration at full budget, seed 0, for all four λ values.
- Final runs: seeds 1 and 2 at λ ∈ {0.0, 0.25} for every arm.
- Seeds are fixed in advance. No run is ever discarded; failed and diverged runs are reported.

## 8. Pre-registration and decision rule

In Phase 0, draft `experiments/stanton/PREREGISTRATION.md` with the metrics, splits, budgets,
grids and the decision rule below. Commit it before any tuning run. Changes afterwards are dated
amendments with a stated reason, made before viewing the affected results wherever possible.

Default decision rule (3-seed means on the test split, at λ = 0 or λ = 0.25, at ≤ equal FE):

- **GO:** A4 beats the best of A1, A2, A3 and A5 by at least 1.0 percentage point of test
  agreement *and* by more than two pooled seed standard deviations, and has lower test KL.
- **Second order is the lever:** A3 and A4 are within two standard deviations of each other and
  both beat A1, A2 and A5 by the GO margin.
- **The path is the lever:** A2 and A4 are within two standard deviations and both beat A1, A3
  and A5 by the GO margin. Continuation works; CG is not needed.
- **Temperature suffices:** A5 matches A2 or A4 within two standard deviations. The effect is
  real but not new.
- **NO-GO:** no arm beats A1 by the GO margin.

## 9. Phases

**Phase 0: reconnaissance and plan.** No training runs.
Read the repo and its tests, Stanton section 6 and the gnosis distillation configs. Check the GPU,
drivers and environment; run `pytest`; download CIFAR-100; check for the teacher checkpoint.
Write `experiments/stanton/PLAN.md` (file layout, implementation plan, compute estimate against
the GPU-hour cap, and any deviations from Stanton) and the draft `PREREGISTRATION.md`.
**Stop.**

**Phase 1: infrastructure and baseline.**
GPU-resident data pipeline and augmentation, splits, evaluation harness, compute calibration,
teacher (load or train), arm A1. Sanity check: A1 with a default configuration at the tuning
budget for λ = 0.4 and λ = 0.25. Expect clearly higher agreement at 0.4.
**Stop and report.**

**Phase 2: methods and tests.**
Implement GGN products, damped CG, LM damping, the target path, both t schedules, and arms A2–A5.
Write the tests in section 10. Smoke-test every arm at a tiny budget.
**Stop and report.**

**Phase 3: tuning and λ sweep (validation only).**
Run the tuning protocol, then the full-budget λ sweep at seed 0. Produce plots of validation
agreement and KL against FE, and final agreement against λ, for all arms. Propose the final
configurations.
**Stop and wait for explicit approval before opening the test split.**

**Phase 4: final seeds and test.**
Run seeds 1 and 2, evaluate on the test split once, apply the decision rule, and write
`experiments/stanton/RESULTS.md` with plots, tables and an honest verdict, including failure
modes. Commit a compact summary JSON to `artifacts/summaries/`.
**Stop.**

## 10. Required tests

- The GGN-vector product equals an explicit `Jᵀ F J v` on a tiny model (float64, CPU).
- CG solves `(G + λI) x = b` on a small SPD system to relative residual 1e-8.
- Limit check: on a tiny underdetermined least-squares problem with identity output metric, the
  damped Gauss–Newton step with λ → 0 and CG run to convergence matches
  `dream.solvers.live_refined_minimum_norm` (float64).
- The KL gradient matches autograd.
- Target-path and temperature-path endpoints are correct.
- All pre-existing tests still pass.

## 11. Engineering rules

- Work on a new git branch `soft-dream-stanton` with small, descriptive commits. Do not push
  unless Fred asks. Never rewrite history or delete files you did not create.
- Suggested layout: `src/dream/ggn.py` (GGN products, CG, damping), `src/dream/schedules.py`
  (t and temperature schedules), `experiments/stanton/` (data, teacher, `distill.py --arm ...`,
  evaluation, plotting), `configs/stanton/*.json`.
- Run outputs go to `/workspace/runs/<run_id>/` (config, `metrics.jsonl`, `final.json`,
  checkpoint). Runs must be resumable from their last checkpoint.
- Launch long runs through a queue script inside tmux (or `nohup`) with log files. If a single run
  leaves the GPU under-utilized, run several concurrently, after measuring that per-run
  throughput does not collapse.
- Budget: hard cap of **30 GPU-hours** unless Fred changes it. Update the ledger after every run.
  Estimate the cost of every batch of runs before launching it, and stop and ask if the projected
  total would exceed the cap.

## 12. Stop and ask when

- a sanity check fails (tests, teacher accuracy, the λ = 0.4 positive control);
- numerical trouble (NaNs, CG divergence, exploding damping) persists after about 30 minutes of
  debugging;
- you want to change metrics, splits, budgets, grids or the decision rule;
- an action is destructive or irreversible, or would spend beyond the GPU-hour cap.
