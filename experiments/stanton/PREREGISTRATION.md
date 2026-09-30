# Preregistration: soft Dream on Stanton's CIFAR-100 fidelity benchmark

Draft of 2026-09-30 (Phase 0). This file is committed before any tuning run. Any later change is
added as a dated amendment in §9 with its reason, made before viewing the affected results
wherever possible. The scientific question and arm definitions are those of `CLAUDE.md` §1 and §6.

## 1. Question

In identical-architecture CIFAR-100 self-distillation (`ResNet20GN1` teacher and student), does
soft Dream (A4: damped Gauss–Newton–CG steps along a moving logit target) reach higher teacher
fidelity than well-tuned first-order distillation, continuation without second order, second
order without continuation, and temperature annealing, at equal compute?

## 2. Data and splits

- CIFAR-100 (torchvision), preprocessing `x/127.5 − 1`, no mean/std normalization.
- Distillation set: all 50,000 training images, random crop 32 (zero padding 4) and horizontal
  flip, applied on the GPU.
- Validation: 5,000 test images, 50 per class, chosen with seed 0. Used for all tuning and all
  Phase 1–3 reporting.
- Test: the other 5,000 test images. Opened once, in Phase 4, after Fred's approval.
- Train-subset: 5,000 training images, 50 per class, seed 0, without augmentation (evaluation only).
- Monitor set: 2,000 training images, 20 per class, seed 0, disjoint from the train-subset,
  without augmentation (used only by the adaptive t schedule).

## 3. Teacher and student initialization

- Teacher: `ResNet20GN1` trained here (the thesis checkpoint is unavailable): Nesterov SGD,
  momentum 0.9, lr 0.1 cosine to 0, weight decay 5e-4, batch 128, 200 epochs, same augmentation,
  seed 0, float32, TF32 off. Its validation accuracy is reported. Sanity gate: validation accuracy
  ≥ 55%; otherwise stop and report.
- Student: `θ_init(λ, s) = λ·θ_T + (1−λ)·θ_R(1000 + s)`, `θ_R` the default PyTorch initialization of
  `ResNet20GN1` under `torch.manual_seed(1000 + s)`. λ ∈ {0.0, 0.1, 0.25, 0.4, 0.5}. The data and
  augmentation stream of a run uses seed `s`.
- Learning-rate scaling as in gnosis, for the first-order arms (A1, A2, A5) only: the peak
  learning rate is `max(lr·(1−λ), lr_min)`, with `lr` the configured value below and `lr_min` the
  cosine floor (1e-6 for SGD, 0 for Adam). The Gauss–Newton arms (A3, A4) have no learning rate.
  (Amendment 1.)

## 4. Metrics

Logged about every 2% of each run's budget, and at the end:

- **Primary:** top-1 agreement with the teacher on validation; mean `KL(p_T ‖ p_S)` at
  temperature 1 on validation, in nats.
- Secondary: validation accuracy against labels; agreement and KL on the train-subset; parameter
  displacement `‖θ − θ_init‖₂`; FE consumed, wall-clock time, GPU-hours.
- A run's result is its metrics at the end of its budget (the last evaluation). No early-stopping
  or best-checkpoint selection.

## 5. Compute

- Unit: forward-equivalents (FE) per image, from `compute_calibration.json` (measured in
  Phase 1 on the A40, float32, TF32 off, at the batch sizes used).
- Charged: every student forward, backward, JVP and VJP (including inside CG), every trial /
  line-search / trust-ratio evaluation, every teacher forward and frozen-initial-student forward
  on augmented batches, and the adaptive schedule's monitor evaluations. Not charged: the
  periodic evaluation of §4 (identical for all arms).
- `B_full` = FE of 200 epochs of A1 on the 50k distillation set (batch 128: student fwd+bwd plus
  teacher forward per image). Every full run gets exactly `B_full`; tuning runs get `B_full/4`.
  A run stops at the first step that would exceed its budget.
- Secondary count (reported, not used for budgets or the decision rule): the same tally with every
  JVP charged as 2 forward passes instead of its measured cost. (Amendment 1.)
- Before any tuning run, the GGN-vector product is implemented with `torch.func.jvp`+`vjp`,
  `torch.func.linearize` (one linearization per step, reused across the K CG iterations), and
  reverse-over-reverse (J v as the VJP of a VJP); the cheapest exact variant on the A40 is used for
  all Gauss–Newton runs, and the choice and timings are recorded in `compute_calibration.json`.
  If linearization is shared across CG iterations, its one-off cost is charged once per step.
  (Amendment 1.)
- Hard cap: 40 GPU-hours in total (Fred may change it). (Amendment 1.)

## 6. Arms and tuning grids

Tuning: λ = 0.25, seed 0, budget `B_full/4`. Selection by final validation agreement; ties
(within 0.1 pp) broken by lower validation KL. At most 6 configurations per arm, all listed here.
Every tuning run is reported.

Common: batch 128 for first-order arms; cosine learning-rate decay over the run's budget to 0
(Adam: to 0; SGD: to 1e-6 as in gnosis); Nesterov momentum 0.9 for SGD; Adam β = (0.9, 0.999).
Learning rates in the tables are before the (1−λ) scaling of §3; at the tuning λ = 0.25 they are
multiplied by 0.75.

### A1: first-order, direct (teacher, t = 1)

Loss `KL(softmax(f_T) ‖ softmax(f_S))` unless stated.

| # | Optimizer | lr | weight decay | Loss |
|---|---|---|---|---|
| A1-1 | SGD-Nesterov | 0.05 | 1e-4 | τ = 1 (Stanton's optimizer recipe) |
| A1-2 | SGD-Nesterov | 0.1 | 1e-4 | τ = 1 |
| A1-3 | SGD-Nesterov | 0.05 | 0 | τ = 1 |
| A1-4 | SGD-Nesterov | 0.05 | 1e-4 | Stanton/gnosis loss at τ = 4: `τ²·CE(softmax(f_T/τ), softmax(f_S/τ))` |
| A1-5 | Adam | 1e-3 | 0 | τ = 1 |
| A1-6 | Adam | 3e-4 | 0 | τ = 1 |

"A1's best optimizer" (used by A2 and A5) means the optimizer, lr and weight decay of the selected
A1 config; the loss temperature is not inherited.

### A2: first-order, moving target

Target `y_t = (1−t)·f_init + t·f_T` in logits, loss `KL(softmax(y_t) ‖ softmax(f_S))`, A1's best
optimizer. Schedules:

| # | Schedule |
|---|---|
| A2-1..3 | fixed: `t = min(1, s/φ)`, s = fraction of budget consumed, φ ∈ {0.25, 0.5, 0.75} |
| A2-4..6 | adaptive, threshold τ_KL ∈ {0.05, 0.15, 0.4} nats |

Adaptive schedule (shared with A4): t starts at 0. A check happens every 0.5% of the budget:
compute mean `KL(q_t ‖ p_θ)` on the monitor set (teacher and f_init logits on the fixed monitor
images are computed once and charged once; each check charges 2,000 student forwards). If it is
below τ_KL, `t ← min(1, t + Δt)`; after two consecutive advances Δt doubles, up to 1/8. If it is
above τ_KL for 3 consecutive checks, Δt halves, down to a floor of 1/1024. Initial Δt = 1/64.

Relative-progress rule (Amendment 1): the first check after each advance records a reference
`KL_ref` at the new t. A later check at the same t also triggers an advance when its KL is at most
`KL_ref / 2`, even if it is above τ_KL. Such an advance counts as an advance for the doubling rule.

Deadline (Amendment 1): if t < 1 when 60% of the budget has been consumed, the controller stops and
t ramps linearly from its current value to 1 by 70% of the budget, so that at least 30% of the
budget runs at t = 1.

Every advance is logged with its reason (`threshold`, `relative` or `deadline`), the budget
fraction, t before and after, Δt and the monitor KL. Relative and deadline advances are "forced"
advances and are summarized per run. Once t = 1, training continues at t = 1 until the budget ends.

### A3: damped GGN-CG, direct (teacher, t = 1)

Step as in `CLAUDE.md` §6: gradient on batch B, GGN of the mean softmax-KL on curvature batch
C ⊆ B, K CG iterations on `(G + λ_d I) δ = −g` warm-started from `0.9·δ_prev`, Levenberg–Marquardt
ratio on B (λ_d ×1.5 if ρ < 0.25, ×2/3 if ρ > 0.75), steps that increase the loss on B are
rejected. Initial λ_d = 1.0. No weight decay, no learning rate.

| # | |B| | |C| | K |
|---|---:|---:|---:|
| A3-1 | 256 | 64 | 5 |
| A3-2 | 512 | 128 | 10 |
| A3-3 | 512 | 128 | 25 |
| A3-4 | 2048 | 512 | 10 |
| A3-5 | 2048 | 512 | 25 |
| A3-6 | 512 | 512 | 10 |

### A4: soft Dream (damped GGN-CG, moving target)

A3's selected numerics (|B|, |C|, K, initial λ_d, β) combined with the six schedules of A2
(fixed φ ∈ {0.25, 0.5, 0.75}; adaptive τ_KL ∈ {0.05, 0.15, 0.4}). A4's grid therefore depends on
A3's tuning result by design (brief §7).

### A5: temperature annealing

Target `softmax(f_T / T)`, student at temperature 1, loss `KL(softmax(f_T/T) ‖ softmax(f_S))`,
A1's best optimizer. `T = T0^(1 − s/φ)` for s ≤ φ, then T = 1.

| # | T0 | φ |
|---|---:|---:|
| A5-1..3 | 4 | 0.25, 0.5, 0.75 |
| A5-4..6 | 16 | 0.25, 0.5, 0.75 |

## 7. Run plan and seeds

1. Phase 1 sanity: A1-1 at `B_full/4`, λ ∈ {0.5, 0.25}, seed 0. Gate: validation agreement at
   λ = 0.5 at least 5 pp above λ = 0.25; otherwise stop and report. (Amendment 1: was 0.4.)
2. Tuning: GGN-product variant chosen first (§5), then all grids above, λ = 0.25, seed 0.
3. λ sweep: each arm's selected config at `B_full`, seed 0, λ ∈ {0, 0.1, 0.25, 0.4, 0.5}.
   Positive control: A1 at λ = 0.5 must reach train-subset agreement ≥ 90%; otherwise stop and
   report. (Amendment 1: λ = 0.5 added and used as the control; 0.4 kept in the sweep.)
4. Finals: seeds 1 and 2, λ ∈ {0, 0.25}, every arm.
5. Test evaluation of the 3 seeds × 2 λ × 5 arms final models, once.

Seeds are fixed as above. No run is discarded; failed, diverged or crashed-then-resumed runs
are reported as such.

## 8. Decision rule

Computed separately at λ = 0 and at λ = 0.25, from 3-seed means (seeds 0, 1, 2) of final test
agreement and test KL, all runs at exactly `B_full`.

Definitions: for arms X and Y, `sd_pool(X, Y) = sqrt((s_X² + s_Y²)/2)` with `s` the sample
standard deviation (n − 1 = 2) of test agreement over the 3 seeds. "X beats Y by the GO margin"
means `mean_X − mean_Y ≥ 1.0 pp` **and** `mean_X − mean_Y > 2·sd_pool(X, Y)`. "X and Y are within
two standard deviations" means `|mean_X − mean_Y| ≤ 2·sd_pool(X, Y)`.

- **GO:** A4 beats each of A1, A2, A3 and A5 by the GO margin, and has lower mean test KL than
  each of them.
- **Second order is the lever:** A3 and A4 are within two SD, and both beat each of A1, A2 and A5
  by the GO margin.
- **The path is the lever:** A2 and A4 are within two SD, and both beat each of A1, A3 and A5 by
  the GO margin.
- **Temperature suffices:** A5 is within two SD of A2 or of A4 (reported whenever A2 or A4 beats
  A1 by the GO margin).
- **NO-GO:** no arm beats A1 by the GO margin.

All outcomes that apply are reported for both λ values; none is suppressed. A GO at only one of
the two λ values is reported as such, together with the other λ's outcome (two comparisons are
made, so a single-λ GO is weaker evidence). Validation results and train-subset results are
reported alongside but do not enter the rule.

## 9. Amendments

### Amendment 1 (2026-09-30, before any training run; requested by Fred)

1. Budget cap 30 → 40 GPU-hours.
2. gnosis's (1−λ) learning-rate scaling for A1, A2 and A5, because Stanton's λ threshold was
   measured with it (reverses decision D5 in `PLAN.md`).
3. λ = 0.5 added to the sweep and used as the positive control (Phase 1 check: 0.5 vs 0.25;
   train-agreement gate at 0.5). λ = 0.4 stays in the sweep. Reason: 0.4 sits close to Stanton's
   transition (between 0.25 and 0.375 in their setup, which differs from ours).
4. Adaptive t schedule: relative-progress advance (KL halved since the last advance), and a
   deadline ramp to t = 1 between 60% and 70% of the budget; all forced advances logged. Reason:
   a threshold-only controller can stall below t = 1 and never distil the teacher.
5. GGN-product implementation chosen by measured cost among jvp+vjp, linearize and
   reverse-over-reverse before tuning; secondary compute count with JVP = 2 FE.
6. Commits at least hourly, each followed by the backup bundle (process, not analysis).

### Amendment 2 (2026-09-30, before any distillation run; decided by Fred)

FE ratios are measured as CUDA-graph replay times (GPU work only), not eager timings. Reason: at
batch 128 an eager forward+backward measured 5.2–5.5 FE because it is CPU launch-bound, whereas its
GPU work is 3.2 FE (as at batch ≥ 512). Eager ratios would raise `B_full` from ≈4.2·10⁷ to
≈6.2·10⁷ FE and give the large-batch Gauss–Newton arms ≈1.5× more real GPU compute than the
batch-128 first-order arms. Eager timings stay in `compute_calibration.json` for reference, and
wall-clock time is reported per run.
