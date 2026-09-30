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
  `ResNet20GN1` under `torch.manual_seed(1000 + s)`. λ ∈ {0.0, 0.1, 0.25, 0.4}. The data and
  augmentation stream of a run uses seed `s`.
- No learning-rate scaling with λ (deviation from gnosis, see PLAN.md §2).

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
- Hard cap: 30 GPU-hours in total (Fred may change it).

## 6. Arms and tuning grids

Tuning: λ = 0.25, seed 0, budget `B_full/4`. Selection by final validation agreement; ties
(within 0.1 pp) broken by lower validation KL. At most 6 configurations per arm, all listed here.
Every tuning run is reported.

Common: batch 128 for first-order arms; cosine learning-rate decay over the run's budget to 0
(Adam: to 0; SGD: to 1e-6 as in gnosis); Nesterov momentum 0.9 for SGD; Adam β = (0.9, 0.999).

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
Once t = 1, training continues at t = 1 until the budget ends. If t never reaches 1, the run is
still evaluated against the teacher (t = 1) and reported as such.

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

1. Phase 1 sanity: A1-1 at `B_full/4`, λ ∈ {0.4, 0.25}, seed 0. Gate: validation agreement at
   λ = 0.4 at least 5 pp above λ = 0.25; otherwise stop and report.
2. Tuning: all grids above, λ = 0.25, seed 0.
3. λ sweep: each arm's selected config at `B_full`, seed 0, λ ∈ {0, 0.1, 0.25, 0.4}. Positive
   control: A1 at λ = 0.4 must reach train-subset agreement ≥ 90%; otherwise stop and report.
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

None yet.
