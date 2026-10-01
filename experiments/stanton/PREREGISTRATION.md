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
- Parameter distance to the teacher `‖θ − θ_T‖₂`, logged next to `‖θ − θ_init‖₂` in every run
  (required for λ > 0; logged for all runs) (Amendment 3).
- Train-set fidelity (Amendment 5): top-1 agreement and mean `KL(p_T ‖ p_S)` at temperature 1 on
  all 50,000 distillation images without augmentation, for the final weights of each arm's
  selected tuning config, every full-budget run and every final-seed run. This KL is reported
  instead of the training loss, which differs across arms (e.g. logit MSE in A5-6). Evaluation
  time is charged to the ledger (not to the runs' FE budgets).
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
| A1-7 | SGD-Nesterov | 0.05 | 1e-4 | τ = 1, **linear lr warmup** from 0 to the peak over the first 10% of the budget, then cosine to 1e-6 (Amendment 3) |

"A1's best optimizer" (used by A2 and A5) means the optimizer, lr, weight decay and lr schedule
(including warmup, if A1-7 is selected) of the selected A1 config; the loss temperature is not
inherited. A1 has 7 configurations, one more than the brief's limit of 6 (Amendment 3).

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
| A5-4..5 | 16 | 0.25, 0.5 |
| A5-6 | Annealing-KD replication (below) | |

A5-6 (Amendment 3) follows Jafari et al. (2021), "Annealing Knowledge Distillation", CIFAR setup:
loss `mean((f_S − Φ(T)·f_T)²)` over batch and classes (MSE on logits, no softmax), with
`Φ(T) = 1 − (T − 1)/τ_max`, τ_max = 10, and T an integer stepping down from 10 to 1 in ten equal
stages over the first 50% of the budget (160 of their 320 epochs), followed by a final phase at
T = 1 (Φ = 1) for the remaining 50%. Deviations from the paper: their stage II fine-tunes on hard
labels with cross-entropy, ours stays label-free on the Φ = 1 MSE loss (as Fred specified; this
study never uses labels); we do not select the best checkpoint on validation (§4); the optimizer is
A1's best (fairness protocol), not their SGD lr 0.1, wd 1e-4 with the TAKD schedule.

### A6: teacher-trajectory continuation (Amendment 4)

Targets are the teacher's own training snapshots in order, from epoch 0 to epoch 200 (= the final
teacher), with spacing 5 or 10 epochs (n = 40 or 20 intervals). Loss
`KL(softmax(f_snapshot) ‖ softmax(f_S))`, A1's selected optimizer (as A2, including the (1−λ) lr
scaling). The snapshot index is advanced by the adaptive rule of A2 (threshold, relative-progress
rule, deadline ramp, logging) with t = k/n: initial step 1 snapshot, doubling after two
consecutive advances up to ⌈n/8⌉ snapshots, halving after three failed checks down to 1 snapshot.
Under the deadline ramp the target is snapshot ⌊t·n⌋ of the linearly ramped t. Each training step
charges one snapshot forward per image (in place of the teacher forward); the monitor logits of
each snapshot are charged once (2,000 forwards) when that snapshot becomes the target.

| # | spacing | τ_KL |
|---|---:|---:|
| A6-1..3 | 5 epochs | 0.05, 0.15, 0.4 |
| A6-4..6 | 10 epochs | 0.05, 0.15, 0.4 |

A6 is tuned at λ = 0.25 like the other arms and included in the λ sweep, the final seeds and the
hand-off.

## 7. Run plan and seeds

1. Phase 1 sanity: A1-1 at `B_full/4`, λ ∈ {0.5, 0.25}, seed 0. Gate: validation agreement at
   λ = 0.5 at least 5 pp above λ = 0.25; otherwise stop and report. (Amendment 1: was 0.4.)
2. Tuning: GGN-product variant chosen first (§5), then all grids above, λ = 0.25, seed 0.
3. λ sweep (arms A1–A6): each arm's selected config at `B_full`, seed 0, λ ∈ {0, 0.1, 0.25, 0.4, 0.5}.
   Positive control: A1 at λ = 0.5 must reach train-subset agreement ≥ 90%; otherwise stop and
   report. (Amendment 1: λ = 0.5 added and used as the control; 0.4 kept in the sweep.)
4. Finals: seeds 1 and 2, λ ∈ {0, 0.25}, for A1, A2, A5 and A6 (A3 and A4 dropped by
   Amendment 9), plus the diagnostics A1-MSE and the anchor run (μ = 0.01) (Amendment 10).
5. Test evaluation, once, after Fred has approved the Phase 3 report: the 3 seeds × 2 λ final
   models of A1, A2, A5, A6, A1-MSE and the anchor run, and the seed-0 models of A3 and A4.

Hand-off diagnostic (Amendments 3 and 4; not part of the decision rule): after the λ sweep, the
seed-0 endpoints of A2, A4 and A6 at λ ∈ {0, 0.25} are continued with A1's selected configuration
(its optimizer, weight decay and loss) for `B_full/4`, with fresh optimizer state and teacher
target. Two learning-rate variants, with `peak = lr·(1−λ)` of A1's selected config and the run's
original λ:

- **primary:** linear warmup over the first 10% of the hand-off budget to `0.1·peak`, then cosine
  to 0;
- **secondary:** linear warmup over the first 10% to the full `peak`, then cosine to 0.

As the control, A1's own endpoints at the same λ get both treatments. Reported: validation
agreement, KL, train-subset agreement and `‖θ − θ_T‖₂` after the hand-off, next to the values
before it.

A1-MSE diagnostic (Amendment 5; not part of the decision rule): A1's selected config (A1-7) with
the loss replaced by the logit MSE that A5-6 uses at T = 1, `mean((f_S − f_T)²)`, without
annealing. Only the learning rate is tuned, lr ∈ {0.02, 0.05, 0.1} (0.05 is A1-7's and A5-6's),
at λ = 0.25, seed 0, `B_full/4`, selected by the rule of §6. The selected config then runs at
`B_full` for λ ∈ {0, 0.1, 0.25, 0.4, 0.5} (seed 0) and in the final seeds 1 and 2 at
λ ∈ {0, 0.25}; its test results are reported next to the decision rule but do not enter it.
Purpose: separate the annealing effect from the loss effect in A5-6.

Patience check (Amendment 6; not part of the decision rule): at λ = 0.25, seed 0, A1-7 and A5-6
run at `4·B_full` (validation only). All schedules stretch with the budget (A1-7's warmup over
the first 10% and cosine decay over the whole run; A5-6's ten temperature stages over the first
50%). Logged about every 2% of the 4× budget (every 8% of `B_full`): validation agreement and KL,
and agreement and `KL(p_T ‖ p_S)` on all 50k distillation images without augmentation. Reported
against compute next to the 1× runs of the λ sweep at λ = 0.25. Questions: does A5-6 at 1×
still beat A1-7 at 4×, and do the two converge with more training? Note that a 4× run at
`B_full` is not a 1× run (its lr and temperature schedules are stretched), so the 1× comparison
uses the 1× sweep runs' endpoints.

Patience check at λ = 0 (Amendment 7; not part of the decision rule): as the Amendment 6 check,
at λ = 0, seed 0, for A1-7, A2-3 and A5-6 at `4·B_full` (validation and 50k clean-train metrics
against compute), next to the 1× λ-sweep runs at λ = 0. Fred's criterion for pursuing this line
further (recorded in advance, see Amendment 7): A2-3 or A5-6 at ≥ 95% train agreement and A1-7 at
least 5 points lower, all at the end of `4·B_full`.

Anchor diagnostic (Amendment 4; not part of the decision rule): at λ = 0.25, seed 0, `B_full`,
A1's selected config plus `(μ(s)/2)·‖θ − θ_init‖²` with `μ(s) = μ·max(0, 1 − s/0.3)` (linear decay
to 0 over the first 30% of the budget), μ ∈ {0.01, 0.1}. It tests whether merely staying close to
the initialization early reproduces a continuation effect.

## 8. Decision rule

Computed separately at λ = 0 and at λ = 0.25, from 3-seed means (seeds 0, 1, 2) of final test
agreement and test KL, all runs at exactly `B_full`.

Definitions: for arms X and Y, `sd_pool(X, Y) = sqrt((s_X² + s_Y²)/2)` with `s` the sample
standard deviation (n − 1 = 2) of test agreement over the 3 seeds. "X beats Y by the GO margin"
means `mean_X − mean_Y ≥ 1.0 pp` **and** `mean_X − mean_Y > 2·sd_pool(X, Y)`. "X and Y are within
two standard deviations" means `|mean_X − mean_Y| ≤ 2·sd_pool(X, Y)`.

- **GO:** A4 beats each of A1, A2, A3, A5 and A6 by the GO margin, and has lower mean test KL
  than each of them.
- **Second order is the lever:** A3 and A4 are within two SD, and both beat each of A1, A2, A5
  and A6 by the GO margin.
- **The path is the lever:** A2 and A4 are within two SD, and both beat each of A1, A3, A5 and A6
  by the GO margin.
- **Temperature suffices:** A5 is within two SD of A2 or of A4 (reported whenever A2 or A4 beats
  A1 by the GO margin).
- **Teacher trajectory suffices:** A6 is within two SD of A4, and both beat A1 by the GO margin.
- **NO-GO:** no arm beats A1 by the GO margin.

(A6 and the "teacher trajectory suffices" outcome were added by Amendment 4.)

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

### Amendment 3 (2026-09-30, before any tuning run; requested by Fred)

1. Teacher retrained with the same recipe and seed, saving weights at epoch 0 and every 5 epochs to
   `/workspace/runs/teacher_seed0/snapshots/`. It is the only teacher from now on. The first
   teacher (65.72% validation accuracy) is kept at `/workspace/runs/teacher_seed0_v1/` for
   reference, with the two Phase 1 sanity runs made against it (archived under
   `/workspace/runs/teacher_v1_runs/`). The two sanity runs are repeated against the new teacher.
   Because training is not bitwise deterministic, the new teacher differs slightly from the first.
2. A1 grid: A1-7 added (Stanton's settings plus linear warmup over the first 10% of the budget).
   Reason: a continuation effect has to beat warmup to count. Warmup enters the comparison through
   A1's tuning (the selected A1 config is the best of seven, warmup included).
3. A5-6 replaced by a faithful Annealing-KD configuration (was T0 = 16, φ = 0.75).
4. `‖θ − θ_T‖₂` logged over training.
5. Hand-off diagnostic added to §7.

### Record (2026-09-30, Phase 2; not an amendment)

GGN-product method chosen as Amendment 1 requires: **`linearize`** (lowest measured GPU cost over
the A3 grid: setup ≈8.6 FE, product ≈7.4–7.6 FE per image; `jvp_vjp` product ≈9.7 FE;
reverse-over-reverse product ≈41 FE at |C| ≥ 512). All three agree to ≈1e-7 relative (float32).
Recorded in `compute_calibration.json` (`ggn_method`, `ggn_grid_cost_fe`). `linearize` re-traces
the model on the CPU at every step (≈1.5 s), which is not GPU work and is absorbed by running
Gauss–Newton runs 8 at a time.

### Amendment 4 (2026-09-30, before any tuning run; requested by Fred)

1. Hand-off: primary variant warms up over 10% of the hand-off budget to 10% of A1's peak lr, then
   cosine to 0; secondary variant warms up to the full peak. A1's control gets both. Sources now
   include A6. (Replaces the Amendment 3 hand-off schedule, which restarted at the full peak lr
   and knocked a smoke-test student off its starting point.)
2. Arm A6, teacher-trajectory continuation, 6 configurations; in the sweep, finals and hand-off.
3. Decision rule: A6 is a competitor (Fred's decision): it joins the comparison sets of GO,
   "second order" and "path", and a "teacher trajectory suffices" outcome is added.
4. Anchor diagnostic at λ = 0.25 with μ ∈ {0.01, 0.1} (my choice of values: the anchor gradient
   μ·‖θ − θ_init‖ is then ≈0.1 and ≈1 at the displacement of 10 that A1 reaches early at λ = 0.25,
   i.e. a moderate and a strong pull relative to the distillation gradient).
5. Every Phase 3 report shows the distance-to-teacher curves of all λ > 0 runs of every arm.

### Amendment 5 (2026-09-30 20:26, before any A1-MSE run; requested by Fred)

Written before any A1-MSE run existed and before any full-budget result of the λ sweep was
inspected.

1. Train-set evaluation on all 50k distillation images without augmentation (agreement and
   `KL(p_T ‖ p_S)`), for the selected tuning configs, the λ sweep and the final seeds; this KL
   replaces "train loss" in all reports.
2. A1-MSE diagnostic arm (A1-7 with A5-6's logit MSE at T = 1, no annealing), lr ∈ {0.02, 0.05,
   0.1} at the tuning budget, then the λ sweep and the final seeds at λ ∈ {0, 0.25}. Not part of
   the decision rule.
3. Reports list every run resumed after a crash or kill and state whether its optimizer, lr
   schedule and RNG states were restored exactly.

Note: the 50k train evaluation of the selected A1, A2, A5 and A6 tuning configs was run
(Fred's request) shortly before this amendment was written; no A1-MSE result existed.

### Amendment 6 (2026-09-30, before any patience-check run; requested by Fred)

Patience check: A1-7 and A5-6 at λ = 0.25, seed 0, `4·B_full`, validation and 50k clean-train
metrics against compute (§7). Diagnostic only. Written before either run existed and before I had
looked at the 1× λ-sweep results of A1-7 or A5-6. Cost estimate ≈2.4 GPU-h.

### Amendment 7 (2026-09-30, before any λ = 0 patience-check run; requested by Fred)

Patience check at λ = 0 for A1-7, A2-3 and A5-6 at `4·B_full` (§7). Fred's message called this
"Amendment 6"; since the λ = 0.25 check of Amendment 6 was already recorded and running, it is
recorded as an additional amendment. Written before these runs existed and before I had looked at
any λ = 0 result of the λ sweep. Cost ≈3.5 GPU-h; projected total with it ≈32 GPU-h (≈36 with a
15% margin) of the 40-hour cap, so all three runs are made (Fred's fallback: A1-7 and A5-6 only).

Addendum (2026-09-30, Fred; recorded before any λ = 0 patience run had started): criterion for
pursuing this line further, **not part of the decision rule**. It is met if A2-3 or A5-6 reaches
at least 95.0% train agreement (all 50k distillation images, no augmentation, final weights at the
end of `4·B_full`) at λ = 0, and A1-7's train agreement at the end of its `4·B_full` run is at
least 5.0 percentage points lower than that arm's. Both qualifying arms are reported if both
qualify.

### Amendment 8 (2026-10-01, before any A1-1 full-budget run; requested by Fred)

Diagnostic, not part of the decision rule: A1-1 (Stanton's exact optimizer recipe, no warmup)
at `B_full`, seed 0, λ ∈ {0, 0.1, 0.25, 0.4, 0.5}, with validation and 50k clean-train metrics.
Purpose: locate the λ barrier under Stanton's recipe in our setup, and separate the warmup effect
(A1-7 vs A1-1 at equal budget) from the budget effect (A1-1 at `B_full/4` vs `B_full`).

### Amendment 9 (2026-10-01, before any final-seed run; requested by Fred)

A3-6 and A4-6 are dropped from the extra seeds. Reason: at seed 0 they trail every first-order
arm by about 20 pp of validation agreement (e.g. 51% vs 71–73% at λ = 0), far beyond the
≈0.5 pp rerun noise, so no outcome of the decision rule can plausibly depend on them. Their seed-0
results stay in the report, and their seed-0 models are evaluated on the test split once.

Consequences for §8, fixed now, before any test result exists: outcomes that require A3 or A4
to beat or to match another arm ("GO", "second order is the lever", "the path is the lever",
"teacher trajectory suffices") are evaluated with their seed-0 test agreement in place of a
3-seed mean. Each such outcome is recorded as **not met** if the seed-0 test agreement of the
required A3/A4 run is more than 10 pp below the 3-seed mean of A1; otherwise it is flagged as
undecidable and reported to Fred. All other comparisons use 3-seed statistics as before.

Note: as defined in §8, "the path is the lever" and "teacher trajectory suffices" are stated
relative to A4, so with A4 far behind they cannot be met. Comparisons of A2 and A6 against A1
and A5 are reported descriptively with the same GO-margin arithmetic; they do not create a new
outcome unless Fred amends §8 before the test split is opened.

### Amendment 10 (2026-10-01, before any of these runs; requested by Fred)

Diagnostics in the final seeds, not part of the decision rule: A1-MSE (A1MSE-1) and the anchor
run (A1-anchor-mu0.01) at seeds 1 and 2, λ ∈ {0, 0.25}. The anchor at λ = 0 tests whether its
benefit depends on the teacher component of the initialization. Because the anchor diagnostic ran
only at λ = 0.25, a seed-0 anchor run at λ = 0, `B_full` is added so both λ have three seeds.
All seed-0 results of these arms were known when this amendment was written (Phase 3 summary).

