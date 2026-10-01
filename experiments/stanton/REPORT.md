# Phase reports

## Phase 0: reconnaissance and plan (2026-09-30)

- Repo read (continuation, solvers, operators, targets, ResNet20GN1, tests); `pytest`: 10/10 pass.
- Environment: NVIDIA A40 46 GB (not the RTX 4090 of the brief), torch 2.8.0+cu128. The repo sits
  on the non-persistent container disk because git cannot run on `/workspace` (geesefs). It is
  backed up as a git bundle in `/workspace/backup/` after each commit.
- CIFAR-100 downloaded to `/workspace/data` (official md5).
- Teacher checkpoint absent and unavailable. A new teacher will be trained in Phase 1.
- Stanton §6 and the gnosis configs read. Deviations are listed in `PLAN.md` §2. The most
  relevant: gnosis distils at τ = 4 (now one A1 config); gnosis scales lr by (1−λ) (not done).
- Throughput probe: ≈60k forward images/s; fwd+bwd ≈3.2 FE; `jvp` ≈6.7 FE; GGN product via
  `jvp`+`vjp` ≈10.7 FE at batch ≥ 512.
- Compute estimate ≈26 of 30 GPU-hours without concurrency gains (`PLAN.md` §5).
- `PLAN.md` and draft `PREREGISTRATION.md` written. Decisions D1–D6 with Fred in `PLAN.md` §6.
- GPU-hours used so far: ≈0 (probe only).

## Phase 0 addendum (2026-09-30)

Amendment 1 (Fred): cap 40 GPU-h; (1−λ) lr scaling for A1/A2/A5; λ = 0.5 added as positive
control; adaptive-schedule relative-progress and deadline rules; GGN-variant selection before
tuning; hourly commits. Amendment 2 (Fred): FE ratios from CUDA-graph GPU time.

## Phase 1: infrastructure and baseline (2026-09-30)

**Built.** `data.py` (CIFAR-100 on the GPU, zero-pad-4 crop + flip as one gather, verified
against a per-image reference; frozen seed-0 splits, digest `9f63b20c490c3709`; the test half is
only loaded with `open_test=True`), `teacher.py`, `calibrate.py`, `evaluate.py`, `distill.py`
(arm A1; resumable; metrics every 2%, checkpoint every 4%), `run_queue.py`, `common.py`
(FE counter with primary and JVP = 2 secondary totals, run index, GPU-hour ledger),
`src/dream/losses.py` + tests (KL gradient vs autograd). `pytest`: 12/12 pass.

**Calibration** (`compute_calibration.json`, A40, float32, TF32 off, CUDA-graph GPU time):

| batch | fwd+bwd | JVP (`torch.func.jvp`) | backward only |
|---:|---:|---:|---:|
| 128 | 3.47 | 6.70 | 2.47 |
| 512 | 3.19 | 6.65 | 2.19 |
| 2048 | 3.24 | 6.75 | 2.24 |

A1 costs 4.468 FE per image (student fwd+bwd + teacher fwd), so **`B_full` = 4.47·10⁷ FE** and the
tuning budget is 1.12·10⁷ FE (≈ 51 epochs). Eager timing at batch 128 gave 4.3–5.5 FE for fwd+bwd
(CPU launch-bound), hence Amendment 2.

**Teacher** (trained; the thesis checkpoint is unavailable): brief §3 recipe, 200 epochs, 14 min.
**Validation accuracy 65.72%** (gate ≥ 55%: pass); train-subset accuracy 82.34% (no
augmentation). Stored at `/workspace/runs/teacher_seed0/teacher_state.pt` with cached logits.

**Infrastructure checks.** Kill-and-resume reproduces the step and FE trajectory exactly. Runs are
not bitwise deterministic (cuDNN): two identical uninterrupted smoke runs differ by ≈0.5 pp
validation agreement after 2% of the budget, from the first evaluations on. Concurrency: 61
steps/s for one A1 run, 74 aggregate for two, no further gain at three or four. Queues run two
at a time.

**Sanity check** (A1-1 = Stanton's optimizer recipe, SGD-Nesterov lr 0.05·(1−λ), wd 1e-4, τ = 1;
tuning budget; seed 0; final values, no checkpoint selection):

| λ | val agree | val KL | val acc | train-subset agree | train-subset KL | ‖θ−θ_init‖ |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | **91.2%** | 0.044 | 65.5% | 93.8% | 0.031 | 24.1 |
| 0.25 | **67.9%** | 0.550 | 62.5% | 75.5% | 0.433 | 42.9 |

Gate (λ = 0.5 at least 5 pp above λ = 0.25): **pass, +23.3 pp.** Agreement trajectories at 0 /
25 / 50 / 75 / 100% of the budget: λ = 0.5: 2.8 / 76.2 / 85.4 / 90.1 / 91.2%; λ = 0.25: 0.7 /
52.6 / 62.9 / 67.0 / 67.9%. The λ = 0.25 student is still slowly improving at the end, and its
train-subset agreement (75.5%) is close to Stanton's 78.95% (λ = 0, 300 epochs). Both students
start with near-zero agreement, so the λ = 0.5 advantage is not simply a head start in function
space.

**Budget.** Ledger 0.53 GPU-h of 40 (teacher 0.23, smoke/resume/concurrency tests 0.17, sanity
0.25). Revised projection ≈24 GPU-h (`PLAN.md` §5).

**Next (Phase 2, awaiting go-ahead).** GGN products (jvp+vjp, linearize, reverse-over-reverse;
pick the cheapest exact one and add it to the calibration), damped CG, LM damping, target path,
t and temperature schedules, arms A2–A5, required tests, smoke runs of every arm.

## Phase 2: methods and tests (2026-09-30)

**Amendment 3 recorded before any tuning run** (teacher retrained with snapshots; A1-7 warmup;
A5-6 Annealing-KD; `‖θ − θ_T‖₂` logged; hand-off diagnostic). Annealing-KD details were taken from
the paper (Jafari et al. 2021, CIFAR: τ_max = 10, integer T stages over 160 of 320 epochs, MSE on
Φ(T)-scaled logits); deviations (no hard-label stage II, no best-checkpoint selection, A1's
optimizer) are listed in `PREREGISTRATION.md` §6.

**Teacher v2** (same recipe and seed, 41 snapshots at epochs 0, 5, …, 200 in
`/workspace/runs/teacher_seed0/snapshots/`): validation accuracy **65.78%** (v1: 65.72%),
train-subset accuracy 84.86% (v1: 82.34%). v1 and its two runs are archived
(`teacher_seed0_v1/`, `teacher_v1_runs/`).

**Sanity check repeated against teacher v2** (A1-1, tuning budget, seed 0):

| λ | val agree | val KL | val acc | train-subset agree | ‖θ−θ_init‖ | ‖θ−θ_T‖ start → end |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | **91.7%** | 0.038 | 65.8% | 94.4% | 24.7 | 22.7 → **12.1** |
| 0.25 | **66.2%** | 0.600 | 61.1% | 74.3% | 43.9 | 34.0 → **48.0** |

Gate: **pass, +25.4 pp** (v1: +23.3 pp). The new distance-to-teacher metric shows the basin
picture directly: at λ = 0.5 the student moves towards the teacher's weights, at λ = 0.25 it moves
away from them while its function approaches the teacher's.

**Implemented.** `src/dream/ggn.py` (GGN products by `jvp`+`vjp`, `linearize`,
reverse-over-reverse; damped CG with warm start; LM damping; reduction ratio),
`src/dream/schedules.py` (fixed and adaptive t with the relative-progress rule and 60–70% deadline
ramp; geometric and Annealing-KD temperature; warmup+cosine lr), `distill.py` arms A1–A5 with
`--init-from` for the hand-off, `make_configs.py` (A2/A5 inherit the selected A1 config, A4 the
selected A3 config), `calibrate_ggn.py`. A1-7 and A3-1…6 configs written.

**Tests** (`pytest`: 28 pass): GGN product = explicit `Jᵀ F J v` for all three methods (float64);
CG to relative residual < 1e-8; warm start; damped GN with λ → 0 equals
`live_refined_minimum_norm` for all three methods; KL gradient = autograd; target-path,
temperature-path and Annealing-KD endpoints; adaptive-t transitions (threshold, doubling,
relative advance, halving, deadline ramp); LM directions; ρ = 1 on a quadratic. Two of my own
tests were wrong at first (a step so long that the predicted change was positive; a warm-start
assertion CG does not guarantee) and were corrected; no code change was needed.

**GGN method** (Amendment 1): `linearize` chosen, lowest measured GPU cost (table in
`compute_calibration.json`):

| per image, FE | `jvp`+`vjp` | `linearize` | reverse-over-reverse |
|---|---:|---:|---:|
| setup, \|C\| = 512 | 1.0 | 8.6 | 3.9 |
| product, \|C\| = 128 | 10.1 | 7.6 | 16.3 |
| product, \|C\| = 512 | 9.6 | 7.4 | 41.1 |

Caveat: `linearize` re-traces the model on the CPU at every step (≈1.5 s). Alone, an A3-2 step
takes 1.9 s (`jvp`+`vjp`: 0.39 s). Eight concurrent Gauss–Newton runs saturate the GPU at the same
≈3.3 steps/s for both methods, so the CPU cost is absorbed by running GN arms 8 at a time
(≈16 min GPU-h per full run, ≈2 h wall each). Measured under saturation, `linearize` realizes
≈80% of its calibrated FE throughput, `jvp`+`vjp` ≈95%, first-order A1 ≈70%; all arms are charged
by the same graphed calibration.

**Smoke tests** (1% of `B_full`, λ = 0.25, seed 0; all completed, no NaNs):

| arm/config | steps | val agree start → end | notes |
|---|---:|---|---|
| A1-7 (warmup) | 782 | 1.1 → 14.9% | |
| A2-1 (fixed φ=0.25) | 638 | 1.1 → 13.5% | extra f_init forward charged |
| A2-5 (adaptive τ=0.15) | 352 | 1.1 → 1.5% | 5 threshold advances to t = 0.16, then deadline ramp |
| A3-2 | 30 | 1.1 → 5.6% | 93% of steps accepted; damping 1 → 7.6 |
| A3-4 (B 2048) | 7 | 1.1 → 3.1% | stops at 88% (next step would exceed budget) |
| A4-2 (fixed) | 29 | 1.1 → 7.0% | |
| A4-5 (adaptive) | 27 | 1.1 → 4.4% | 59% accepted; deadline ramp |
| A5-1 (geometric) | 782 | 1.1 → 15.4% | |
| A5-6 (Annealing-KD) | 782 | 1.1 → 6.3% | |

GN resume (kill and restart) works; the hand-off (`--init-from`) works. Numbers at 1% budget are
plumbing checks, not comparisons.

**Points for Fred before Phase 3.**
1. *Hand-off design.* As preregistered, the hand-off restarts A1's schedule at its peak lr
   (0.0375 at λ = 0.25). In the smoke test this knocked the student off its starting point
   (7.0% → 1–2% agreement early on). The A1 control gets the same treatment, so the comparison is
   symmetric, but it measures robustness to a large lr kick more than whether the endpoint sits
   in a better basin. Alternative: start the hand-off with A1-7's 10% warmup, or with a lower
   peak lr. Your call; it is a diagnostic, and changing it now is before any result.
2. *Adaptive-t monitor cost.* 200 checks × 2,000 forwards = 0.4M FE, i.e. 3.6% of a tuning budget
   and 0.9% of `B_full`, charged to A2/A4 adaptive configs as preregistered.
3. *A1 has 7 configs* (brief limit 6), by Amendment 3.

**Budget.** Ledger 0.96 GPU-h of 40 (Phase 2: teacher v2 0.22, calibration and smoke tests,
sanity reruns). Revised projection for Phases 3–4 including the hand-off: ≈16 GPU-h plus 25%
contingency, total ≈21 GPU-h.

## Phase 3: tuning and λ sweep (validation only), 2026-09-30 to 2026-10-01

All numbers are on validation or on training images; the test split has not been opened. One
seed (0) per configuration; identical reruns differ by about 0.5 pp of validation agreement
(cuDNN nondeterminism), so single-seed gaps below about 1 pp are noise. "Train agreement" and
"train KL" are on all 50,000 distillation images without augmentation (Amendment 5);
`KL(p_T ‖ p_S)` at temperature 1 replaces training losses, which differ across arms.

### Headline

1. **Soft Dream (A4) and Gauss–Newton (A3) lose clearly.** At full budget they reach 51% (λ = 0)
   to 67% (λ = 0.5) validation agreement, 20–30 pp behind every first-order arm, and their
   weights drift far from the teacher (‖θ − θ_T‖ ≈ 160–560, versus 5–70 for first-order arms).
   Per FE budget they take only a few hundred very expensive steps.
2. **At λ = 0 no intervention helps at 1× compute**: all first-order arms reach 71–73% validation
   agreement. The adaptive and fixed continuation paths (A2, A6) are within noise of A1.
3. **At λ = 0.25, Stanton's failure case mostly disappears under our recipe**: A1-7 (Stanton's
   optimizer plus 10% lr warmup) reaches 89.8% validation and 93.7% train agreement at 1×,
   and **97.8% / 98.8% at 4×** (patience check). The barrier is a compute and optimizer-schedule
   effect in our setup, not a hard basin barrier.
4. At λ = 0.25 and 1×, the moving target (A2, 93.3%), Annealing-KD (A5, 93.4%) and plain logit
   MSE without annealing (A1-MSE, 94.3%) all beat A1-7 (89.8%). **A1-MSE ≥ A5-6** says A5's gain
   is a loss effect, not an annealing effect. **A weak anchor to the initialization beats all of
   them** (μ = 0.01: 96.0%).
5. Patience check at λ = 0 (Fred's pre-recorded criterion): **not met** (best train agreement
   at 4×: A5-6 89.5%, below 95%). But A5-6 at 4× reaches 85.0% *validation* agreement at λ = 0,
   12 pp above A1-7 and A2-3 at 4× (73.2%, 73.5%), the only large λ = 0 effect seen so far
   (one seed).

### Tuning (λ = 0.25, seed 0, `B_full/4`), selected by validation agreement

| arm | selected | val agree | runner-up | notes |
|---|---|---:|---|---|
| A1 | **A1-7** (A1-1 + 10% warmup) | 72.88% | A1-4 (τ = 4) 68.54% | Stanton's recipe A1-1: 66.22%; Adam worst (61.7–64.4%) |
| A2 | **A2-3** (fixed t, φ = 0.75) | 83.82% | A2-5 (adaptive, τ_KL = 0.15) 82.86% | all 6 configs ≥ 74.9% |
| A3 | **A3-6** (\|B\| = \|C\| = 512, K = 10) | 33.52% | A3-1 32.08% | 100–230 GN steps per tuning budget |
| A4 | **A4-6** (adaptive, τ_KL = 0.4) | 39.76% | A4-5 38.76% | |
| A5 | **A5-6** (Annealing-KD) | 87.20% | A5-3 (T0 = 4, φ = 0.75) 85.86% | |
| A6 | **A6-3** (spacing 5, τ_KL = 0.4) | 72.00% | A6-4 71.62% | |
| A1-MSE | **A1MSE-1** (lr 0.02) | 72.34% | A1MSE-2 71.74% | diagnostic (Amendment 5) |

Full per-config tables (with train-subset agreement, distances and schedule summaries) are
produced by `python analyze.py tuning`. Note on A3-1: its weights drifted to ‖θ − θ_T‖ ≈ 450
with agreement similar to the other A3 configs. A plausible mechanism: GroupNorm makes the
preceding weights scale-invariant, so low-curvature directions let the norm grow without
changing the function.

### λ sweep (`B_full`, seed 0)

Validation agreement:

| arm | λ = 0 | 0.1 | 0.25 | 0.4 | 0.5 |
|---|---:|---:|---:|---:|---:|
| A1-7 | 71.20% | 70.70% | 89.84% | 96.90% | 97.18% |
| A2-3 | 72.00% | 71.14% | 93.28% | 96.68% | 96.56% |
| A3-6 | 51.24% | 51.74% | 59.48% | 65.02% | 66.92% |
| A4-6 | 51.18% | 50.94% | 55.88% | 66.48% | 67.36% |
| A5-6 | 72.88% | 73.26% | 93.36% | 98.66% | 95.48% |
| A6-3 | 72.30% | 71.90% | 84.22% | 97.14% | 97.12% |
| A1-MSE (diag.) | 71.46% | 72.72% | 94.28% | 98.86% | 99.02% |

Validation KL(p_T ‖ p_S):

| arm | λ = 0 | 0.1 | 0.25 | 0.4 | 0.5 |
|---|---:|---:|---:|---:|---:|
| A1-7 | 0.471 | 0.477 | 0.060 | 0.004 | 0.003 |
| A2-3 | 0.462 | 0.460 | 0.021 | 0.005 | 0.006 |
| A3-6 | 1.076 | 1.076 | 0.792 | 0.634 | 0.562 |
| A4-6 | 1.094 | 1.081 | 0.923 | 0.569 | 0.557 |
| A5-6 | 0.399 | 0.403 | 0.023 | 0.001 | 0.010 |
| A6-3 | 0.457 | 0.444 | 0.136 | 0.003 | 0.003 |
| A1-MSE | 0.448 | 0.424 | 0.017 | 0.000 | 0.000 |

Train agreement / train KL (50k, no augmentation):

| arm | λ = 0 | 0.1 | 0.25 | 0.4 | 0.5 |
|---|---:|---:|---:|---:|---:|
| A1-7 | 84.62% / 0.253 | 84.62% / 0.257 | 93.67% / 0.037 | 98.44% / 0.002 | 98.57% / 0.002 |
| A2-3 | 83.25% / 0.272 | 83.19% / 0.269 | 96.04% / 0.014 | 98.05% / 0.004 | 97.93% / 0.004 |
| A3-6 | 52.32% / 1.137 | 53.54% / 1.097 | 62.25% / 0.805 | 68.47% / 0.606 | 70.71% / 0.541 |
| A4-6 | 52.33% / 1.124 | 52.31% / 1.138 | 57.37% / 0.955 | 69.53% / 0.561 | 69.85% / 0.552 |
| A5-6 | 77.37% / 0.338 | 78.22% / 0.324 | 95.55% / 0.017 | 99.27% / 0.000 | 97.02% / 0.007 |
| A6-3 | 84.07% / 0.260 | 84.12% / 0.255 | 90.78% / 0.083 | 98.55% / 0.002 | 98.60% / 0.002 |
| A1-MSE | 76.73% / 0.362 | 77.34% / 0.346 | 96.26% / 0.012 | 99.37% / 0.000 | 99.40% / 0.000 |

Final distance to the teacher's weights ‖θ − θ_T‖₂:

| arm | λ = 0 | 0.1 | 0.25 | 0.4 | 0.5 |
|---|---:|---:|---:|---:|---:|
| A1-7 | 62.0 | 55.4 | 19.8 | 4.7 | 4.5 |
| A2-3 | 56.8 | 48.7 | 11.8 | 9.6 | 11.0 |
| A3-6 | 309.5 | 429.0 | 466.6 | 555.3 | 513.6 |
| A4-6 | 268.0 | 156.8 | 368.1 | 487.0 | 479.8 |
| A5-6 | 72.3 | 64.3 | 26.4 | 14.7 | 14.3 |
| A6-3 | 60.1 | 53.1 | 27.7 | 4.6 | 4.4 |
| A1-MSE | 68.9 | 61.5 | 24.4 | 8.9 | 7.2 |

Figures (`experiments/stanton/figures/`):
- `sweep_teacher_distance.png`: ‖θ − θ_T‖ against compute for every λ > 0 run of every arm
  (y-axes differ per arm). First-order arms at λ ≥ 0.25 move *towards* the teacher; at
  λ = 0.1 they move away (40 → 55–65). The GN arms move away at every λ.
- `tuning_teacher_distance.png`: the same for all tuning runs (λ = 0.25).
- `sweep_agreement.png`, `sweep_kl.png`: validation agreement and KL against compute, one panel
  per λ. The continuation arm A2 starts slowly (its early targets are close to its own outputs)
  and catches up by ≈60% of the budget.
- `sweep_agreement_vs_lambda.png`: final validation agreement against λ.
- `sweep_train_agreement_vs_lambda.png`: **train agreement against λ per arm, the view of Stanton
  et al. Figure 6(b).**

**Comparison with Stanton et al. Figure 6(b).** Stanton report near-perfect train agreement for
λ ≥ 0.375 and a sharp drop at λ ≤ 0.25 (SGD, 300 epochs; train agreement 78.95% at λ = 0).
In our setup the transition sits between **λ = 0.1 and 0.25**, not between 0.25 and 0.375: at
λ = 0.25 every first-order arm reaches 90.8–96.3% train agreement at 1×. Differences that matter:
a 200-epoch-equivalent budget (theirs: 300 epochs); our λ grid {0, 0.1, 0.25, 0.4, 0.5} (theirs
includes 0.375); GroupNorm(1, C) instead of their LayerNorm ResNet-20; a weaker teacher (65.8%
vs their ≈70.5%); τ = 1 KL instead of their τ = 4 default; and, for A1-7, a 10% lr warmup.
Amendment 8 (A1-1, Stanton's exact recipe, at full budget for all λ; running now) will separate
the warmup effect from the budget effect; at the tuning budget, warmup alone added 6.7 pp at
λ = 0.25 (A1-1 66.2% → A1-7 72.9%). At λ = 0 our A1-7 train agreement (84.6%) is above Stanton's
78.95%.

### Diagnostics

**Anchor (Amendment 4), λ = 0.25, seed 0.** A1-7 plus `(μ/2)‖θ − θ_init‖²`, decaying to 0 by 30%:

| | val agree | val KL | train agree | ‖θ − θ_T‖ | ‖θ − θ_init‖ |
|---|---:|---:|---:|---:|---:|
| A1-7 (μ = 0) | 89.84% | 0.060 | 93.67% | 19.8 | 38.0 |
| μ = 0.01 | **95.98%** | 0.007 | 97.83% | **8.0** | 33.7 |
| μ = 0.1 | 91.60% | 0.038 | 94.75% | 15.4 | 36.5 |

A weak early pull towards θ_init (which at λ = 0.25 is a quarter of the way to θ_T) is the
strongest intervention at λ = 0.25 and ends closest to the teacher's weights. The λ = 0 anchor
run (Amendment 10) tests whether this needs the teacher component; it is running.

**A1-MSE (Amendment 5).** A1-7 with A5-6's logit MSE and no annealing matches or beats A5-6 at
every λ ≥ 0.25 (94.3 vs 93.4, 98.9 vs 98.7, 99.0 vs 95.5%), and is equal at λ ≤ 0.1. At λ ≤ 0.1
both MSE arms have *lower* train agreement than KL arms (77% vs 83–85%) with equal validation
agreement. So in A5-6 the loss matters and the annealing does not, at 1×.

**Hand-off (Amendment 4), seed 0, `B_full/4` with A1-7.** Validation agreement before → after:

| source | λ | before | primary (warmup to 0.1·peak) | secondary (warmup to peak) |
|---|---:|---:|---:|---:|
| A1-7 (control) | 0.25 | 89.84% | 90.14% | 90.62% |
| A2-3 | 0.25 | 93.28% | **94.00%** | 87.42% |
| A6-3 | 0.25 | 84.22% | 84.86% | 85.30% |
| A4-6 | 0.25 | 55.88% | 64.36% | 68.08% |
| A1-7 (control) | 0 | 71.20% | 71.36% | 71.92% |
| A2-3 | 0 | 72.00% | 71.64% | 71.52% |
| A6-3 | 0 | 72.30% | 72.30% | 72.12% |
| A4-6 | 0 | 51.18% | 60.74% | 66.26% |

At λ = 0.25 the A2 endpoint keeps its lead over the A1 control under the gentle hand-off (94.0
vs 90.1%) but loses it under the full-lr one (87.4%): it sits in a better region that a large lr
kick leaves. At λ = 0 nothing changes. A4 endpoints improve under first-order training but stay
far behind.

**Patience check (Amendments 6 and 7), `4·B_full`, seed 0** (`figures/patience_4x.png`; 1× =
the λ-sweep run, 4× = end of the 4× run):

| λ | run | val agree 1× → 4× | train agree 1× → 4× | train KL 1× → 4× |
|---:|---|---|---|---|
| 0.25 | A1-7 | 89.84 → **97.76%** | 93.67 → **98.79%** | 0.037 → 0.002 |
| 0.25 | A5-6 | 93.36 → 91.42% | 95.55 → 94.29% | 0.017 → 0.028 |
| 0 | A1-7 | 71.20 → 73.22% | 84.62 → 87.28% | 0.253 → 0.200 |
| 0 | A2-3 | 72.00 → 73.46% | 83.25 → 86.90% | 0.272 → 0.197 |
| 0 | A5-6 | 72.88 → **85.02%** | 77.37 → 89.54% | 0.338 → 0.091 |

- λ = 0.25 (Amendment 6): **A5-6 at 1× does not beat A1-7 at 4×** (93.4 vs 97.8%), and the two do
  not converge: A1-7 keeps improving to near-perfect agreement, A5-6 plateaus slightly *below*
  its 1× value. Caveat: a 4× run stretches its schedules, so "A5-6 at 4×" is a different
  schedule, not merely more steps.
- λ = 0 (Amendment 7, Fred's criterion recorded in advance): **criterion not met.** No arm reaches
  95% train agreement (best: A5-6 89.5%), and A1-7 is 2.3 pp below A5-6, not 5. Unexpected and
  noted for Fred: A5-6 at 4× reaches **85.0% validation agreement** at λ = 0, 12 pp above A1-7
  and A2-3 at 4× and above its own train-agreement gain; its validation curve separates from the
  others after ≈2× compute. One seed; not part of any decision.

### What this implies for Phase 4 (no test data used)

On validation, the GO outcome cannot occur (A4 trails A1 by 15–34 pp), and by Amendment 9 the
outcomes that need A3 or A4 will be recorded as not met if the test numbers confirm the gap. The
decisive comparisons are A1 versus A2, A5 and A6 at λ = 0 (all within about 1.7 pp at seed 0)
and at λ = 0.25 (A2 and A5 +3.5 pp over A1). The final configurations are the preregistered
selections: A1-7, A2-3, A5-6, A6-3 (decision rule), A1MSE-1 and A1-anchor-mu0.01 (diagnostics),
seed-0 only for A3-6 and A4-6.

### Runs resumed after a crash or kill

Causes: GPU out-of-memory crashes when several queues shared the GPU (my scheduling error; fixed
by a memory-aware queue, a check on every launch, and per-run locks), and two kills of queue
sessions by me. No run was discarded.

| run | resumed from checkpoint at step | fresh restarts | finished |
|---|---|---:|---|
| A1-7 λ = 0.1, 0.25, 0.4, 0.5 (full) | 7821 | 0 | yes |
| A1-7 λ = 0 (full) | 7821, 7821 | 0 | yes |
| A2-3 λ = 0 (full) | 6390 | 0 | yes |
| A2-3 λ = 0.25, 0.4 (full) | 1278 | 0 | yes |
| A2-3 λ = 0.1, 0.5 (full) | – | 1 | yes |
| A3-6 λ = 0.1 (full) | 92 | 0 | yes |
| A3-6 λ = 0.25, 0.5 (full) | – | 1 | yes |
| A4-3 (tune) | 5, 5, 5 | 0 | yes |
| A4-4 (tune) | – | 1 | yes |
| A4-5 (tune) | 92 | 0 | yes |
| A4-6 λ = 0.1 (full) | 91 | 0 | yes |
| A6-3 λ = 0 (full) | – | 1 | yes |

Restored exactly on every resume: model or parameter vector, optimizer state (momentum / Adam
moments) or, for GN arms, the previous CG step and LM damping; the data-sampler CUDA generator
state with epoch order and cursor; the FE counter, and hence the lr schedule, which is a function
of the FE fraction; the t / temperature schedule state. Not restored: work after the last
checkpoint is recomputed, and because cuDNN kernels are not bitwise deterministic, the recomputed
steps are not bit-identical to the lost ones. A "fresh restart" (crash before the first
checkpoint) reruns from step 0 with the same seeds.

### Budget

GPU busy-clock (the figure checked against the cap): **19.0 GPU-h of 40** at the end of Phase 3
runs (it includes Phase 4 runs that have been running since 07:12). The per-run ledger in
`budget.json` sums to more (≈65) because it divides each run's wall-clock by its own queue's
concurrency while several queues shared the GPU; it is an attribution, not the budget figure
(noted in the file). Projection to the end of Phase 4: ≈29 GPU-h.

### Status and next step

Phase 4 training (Amendments 8–10) is running: A1-1 at five λ, the λ = 0 seed-0 anchor run, and
seeds 1 and 2 at λ ∈ {0, 0.25} for A1-7, A2-3, A5-6, A6-3, A1MSE-1 and A1-anchor-mu0.01. Expected
to finish around 18:00–20:00. **The test split stays closed until Fred approves this report.**
