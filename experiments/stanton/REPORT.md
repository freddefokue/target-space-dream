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
