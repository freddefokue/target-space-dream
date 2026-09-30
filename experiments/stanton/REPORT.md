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
