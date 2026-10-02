# Plan: soft Dream on Stanton's CIFAR-100 fidelity benchmark

Status: Phase 0, 2026-09-30. The brief is `CLAUDE.md` at the repository root. Decisions taken with
Fred are in §6.

## 1. Environment findings (Phase 0)

| Item | Brief assumes | Found | Consequence |
|---|---|---|---|
| GPU | RTX 4090, 24 GB | **NVIDIA A40, 46 GB**, driver 580.159, CUDA 13.0 | All FE calibration and wall-clock numbers are A40 numbers. Memory is not a constraint. |
| Software | — | Python 3.12.3, torch 2.8.0+cu128, torchvision 0.23.0 | `pip install -e '.[test,vision]'` done; 10/10 tests pass. |
| Repo location | on `/workspace` | **`/target-space-dream` (container disk, not persistent)** | See §1.1. |
| `/workspace` | persistent storage | geesefs (S3-backed FUSE). **git cannot run there** (`chmod ... Operation not permitted`). Appends and atomic rename work (~35 ms per fsynced append). | Runs, data and ledgers live on `/workspace`; the git repo cannot. |
| CIFAR-100 | download | `/workspace/data/cifar-100-python`, tarball md5 `eb9058c3a382ffc7106e4002c42a8d85` (official) | Done. |
| Teacher checkpoint | `/workspace/inputs/teacher_state.pt` | **absent**, and Fred cannot upload it | Train one in Phase 1 (brief §3 recipe). |
| Python packages | — | installed into system site-packages (not persistent) | After a pod restart: `pip install -e '.[test,vision]'` again. |

### 1.1 Keeping the repo safe

After every commit: `git bundle create /workspace/backup/target-space-dream.bundle --all`.
Restore after a pod restart with
`git clone -b soft-dream-stanton /workspace/backup/target-space-dream.bundle /target-space-dream`.
Nothing is pushed to GitHub unless Fred asks.

### 1.2 Throughput probe (not a calibration; Phase 1 does that properly)

`ResNet20GN1` (278,132 parameters), float32, TF32 off, A40:

| Batch | forward, no grad | fwd+bwd / fwd | `jvp` / fwd | `jvp`, then fresh `vjp` (GGN product) / fwd |
|---:|---:|---:|---:|---:|
| 128 | 2.2 ms (≈58k img/s) | 3.4× | 6.8× | 16.2× |
| 512 | 8.2 ms (≈62k img/s) | 3.2× | 6.7× | 10.7× |
| 1024 | 15.8 ms (≈65k img/s) | 3.2× | 6.7× | 10.7× |

Forward-mode `torch.func.jvp` is expensive for this conv net (≈6.7 FE). Phase 2 will compare
`jvp`+`vjp`, `linearize`, and the double-VJP trick (`J v` as the VJP of a VJP) and use the
cheapest exact one. **All arms are charged measured FE**, so an expensive GGN product costs the
Gauss–Newton arms steps; they will not be made to look cheaper than they are.

## 2. Stanton recipe and deviations

| Aspect | Stanton / gnosis | This study | Why |
|---|---|---|---|
| Model | Pre-act ResNet-20 with LayerNorm (implementation unreleased; gnosis ships BatchNorm) | `ResNet20GN1`: pre-act ResNet-20, GroupNorm(1, C) | Thesis model; GN(1,C) is LayerNorm over (C,H,W), no running statistics, as Stanton required. |
| Input scaling | torchvision `ToTensor` ("unitcube", [0,1]) | `x/127.5 − 1` | Thesis convention (brief §3). |
| Augmentation | random crop 32, pad 4 + h-flip | same, on GPU | — |
| Teacher | 200 epochs, SGD m0.9, lr 0.1 cosine→0, wd 1e-4, batch 256 | 200 epochs, Nesterov m0.9, lr 0.1 cosine, wd 5e-4, batch 128, seed 0 (brief recipe), trained here | Thesis teacher unavailable. Stanton's §6 teacher: 70.5% test; thesis teacher: 63.74%. |
| Distillation loss | `τ²·CE(softmax(z_t/τ), softmax(z_s/τ))`; gnosis default τ = 4; §6 text does not state τ | KL at τ = 1 for A1–A4, plus **one A1 config with Stanton's τ = 4 loss**; A5 anneals τ to 1 | Brief §6; τ = 4 added to A1 for fairness (decision D4). Fidelity is always measured at τ = 1. |
| Student optimizer | SGD Nesterov m0.9, lr 0.05 cosine→1e-6, wd 1e-4, batch 128, 300 epochs | tuned grid containing this recipe (A1 config 1), at 200-epoch-equivalent compute | Brief fixes `B_full` = 200 epochs of A1. |
| Interpolated init | `λ·θ_t + (1−λ)·θ_r` (gnosis code: `r·θ_r + (1−r)·θ_t`, `r = 1−λ`), **lr scaled by (1−λ)** | Same interpolation and the same (1−λ) lr scaling for A1, A2, A5 (Amendment 1) | Stanton's λ threshold was measured with it. |
| λ grid | {0, 0.25, 0.375, …}; transition between 0.25 and 0.375 | {0, 0.1, 0.25, 0.4, 0.5}; λ = 0.5 is the positive control (Amendment 1) | 0.4 lies only just above Stanton's transition. |
| Reported §6 numbers | train agreement 78.95% (SGD, 300 epochs), 83.3% (5k epochs) | our A1 train-subset agreement is the comparable number | Sanity reference, not a target. |

Additional design choice: **θ_R uses seed `1000 + s`**, never the teacher's training seed 0, so
that λ = 0 at seed 0 is not accidentally the teacher's own initialization.

## 3. File layout

```text
src/dream/ggn.py              GGN-vector products (softmax-KL output metric), CG with warm start,
                              Levenberg–Marquardt damping controller
src/dream/schedules.py        fixed-φ and adaptive t schedules; geometric temperature schedule
src/dream/losses.py           KL(q ‖ softmax(z)) and its logit gradient p − q
tests/unit/test_ggn.py        GGN vs explicit Jᵀ F J v; CG to 1e-8; λ→0 limit vs live_refined_minimum_norm
tests/unit/test_schedules.py  path/temperature endpoints, adaptive controller transitions
tests/unit/test_losses.py     KL gradient vs autograd
experiments/stanton/
  data.py                     GPU-resident CIFAR-100, crop+flip on GPU, splits (val/test/train-subset/monitor)
  teacher.py                  train (or load) the teacher
  init.py                     θ_init(λ, seed)
  evaluate.py                 agreement, KL, accuracy, displacement on val / train-subset (test gated)
  accounting.py               FE counter, calibration table, run index, budget ledger
  calibrate.py                writes compute_calibration.json
  distill.py                  --arm A1..A5 --config ... --lam ... --seed ... --budget full|tune|<FE>
  run_queue.py                sequential/concurrent run queue for tmux, resumable
  plot.py                     agreement/KL vs FE, final agreement vs λ
  PLAN.md, PREREGISTRATION.md, REPORT.md, compute_calibration.json, RESULTS.md (Phase 4)
configs/stanton/*.json        one file per tuning config and per selected final config
/workspace/runs/<run_id>/     config.json, metrics.jsonl, final.json, ckpt.pt (not in git)
/workspace/runs/index.jsonl, /workspace/runs/budget.json
```

## 4. Implementation plan

**Data.** Load the CIFAR pickle once and keep uint8 tensors on the GPU (50k×3×32×32 = 150 MB).
Augmentation per batch: zero-pad 4 (as torchvision `RandomCrop(padding=4)`), random crop offsets,
random horizontal flip, all drawn from a seeded per-run `torch.Generator` so that a resumed run
replays the same stream. Splits are built with seed 0 and stored as index files in
`/workspace/data/splits/`: val = 50 per class from the test set, test = the rest;
train-subset = 50 per class from the train set (5,000, no augmentation); monitor = 2,000
class-balanced train images disjoint from the train-subset.

**Evaluation.** Teacher logits on val, test and the train-subset are computed once and cached.
Evaluation passes are logged separately and **not** charged to the optimization budget
(identical for all arms). The adaptive schedule's monitor evaluations **are** charged, since they
steer the optimizer. The test split is loaded by a separate code path that refuses to run without
an explicit `--open-test` flag (Phase 4 only).

**FE accounting.** Calibrate cost ratios (fwd, fwd+bwd, JVP, GGN product, no-grad fwd) per batch
size used, median of repeated timings, float32, TF32 off. Each arm increments an FE counter by
`ratio × images` for every pass. Teacher and frozen-init forwards on augmented batches are
charged (they cannot be cached because augmentation is random). A run stops at the first step
that would exceed its budget.

**A1/A2/A5.** One first-order loop with a pluggable target (teacher; moving target with t(FE);
tempered teacher with T(FE)). The lr schedule is cosine over the run's FE budget.

**A3/A4.** Per step: forward on B (student, teacher, and f_init for A4); gradient; K CG iterations
with GGN on C ⊆ B, warm start `β·δ_prev`; LM trial evaluation at θ+δ on B (one forward);
accept/reject, update λ_d. No learning rate: the step is δ. Numerical guards: NaN → reject and
×10 damping; CG stops early on non-positive curvature (should not happen for GGN + λI).

**Resumability.** Checkpoint (params, optimizer/CG/LM state, schedule state, RNG states, FE
counter, step) every ~5% of budget, atomically (save to tmp, `os.replace`).

**Concurrency.** At batch 128 a single run is launch-bound; Phase 1 measures per-run throughput
with 1–4 concurrent processes and picks the level before per-run speed collapses.

## 5. Compute estimate

Original Phase 0 estimate: ≈26 GPU-h (0.4 GPU-h per full run). **Revised after Phase 1
calibration (2026-09-30):**

- `B_full` = 4.468 FE/image × 200 × 50,000 = **4.47·10⁷ FE** (CUDA-graph ratios, Amendment 2).
- A1 at batch 128 runs at 61 steps/s alone, 74 steps/s aggregate with 2 concurrent runs (3 or 4
  add nothing). A full first-order run is 78,200 steps: ≈0.30 GPU-h at concurrency 2.
- Gauss–Newton arms: unknown until Phase 2; assumed 0.35 GPU-h per full run.

| Item | Runs | GPU-h |
|---|---:|---:|
| Teacher training (actual) | 1 | 0.23 |
| Phase 1 smoke, resume, concurrency tests (actual) | — | 0.17 |
| Phase 1 sanity (λ 0.5, 0.25 at tuning budget) | 2 | 0.15 |
| Phase 2 smoke tests, GGN-variant timing, debugging | — | 1.5 |
| Phase 3 tuning: 5 arms × 6 configs at `B_full/4` | 30 | 2.5 |
| Phase 3 λ sweep: 5 arms × 5 λ, seed 0 | 25 | 8.0 |
| Phase 4 finals: 5 arms × 2 λ × seeds 1, 2 | 20 | 6.5 |
| **Subtotal** | | **19.1** |
| Contingency (+25%) | | 4.8 |
| **Projected total** | | **≈ 24 of 40** |

GPU-hours mean pod wall-clock while the GPU is occupied by this project; a run's ledger charge is
its wall-clock divided by the number of runs sharing the GPU.

Reproducibility note: cuDNN kernels are not bitwise deterministic here, so two identical runs
differ slightly (0.5 pp validation agreement after 2% of the budget in a smoke test). Resumed runs
reproduce the step and FE trajectory exactly. Seed replicates capture this noise.

## 6. Decisions (Fred, 2026-09-30)

- D1. GPU-hour cap: 30, may be changed later. **Superseded: 40 (Amendment 1).**
- D2. The thesis teacher cannot be uploaded; train a new one with the brief's recipe.
- D3. Back the repo up as a git bundle in `/workspace/backup/` after every commit.
- D4. A1's grid includes one config with Stanton's τ = 4 distillation loss.
- D5. ~~No (1−λ) lr scaling~~ **Reversed by Fred: (1−λ) lr scaling for A1, A2, A5 (Amendment 1).**
- D6. Adaptive-schedule thresholds fixed now in the preregistration (left to me).
- D7 (Amendment 1). λ = 0.5 added; it replaces 0.4 as the positive control.
- D8 (Amendment 1). Adaptive t schedule gains a relative-progress advance (monitor KL halved since
  the last advance) and a deadline ramp to t = 1 over 60–70% of the budget; forced advances logged.
- D9 (Amendment 1). Before tuning, pick the cheapest exact GGN product among jvp+vjp, linearize and
  reverse-over-reverse; also report a secondary compute count with JVP = 2 FE.
- D10. Commit at least hourly, each commit followed by the backup bundle.
- D11 (Amendment 3). Teacher retrained with snapshots every 5 epochs; the first teacher is
  archived as `teacher_seed0_v1`; Phase 1 sanity runs repeated.
- D12 (Amendment 3). A1-7 (warmup) added; A5-6 is an Annealing-KD replication; `‖θ − θ_T‖₂`
  logged; hand-off diagnostic after the λ sweep (≈0.5 GPU-h).
- D13 (Amendment 5). 50k clean-train agreement and KL for selected tuning configs, sweep and final
  seeds (`train_eval.py`), reported instead of train loss; A1-MSE diagnostic arm; list of resumed
  runs with restoration status.
- D14 (2026-10-02). GPU-hour cap raised to 50 for the separate follow-up (`FOLLOWUP_PREREG.md`).
