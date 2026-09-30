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
