# Follow-up preregistration: annealing versus fixed temperature at λ = 0, 4× compute

Written and committed on 2026-10-02 before any run of this follow-up started. This is a new,
separate study. The original study (`PREREGISTRATION.md`, `RESULTS.md`) and its decision rule
are closed and are not revisited.

## Question

At λ = 0 with `4·B_full` (schedules stretched with the budget, as in Amendment 7), does annealing
(A5-6, Annealing-KD) beat the best fixed temperature, or do soft targets plus patience suffice?

## What is already known (disclosure)

This follow-up is motivated by post-hoc results (`RESULTS.md`, Amendment 11 addendum), all of
which were seen before this document was written:

- A5-6 at λ = 0, `4·B_full`, seeds 0, 1, 2: validation 85.02 / 86.10 / 84.22%, **test 85.44 /
  85.48 / 82.78%** (post-hoc test evaluation of 2026-10-02).
- A1-7 with fixed τ = 4 at λ = 0, `4·B_full`, seed 0: validation 80.24%, **test 79.78%**.
- A1-7 (τ = 1) at λ = 0, `4·B_full`, seeds 0 and 1: validation 73.22 / 72.76%, test 72.12 /
  72.12%. A2-3 seed 0 and A1MSE-1 seed 0 at 4× were also evaluated.

**Consequence.** The decision rule below compares A5-6's 3-seed test mean with the best
fixed-temperature arm's. A5-6's three test values are already known, and so is the τ = 4 seed-0
test value. The rule is therefore only partly blind: the new information is the test agreement
of the selected fixed-τ arm's seeds 1 and 2 (and of τ = 2 or 8 if one of them is selected). The
final test evaluation re-evaluates the existing models with the same code and data, which
reproduces their known values. This is recorded here so the result is read accordingly.

## Setup (unchanged from the original study unless stated)

ResNet20GN1 teacher (v2, 65.78% validation / 65.80% test accuracy) and student; CIFAR-100 with
the original splits; λ = 0 (student initialized from `θ_R(1000 + seed)`); FE accounting with the
original calibration; `4·B_full` per run; float32, TF32 off. Metrics logged about every 2% of the
budget: validation agreement and KL, and agreement and `KL(p_T ‖ p_S)` on all 50k distillation
images without augmentation (`--full-train-eval`). Agreement and KL are always measured at
temperature 1.

## Runs (λ = 0, `4·B_full`)

1. **A1-7 (τ = 1)**, seed 2 (seeds 0 and 1 exist). Baseline spread; not part of the rule.
2. **Fixed temperature**: A1-7 with loss `τ²·KL(softmax(f_T/τ) ‖ softmax(f_S/τ))` (Hinton
   scaling), no annealing; configs `A1-7-tau2`, `A1-7-tau4` (existing run), `A1-7-tau8`.
   τ ∈ {2, 8} at seed 0.
   **Selection of τ (validation only):** among τ ∈ {2, 4, 8}, the highest seed-0 validation
   agreement at the end of `4·B_full`; ties within 0.1 pp go to the lower validation KL. Then
   the selected τ runs seeds 1 and 2. If τ = 4 is selected, its existing seed-0 run counts as
   seed 0.
3. **A5-short**: A5-6 with its ten annealing stages (T = 10 … 1, MSE on `Φ(T)·f_T`) compressed
   into the first `0.5·B_full` of the `4·B_full` run (schedule fraction φ = 0.125 instead of 0.5),
   then A5-6's own final phase (T = 1, i.e. MSE on the unscaled teacher logits) for the remaining
   `3.5·B_full`. Same optimizer, warmup and cosine schedule as A5-6 at 4×. Seeds 0, 1, 2.

All new runs: A1-7 seed 2, τ = 2 seed 0, τ = 8 seed 0, A5-short seeds 0–2 (stage A, in
parallel); then the selected τ's seeds 1 and 2 (stage B). Eight runs, ≈1 GPU-h each.

## Decision rule

Same margin as the original study. Let F be the selected fixed-τ arm (3 seeds) and A the A5-6
arm at `4·B_full` (seeds 0, 1, 2). With `sd` the sample standard deviation over the three seeds
(n − 1 = 2) and `sd_pool = sqrt((sd_A² + sd_F²) / 2)`:

- **"Annealing beats fixed temperature"** if `mean_test(A) − mean_test(F) ≥ 1.0 pp` **and**
  `> 2·sd_pool`.
- Otherwise **"soft targets plus patience suffice"**.

Secondary, descriptive only: A5-short versus A5-6 (same arithmetic, no verdict), and A1-7 (τ = 1,
3 seeds) versus both, to show the size of the effect over the baseline.

## Test split

Evaluated once, at the end, after all runs have finished and τ has been selected, for all λ = 0
`4·B_full` runs: A1-7 seeds 0–2, A2-3 seed 0, A5-6 seeds 0–2, A1MSE-1 seed 0, the fixed-τ runs
(τ = 2, 4, 8 seed 0 and the selected τ's seeds 1 and 2), and A5-short seeds 0–2. Output:
`/workspace/runs/test_results_followup.json`. The script refuses to run twice.

## Budget

GPU-hour cap raised to **50** (Fred, 2026-10-02). Used before this follow-up: 30.1 GPU-h (GPU
busy-clock). Estimate for this follow-up: ≈8 GPU-h, projected total ≈38. Stop and ask if a
projection would exceed 50.

## Reporting

`RESULTS_FOLLOWUP.md`: per-seed validation, 50k train and test values for every run; agreement,
train agreement and train KL against compute; the τ selection table; the decision with its
arithmetic; the secondary comparisons; and failure modes. The backup is updated at the end.
