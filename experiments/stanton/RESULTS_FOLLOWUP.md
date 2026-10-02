# Results of the follow-up: annealing versus fixed temperature at λ = 0, 4× compute

Preregistration: `FOLLOWUP_PREREG.md` (committed 2026-10-02 before any run; Amendment F1 added
τ = 16 before any result existed). The original study and its decision rule are closed. Test
split evaluated once for all 16 λ = 0 `4·B_full` runs (`/workspace/runs/test_results_followup.json`);
decision arithmetic in `followup_decision.json`; τ selection in `followup_tau_selection.json`.
"±" is the sample standard deviation over seeds 0, 1, 2 (n − 1 = 2).

## Verdict

**Soft targets plus patience suffice.** At λ = 0 with `4·B_full`, Annealing-KD (A5-6) does not
beat the best fixed temperature (τ = 8):

| | test agreement (seeds 0, 1, 2) | mean ± sd | test KL |
|---|---|---:|---:|
| A5-6 (annealing) | 85.44, 85.48, 82.78% | 84.57 ± 1.55% | 0.128 |
| fixed τ = 8 | 85.00, 87.96, 86.40% | 86.45 ± 1.48% | 0.092 |

Rule: A5-6 − τ8 = **-1.89 pp**; the rule needed ≥ +1.0 pp and > 2·sd_pool =
3.03 pp (sd_pool = 1.51 pp). The fixed temperature is
numerically *ahead*, within 2 SD. Disclosure (see the preregistration): A5-6's three test values
and τ = 4's seed-0 test value were known beforehand from a post-hoc evaluation; the re-evaluation
reproduced them exactly. The genuinely new test information was τ = 8 (all seeds; τ = 8 was not
run before this follow-up) and τ = 2, 16, A5-short and A1-7 seed 2.

## τ selection (validation only, seed 0, end of `4·B_full`)

| τ | val agreement | val KL |
|---:|---:|---:|
| 2 | 74.04% | 0.406 |
| 4 | 80.24% | 0.220 |
| **8 (selected)** | **86.20%** | **0.106** |
| 16 | 82.82% | 0.171 |

Validation agreement peaks at τ = 8 and falls at τ = 16 (single seed per τ except τ = 8).

## All runs (λ = 0, `4·B_full`)

| run | seeds | val agree per seed | train agree (50k) per seed | train KL per seed | **test agree per seed** | test mean ± sd | test KL mean | test acc mean |
|---|---|---|---|---|---|---:|---:|---:|
| A1-7 (τ = 1) | 0, 1, 2 | 73.22, 72.76, 72.34 | 87.28, 87.31, 87.34 | 0.200, 0.203, 0.201 | 72.12, 72.12, 72.36 | 72.20 ± 0.14 | 0.459 | 68.49 |
| fixed τ = 2 | 0 | 74.04 | 84.76 | 0.207 | 73.98 | 73.98 | 0.413 | 67.48 |
| fixed τ = 4 | 0 | 80.24 | 85.85 | 0.159 | 79.78 | 79.78 | 0.224 | 66.40 |
| **fixed τ = 8 (selected)** | 0, 1, 2 | 86.20, 89.08, 86.70 | 90.55, 92.62, 90.72 | 0.076, 0.048, 0.070 | 85.00, 87.96, 86.40 | 86.45 ± 1.48 | 0.092 | 65.61 |
| fixed τ = 16 | 0 | 82.82 | 87.33 | 0.128 | 81.50 | 81.50 | 0.174 | 66.08 |
| **A5-6 (Annealing-KD, annealed over 2·B_full)** | 0, 1, 2 | 85.02, 86.10, 84.22 | 89.54, 90.21, 87.92 | 0.091, 0.082, 0.117 | 85.44, 85.48, 82.78 | 84.57 ± 1.55 | 0.128 | 65.72 |
| A5-short (annealed over 0.5·B_full) | 0, 1, 2 | 87.76, 85.52, 87.04 | 90.88, 89.04, 90.73 | 0.068, 0.097, 0.074 | 86.68, 83.78, 85.78 | 85.41 ± 1.48 | 0.108 | 65.84 |
| A2-3 (moving target) | 0 | 73.46 | 86.90 | 0.197 | 72.80 | 72.80 | 0.432 | 67.76 |
| A1-MSE | 0 | 74.76 | 80.36 | 0.272 | 74.38 | 74.38 | 0.380 | 65.08 |

Figure: `figures/followup_lambda0_4x.png` (top: main arms, seeds 0–2; bottom: the τ sweep at
seed 0; validation agreement, 50k train agreement and train KL against compute).

## Secondary comparisons (descriptive only)

| comparison | difference | 2·sd_pool | ≥ 1 pp and > 2·sd_pool? |
|---|---:|---:|---|
| A5-short − A5-6 | +0.85 pp | 3.03 pp | no (within 2 SD) |
| A5-6 − A1-7 (τ = 1) | +12.37 pp | 2.20 pp | yes |
| τ = 8 − A1-7 (τ = 1) | +14.25 pp | 2.10 pp | yes |
| A5-short − A1-7 (τ = 1) | +13.21 pp | 2.11 pp | yes |

- **Annealing length does not matter**: annealing compressed into the first `0.5·B_full` (A5-short)
  is indistinguishable from annealing over `2·B_full` (A5-6), +0.85 pp.
- **All soft-target arms beat the τ = 1 baseline by 12–14 pp** at 4× compute, with small seed
  spread in the baseline (72.20 ± 0.14%).

## Interpretation

At λ = 0, the large fidelity gain at 4× compute found post hoc in the original study comes from
**soft targets plus long training**, not from annealing. A fixed τ = 8 does at least as well as
Annealing-KD; the annealing schedule's length does not matter; τ = 1 KL, τ = 1 logit MSE (A1-MSE)
and the moving target (A2-3) stay at 72–74%. The effect needs both ingredients: at `1·B_full` all
arms were within ≈1–2 pp of each other (original study), and at 4× compute τ = 1 gains only ≈1 pp.
The τ curve is not monotone (τ = 8 > τ = 16 > τ = 4 > τ = 2 at seed 0), so "large τ ≥ 10 is best
with long training" (the stated motivation for τ = 16) does not hold here; τ = 16 was 3.4 pp below
τ = 8 on validation at seed 0.

Fidelity and accuracy separate again: the high-fidelity students (τ = 8, A5-6, A5-short) score
65.6–65.8% test accuracy, essentially the teacher's 65.80%, while the τ = 1 students reach 68.5%.

## Failure modes and caveats

- Only seed 0 for τ = 2, 4, 16, A2-3 and A1-MSE; the τ selection rests on one seed per τ.
- Partly unblinded rule (see Verdict). Validation and test agree closely for every run here.
- A5-short seed 0 briefly collapsed to 14% validation agreement at ≈0.5·`B_full`, when its
  compressed annealing ends and close to the end of the lr warmup (0.4·`B_full`); it recovered by
  the next evaluation and its final value is unaffected. No other run showed this.
- Stretched schedules: in every 4× run the warmup and cosine decay are stretched with the budget;
  "4× compute" is not "the 1× run continued".
- Resumed runs: at ≈10:05 I stopped the follow-up's first orchestrator to add τ = 16, which also
  killed the six running stage-A runs (A1-7 seed 2, τ = 2 and τ = 8 seed 0, A5-short seeds 0–2).
  All six resumed from their first checkpoint (step 6257 of 312,800, i.e. 2% of the run) with model, optimizer,
  sampler RNG, FE counter and schedule state restored exactly; the ≈2% of work after that
  checkpoint was recomputed (not bit-identical, cuDNN). No run was discarded and nothing failed.

## Compute

Follow-up: ≈9.0 GPU-h (9 new 4× runs). GPU busy-clock total: **39.1 GPU-h of the raised cap of
50**.
