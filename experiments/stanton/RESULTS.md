# Results: soft Dream on Stanton's CIFAR-100 fidelity benchmark

Test split evaluated once on 2026-10-01 (`/workspace/runs/test_results.json`, 40 models), after
Fred approved the Phase 3 report. Decision rule: `PREREGISTRATION.md` §8 with Amendment 9,
applied mechanically by `decide.py` (output: `decision.json`). Teacher test accuracy: 65.80%.

**"±" is the sample standard deviation over seeds 0, 1 and 2 (n − 1 = 2) everywhere in this
document.** Ranges are given separately as [min, max]. "Agreement" is top-1 agreement with the
teacher; "KL" is the mean KL(p_T ‖ p_S) at temperature 1 in nats. Test = the 5,000 held-out test
images; validation = the other 5,000 official test images; train = all 50,000 distillation images
without augmentation.

## Verdict

**Soft Dream does not work in this setting. The preregistered GO outcome is not met at λ = 0 or
λ = 0.25.** The two Gauss–Newton arms (A3 Gauss–Newton direct, A4 soft Dream) end 20–26 pp below
first-order distillation in test agreement at both λ. At matched compute they afford only a few
hundred expensive steps, and their weights drift far from the teacher's.

Preregistered outcomes (§8, Amendment 9):

| outcome | λ = 0 | λ = 0.25 |
|---|---|---|
| GO (A4 beats all, lower KL) | not met | not met |
| second order is the lever (A3 ≈ A4, both beat the rest) | not met | not met |
| the path is the lever (A2 ≈ A4, both beat the rest) | not met | not met |
| teacher trajectory suffices (A6 ≈ A4, both beat A1) | not met | not met |
| temperature suffices (A5 within 2 SD of A2 or A4, reported when A2 or A4 beats A1) | not applicable (neither A2 nor A4 beats A1) | **met** |
| NO-GO (no arm beats A1 by the GO margin) | **not met** (A5 beats A1) | **not met** (A2 and A5 beat A1) |

The four outcomes that need A3 or A4 are "not met" under Amendment 9: A3 and A4 (seed 0) are
19.8–26.1 pp below A1's 3-seed mean, more than the 10 pp threshold.

What the rule finds instead:

- **λ = 0 (random initialization): only Annealing-KD (A5) beats the baseline**, by a small
  margin: 72.77 ± 0.87% vs 71.13 ± 0.14% test agreement (+1.64 pp; GO margin needs ≥ 1.0 pp and
  > 2·sd_pool = 1.26 pp), with lower KL (0.422 vs 0.490). The moving target (A2, −0.27 pp) and
  the teacher trajectory (A6, +0.55 pp) do not. The A1-MSE diagnostic (A1 with A5's loss and no
  annealing) is within 2 SD of A5 (−0.93 pp, 2·sd_pool = 1.48 pp), so this small gain cannot be
  attributed to annealing rather than to the MSE loss. Higher fidelity comes with lower student
  accuracy (A5 64.97% vs A1 67.44%; the A1 students *exceed* the teacher's 65.80%).
- **λ = 0.25: the moving target (A2) and Annealing-KD (A5) both beat the baseline by a wide
  margin and are statistically indistinguishable from each other** ("temperature suffices" is
  met): A2 94.57 ± 0.94%, A5 96.18 ± 2.97%, A1 81.53 ± 6.46% (A1 is seed-dependent: 88.9, 76.8,
  79.0%). A6 does not (78.20 ± 4.21%). The anchor diagnostic (A1 plus a weak, decaying pull
  towards θ_init) matches the best arm with far less seed spread: 96.17 ± 0.23%.

Interpretation, beyond the rule: in this setup the λ = 0.25 failure of first-order distillation
is real under Stanton's recipe (A1-1 below) and is overcome reliably by *staying close to the
initialization early in training*, which a moving target, a soft early target (temperature or
the MSE logit scale) and an explicit anchor all do. It is not overcome by second-order steps.
Nothing tested substantially closes the gap at λ = 0 at 1× compute; the only notable λ = 0 effect
is A5-6 at 4× compute (patience check, one seed).

## Per-seed results

### Test split (decision-rule data)


#### λ = 0: test split

| arm (config) | agreement, mean ± sd (%) | range (%) | per seed 0, 1, 2 (%) | KL(p_T‖p_S), mean ± sd | KL per seed | accuracy, mean (%) |
|---|---:|---:|---|---:|---|---:|
| A1 (A1-7) | 71.13 ± 0.14 | [71.04, 71.30] | 71.06, 71.04, 71.30 | 0.4896 ± 0.0061 | 0.4897, 0.4834, 0.4957 | 67.44 |
| A2 (A2-3) | 70.87 ± 0.67 | [70.30, 71.60] | 71.60, 70.30, 70.70 | 0.4788 ± 0.0007 | 0.4795, 0.4781, 0.4789 | 67.15 |
| A5 (A5-6) | 72.77 ± 0.87 | [72.10, 73.76] | 73.76, 72.10, 72.46 | 0.4218 ± 0.0099 | 0.4104, 0.4285, 0.4264 | 64.97 |
| A6 (A6-3) | 71.69 ± 0.37 | [71.34, 72.08] | 72.08, 71.64, 71.34 | 0.4591 ± 0.0064 | 0.4659, 0.4582, 0.4532 | 67.17 |
| A1-MSE (diag.) (A1MSE-1) | 71.84 ± 0.59 | [71.16, 72.18] | 72.18, 72.18, 71.16 | 0.4536 ± 0.0083 | 0.4510, 0.4469, 0.4629 | 64.70 |
| anchor μ=0.01 (diag.) (A1-anchor-mu0.01) | 70.35 ± 0.56 | [69.72, 70.78] | 70.56, 69.72, 70.78 | 0.5027 ± 0.0089 | 0.4940, 0.5024, 0.5117 | 66.04 |
| A3 (seed 0 only) (A3-6) | 50.16 | – | – | 1.1219 | – | 47.66 |
| A4 (seed 0 only) (A4-6) | 51.30 | – | – | 1.0984 | – | 48.30 |

#### λ = 0.25: test split

| arm (config) | agreement, mean ± sd (%) | range (%) | per seed 0, 1, 2 (%) | KL(p_T‖p_S), mean ± sd | KL per seed | accuracy, mean (%) |
|---|---:|---:|---|---:|---|---:|
| A1 (A1-7) | 81.53 ± 6.46 | [76.76, 88.88] | 88.88, 76.76, 78.96 | 0.1952 ± 0.1215 | 0.0611, 0.2979, 0.2266 | 66.39 |
| A2 (A2-3) | 94.57 ± 0.94 | [93.50, 95.28] | 93.50, 94.94, 95.28 | 0.0157 ± 0.0062 | 0.0226, 0.0140, 0.0105 | 66.03 |
| A5 (A5-6) | 96.18 ± 2.97 | [92.78, 98.28] | 92.78, 97.48, 98.28 | 0.0090 ± 0.0127 | 0.0237, 0.0022, 0.0011 | 65.71 |
| A6 (A6-3) | 78.20 ± 4.21 | [74.62, 82.84] | 82.84, 77.14, 74.62 | 0.2639 ± 0.1105 | 0.1443, 0.2851, 0.3622 | 67.13 |
| A1-MSE (diag.) (A1MSE-1) | 90.21 ± 4.59 | [85.00, 93.66] | 93.66, 85.00, 91.96 | 0.0538 ± 0.0539 | 0.0171, 0.1157, 0.0287 | 65.55 |
| anchor μ=0.01 (diag.) (A1-anchor-mu0.01) | 96.17 ± 0.23 | [95.90, 96.30] | 96.30, 96.30, 95.90 | 0.0075 ± 0.0009 | 0.0069, 0.0070, 0.0086 | 66.01 |
| A3 (seed 0 only) (A3-6) | 58.88 | – | – | 0.8006 | – | 55.34 |
| A4 (seed 0 only) (A4-6) | 55.40 | – | – | 0.9436 | – | 52.72 |


### Validation and train, per seed (same final models)

#### λ = 0: validation and train (50k, no augmentation), per seed

| arm (config) | val agreement per seed (%) | val mean ± sd | train agreement per seed (%) | train mean ± sd | train KL per seed |
|---|---|---:|---|---:|---|
| A1 (A1-7) | 71.20, 71.36, 71.50 | 71.35 ± 0.15 | 84.62, 84.40, 84.44 | 84.49 ± 0.12 | 0.2533, 0.2605, 0.2579 |
| A2 (A2-3) | 72.00, 72.04, 70.18 | 71.41 ± 1.06 | 83.25, 82.92, 83.16 | 83.11 ± 0.17 | 0.2724, 0.2740, 0.2732 |
| A5 (A5-6) | 72.88, 72.80, 73.04 | 72.91 ± 0.12 | 77.37, 77.27, 77.47 | 77.37 ± 0.10 | 0.3382, 0.3413, 0.3384 |
| A6 (A6-3) | 72.30, 72.18, 71.58 | 72.02 ± 0.39 | 84.07, 83.96, 83.99 | 84.01 ± 0.06 | 0.2602, 0.2604, 0.2556 |
| A1-MSE (diag.) (A1MSE-1) | 71.46, 71.98, 71.86 | 71.77 ± 0.27 | 76.73, 76.78, 76.40 | 76.64 ± 0.21 | 0.3616, 0.3583, 0.3645 |
| anchor μ=0.01 (diag.) (A1-anchor-mu0.01) | 71.08, 70.66, 71.38 | 71.04 ± 0.36 | 83.78, 83.43, 83.67 | 83.63 ± 0.18 | 0.2671, 0.2752, 0.2758 |
| A3 (seed 0 only) (A3-6) | 51.24 | – | 52.32 | – | 1.1374 |
| A4 (seed 0 only) (A4-6) | 51.18 | – | 52.33 | – | 1.1244 |

#### λ = 0.25: validation and train (50k, no augmentation), per seed

| arm (config) | val agreement per seed (%) | val mean ± sd | train agreement per seed (%) | train mean ± sd | train KL per seed |
|---|---|---:|---|---:|---|
| A1 (A1-7) | 89.84, 76.72, 80.18 | 82.25 ± 6.80 | 93.67, 87.01, 88.22 | 89.64 ± 3.55 | 0.0373, 0.1669, 0.1310 |
| A2 (A2-3) | 93.28, 94.68, 95.50 | 94.49 ± 1.12 | 96.04, 96.81, 97.25 | 96.70 ± 0.61 | 0.0143, 0.0089, 0.0069 |
| A5 (A5-6) | 93.36, 98.00, 98.44 | 96.60 ± 2.81 | 95.55, 98.64, 98.96 | 97.72 ± 1.88 | 0.0170, 0.0016, 0.0008 |
| A6 (A6-3) | 84.22, 77.86, 74.46 | 78.85 ± 4.95 | 90.78, 87.02, 85.83 | 87.88 ± 2.58 | 0.0830, 0.1627, 0.2001 |
| A1-MSE (diag.) (A1MSE-1) | 94.28, 86.28, 93.04 | 91.20 ± 4.31 | 96.26, 89.61, 95.00 | 93.62 ± 3.53 | 0.0125, 0.0891, 0.0204 |
| anchor μ=0.01 (diag.) (A1-anchor-mu0.01) | 95.98, 96.52, 96.04 | 96.18 ± 0.30 | 97.83, 97.72, 97.46 | 97.67 ± 0.19 | 0.0044, 0.0046, 0.0055 |
| A3 (seed 0 only) (A3-6) | 59.48 | – | 62.25 | – | 0.8052 |
| A4 (seed 0 only) (A4-6) | 55.88 | – | 57.37 | – | 0.9547 |

Figure: `figures/test_agreement_per_seed.png` (test agreement per seed and mean, per arm and λ).

## Stanton's recipe across λ (Amendment 8; seed 0, `B_full`, validation and train)

| λ | A1-1 val agree | A1-1 train agree (50k) | A1-7 val agree (warmup) | A1-7 train agree |
|---:|---:|---:|---:|---:|
| 0 | 70.42% | 84.11% | 71.20% | 84.62% |
| 0.1 | 71.02% | 84.38% | 70.70% | 84.62% |
| 0.25 | **71.22%** | **84.28%** | 89.84% | 93.67% |
| 0.4 | 97.16% | 98.38% | 96.90% | 98.44% |
| 0.5 | 96.90% | 98.46% | 97.18% | 98.57% |

Under Stanton's exact optimizer recipe, λ = 0.25 behaves like λ = 0 and the jump happens between
0.25 and 0.4, as in Stanton et al. Figure 6(b) (their transition: between 0.25 and 0.375). Budget
versus warmup: at λ = 0.25, A1-1 gains 5.0 pp of validation agreement from 4× more compute
(66.22% at `B_full/4` → 71.22% at `B_full`) but stays below the barrier; adding warmup (A1-7) lifts
seed 0 to 89.8%, but over three seeds A1-7 reaches 82.3 ± 6.8% (validation), i.e. warmup carries
A1 past the barrier only on some seeds. Differences from Stanton's setup: 200- vs 300-epoch-
equivalent budget, λ grid {0, 0.1, 0.25, 0.4, 0.5} (no 0.375), GroupNorm(1, C) instead of their
LayerNorm ResNet-20, a weaker teacher (65.8% vs ≈70.5%), τ = 1 KL instead of their τ = 4 default.
Figure: `figures/sweep_train_agreement_vs_lambda.png` (all arms, seed 0).

## Patience check (Amendments 6 and 7; seed 0, validation and train only)

A1-7, A2-3 and A5-6 at 4× `B_full` (schedules stretched with the budget), against their 1× runs:

| λ | run | val agree 1× → 4× | train agree (50k) 1× → 4× | train KL 1× → 4× |
|---:|---|---|---|---|
| 0.25 | A1-7 | 89.84 → 97.76% | 93.67 → 98.79% | 0.037 → 0.002 |
| 0.25 | A5-6 | 93.36 → 91.42% | 95.55 → 94.29% | 0.017 → 0.028 |
| 0 | A1-7 | 71.20 → 73.22% | 84.62 → 87.28% | 0.253 → 0.200 |
| 0 | A2-3 | 72.00 → 73.46% | 83.25 → 86.90% | 0.272 → 0.197 |
| 0 | A5-6 | 72.88 → 85.02% | 77.37 → 89.54% | 0.338 → 0.091 |

- **λ = 0.25 (Amendment 6).** A5-6 at 1× does not beat A1-7 at 4× (93.36% vs 97.76% validation);
  the two do not converge with more training (A1-7 rises to near-perfect agreement, A5-6 ends
  slightly below its 1× value). Caveats: seed 0 only, and A1-7's seed 0 is its best seed at 1×
  (89.8% vs 76.7 and 80.2%); the 4× runs use stretched schedules, not just more steps.
- **λ = 0 (Amendment 7). Verdict on Fred's pre-recorded criterion: NOT MET.** The criterion
  required A2-3 or A5-6 at 4× to reach ≥ 95.0% train agreement at λ = 0 with A1-7 at 4× at least
  5.0 pp lower. Neither condition holds: A5-6 reaches 89.54% and A2-3 86.90% (both below 95%),
  and A1-7 (87.28%) is 2.26 pp below A5-6 and 0.38 pp *above* A2-3. Noted, not part of any
  decision: A5-6 at 4× reaches 85.02% *validation* agreement at λ = 0, 11.8 pp above A1-7 at 4×
  (73.22%) and A2-3 at 4× (73.46%), its curve separating after ≈2× compute. This is one seed.

Figure: `figures/patience_4x.png`.

## Hand-off diagnostic (Amendment 4; seed 0, `B_full/4` of A1-7, validation)

| source endpoint | λ | before | primary (warmup to 0.1·peak lr) | secondary (warmup to peak lr) |
|---|---:|---:|---:|---:|
| A1-7 (control) | 0.25 | 89.84% | 90.14% | 90.62% |
| A2-3 | 0.25 | 93.28% | 94.00% | 87.42% |
| A6-3 | 0.25 | 84.22% | 84.86% | 85.30% |
| A4-6 | 0.25 | 55.88% | 64.36% | 68.08% |
| A1-7 (control) | 0 | 71.20% | 71.36% | 71.92% |
| A2-3 | 0 | 72.00% | 71.64% | 71.52% |
| A6-3 | 0 | 72.30% | 72.30% | 72.12% |
| A4-6 | 0 | 51.18% | 60.74% | 66.26% |

At λ = 0.25 the A2 endpoint keeps its advantage over the A1 control under the gentle hand-off but
loses it under the full-lr one: continuation reaches a better region that a large lr kick leaves.
At λ = 0 no endpoint differs. The soft Dream endpoints (A4) improve under first-order training but
stay 10–30 pp behind.

## Other diagnostics (seed 0 unless stated)

- **A1-MSE** (Amendment 5; 3 seeds on test): within 2 SD of A1 at both λ (λ = 0: +0.71 pp;
  λ = 0.25: +8.67 pp, 2·sd_pool = 11.2 pp) and within 2 SD of A5 at both λ. The MSE loss alone
  gives most of A5's λ = 0.25 benefit on some seeds (93.7, 85.0, 92.0%) but less reliably.
- **Anchor μ = 0.01** (Amendments 4 and 10; 3 seeds on test): at λ = 0.25 the most reliable arm
  (96.17 ± 0.23%); at λ = 0 no benefit (70.35 ± 0.56%, 0.78 pp below A1). Its benefit depends on
  the teacher component of the initialization. The stronger anchor μ = 0.1 (λ = 0.25, seed 0,
  validation): 91.60%.
- **Distance to the teacher's weights** (`figures/sweep_teacher_distance.png`): first-order arms
  at λ ≥ 0.25 move towards θ_T (to 4–28), at λ ≤ 0.1 away from it; A3/A4 drift to 155–560 at
  every λ, consistent with weight-norm growth in GroupNorm's scale-invariant directions.

## Failure modes and caveats

- **Soft Dream / Gauss–Newton under matched compute.** Each damped GGN-CG step costs
  48k–113k FE (≈85–200 first-order batch-128 steps), so A3/A4 take 100–230 steps per tuning budget and
  under 1,000 per full budget. In most configurations the trust ratio stayed in the 0.25–0.75 band
  where Levenberg–Marquardt keeps the damping unchanged (damping 3–17), so steps stayed small; in
  the selected A3-6 the damping fell to its floor late in tuning. The tuning grid (6
  configurations) varied batch sizes and CG iterations, not the damping rule, trust-ratio
  thresholds or any weight-norm control; a
  differently engineered second-order method could do better. This study tests the
  preregistered soft Dream, not second-order distillation in general.
- **GGN method choice.** `linearize` was chosen by GPU-only cost (≈7.4 FE per product); it costs
  ≈1.5 s of CPU per step, absorbed by concurrency. Charging wall-clock instead would have made the
  GN arms look even more expensive.
- **Seeds.** Three seeds; A1-7 and A6-3 at λ = 0.25 are strongly seed-dependent (sd 6.5 and
  4.2 pp), which widens the pooled SD in every comparison against them.
- **Single tuning point.** All tuning was at λ = 0.25 and a quarter budget; selections may not be
  optimal at λ = 0 or at full budget (e.g. A1-1 vs A1-7 differ little at λ = 0).
- **Accounting.** FE are charged at CUDA-graph GPU cost (Amendment 2). Runs are not bitwise
  reproducible (cuDNN); identical reruns differ by ≈0.5 pp. Eighteen runs were resumed after
  out-of-memory crashes or kills (list in `REPORT.md`, Phase 3); all state that determines the
  trajectory was restored, the work after the last checkpoint was recomputed.
- **Setup differences from Stanton** listed in the A1-1 section.

## Addendum (2026-10-01 20:40): correction to the Phase 3 report

The Phase 3 report (in `REPORT.md`, original text unchanged) stated, from seed 0 only:

> 3. **At λ = 0.25, Stanton's failure case mostly disappears under our recipe**: A1-7 (Stanton's
>    optimizer plus 10% lr warmup) reaches 89.8% validation and 93.7% train agreement at 1×,
>    and **97.8% / 98.8% at 4×** (patience check). The barrier is a compute and optimizer-schedule
>    effect in our setup, not a hard basin barrier.

and, in its comparison with Stanton's Figure 6(b): "In our setup the transition sits between
**λ = 0.1 and 0.25**, not between 0.25 and 0.375".

**Correction.** Both statements are wrong in general. Seed 0 was A1-7's best seed at λ = 0.25:
over three seeds A1-7 reaches 82.3 ± 6.8% validation (89.8, 76.7, 80.2%) and 81.5 ± 6.5% test
agreement. Under Stanton's exact recipe (A1-1, Amendment 8) λ = 0.25 behaves like λ = 0 (71.2%
validation) and the transition lies between 0.25 and 0.4, matching Stanton. Warmup carries A1
past the barrier only on some seeds; the moving target, Annealing-KD and the anchor do so
reliably. The 4× patience result at λ = 0.25 is seed 0 only.

## Compute

GPU busy-clock (pod wall-clock while any study process held the GPU; the figure checked against
the cap): **25.3 GPU-h of 40**. The per-run ledger (`budget.json`) over-counts (≈73) because it
divides wall-clock by each queue's own concurrency while several queues shared the GPU; it is an
attribution only. The test evaluation took 6 s.

## Files

- `experiments/stanton/RESULTS.md` (this file), `REPORT.md` (phase reports, including the
  Phase 3 report and its correction), `PREREGISTRATION.md` (with Amendments 1–10), `PLAN.md`.
- `experiments/stanton/decision.json`: the rule's inputs and outputs per λ.
- `experiments/stanton/figures/`: all figures.
- `artifacts/summaries/stanton_soft_dream_summary.json`: compact summary.
- Per-run data: `/workspace/runs/<run_id>/` (config, metrics, final metrics, 50k train evaluation,
  final weights); `/workspace/runs/test_results.json`.

## Addendum (2026-10-02): POST-HOC λ = 0 follow-up at 4× compute (Amendment 11)

**Post-hoc and exploratory.** Motivated by the seed-0 patience result; recorded in
`PREREGISTRATION.md` (Amendment 11 and its addendum) before any of these runs started. The
decision rule above is closed and unchanged. The test numbers below come from a second, post-hoc
evaluation of the test split (`/workspace/runs/test_results_posthoc_amendment11.json`), made after
the single preregistered one; they do not enter any decision.

All runs: λ = 0, `4·B_full`, schedules stretched with the budget, end-of-run values.

| run | seed | val agree at 1× / 2× / 3× / 4× compute | val agree (end) | train agree (50k) | train KL (50k) | **test agree (post-hoc)** | test KL | test acc |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| A1-7 (baseline) | 0 | 67.1 / 67.9 / 71.7 / 73.2% | 73.22% | 87.28% | 0.200 | 72.12% | 0.455 | 68.22% |
| A1-7 (baseline) | 1 | 67.2 / 68.7 / 70.4 / 72.8% | 72.76% | 87.31% | 0.203 | 72.12% | 0.461 | 68.76% |
| A2-3 (moving target) | 0 | 64.9 / 69.8 / 71.3 / 73.5% | 73.46% | 86.90% | 0.197 | 72.80% | 0.432 | 67.76% |
| A5-6 (Annealing-KD) | 0 | 69.1 / 72.3 / 83.2 / 85.0% | 85.02% | 89.54% | 0.091 | 85.44% | 0.121 | 65.92% |
| A5-6 (Annealing-KD) | 1 | 68.4 / 76.4 / 83.7 / 86.1% | **86.10%** | 90.21% | 0.082 | 85.48% | 0.109 | 65.84% |
| A5-6 (Annealing-KD) | 2 | 68.2 / 74.0 / 81.0 / 84.2% | **84.22%** | 87.92% | 0.117 | 82.78% | 0.153 | 65.40% |
| A1-MSE (MSE, no annealing) | 0 | 69.5 / 71.8 / 74.0 / 74.8% | 74.76% | 80.36% | 0.272 | 74.38% | 0.380 | 65.08% |
| A1-7, fixed τ = 4 (τ²-scaled KL, no annealing) | 0 | 68.9 / 71.9 / 76.3 / 80.2% | 80.24% | 85.85% | 0.159 | 79.78% | 0.224 | 66.40% |

The "1× compute" column is the 4× run's own curve at `B_full` (stretched schedules), not the 1×
sweep run. A5-6 over three seeds at 4×: validation 85.11 ± 0.94%, test 84.57 ± 1.55% (sample
SD); A1-7 over two seeds: validation 72.99%, test 72.12%. Figure: `figures/posthoc_lambda0_4x.png`.

**Verdict on Fred's pre-recorded replication criterion** (both new A5-6 seeds ≥ 80.0% validation
agreement at the end of `4·B_full`): **MET.** Seed 1 reaches 86.10% and seed 2 84.22%, close to
seed 0 (85.02%). The post-hoc test agreement agrees (85.5%, 82.8%; seed 0 85.4%).

Reading (post-hoc, few seeds):

1. **The λ = 0 effect replicates.** At 4× compute, Annealing-KD reaches ≈85% validation and test
   agreement at λ = 0, ≈12 pp above the baseline (A1-7, two seeds, 72.8–73.2%) and the moving
   target (A2-3, 73.5%). The separation appears only after ≈2× compute (at 1× all runs are at
   67–69%).
2. **It is not the MSE loss.** A1-MSE at 4× reaches only 74.8% (test 74.4%), close to the baseline.
   At λ = 0 and 4× compute the MSE loss without annealing does almost nothing.
3. **Soft targets explain part of it; annealing adds the rest.** A fixed τ = 4 (Stanton's default)
   reaches 80.2% (test 79.8%), about 7 of the ≈12 pp. Annealing from τ_max = 10 down to 1
   (A5-6) adds about 5 pp more (one seed for τ = 4).
4. **Fidelity versus accuracy.** The high-fidelity A5-6 students score 65.4–65.9% test accuracy,
   close to the teacher's 65.80%, whereas the baseline students reach 68.2–68.8%, above the
   teacher: more training moves the baseline towards the labels rather than towards the teacher.
5. **The baseline spread at 4× is small** (A1-7: 73.22 vs 72.76% validation, identical test
   agreement), so the gap is not baseline noise.

Caveats: one seed for A1-MSE, τ = 4 and A2-3 at 4×; post-hoc; stretched schedules (a 4× A5-6 run
anneals over 2·`B_full`, which is a different schedule, not just more steps); at λ = 0.25 the
seed-0 4× A5-6 run did *not* improve over 1× (91.4% vs 93.4%, Amendment 6).

Compute for Amendment 11: ≈4.9 GPU-h. GPU busy-clock total: **30.1 GPU-h of 40**.
