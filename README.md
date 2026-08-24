# DREAM

**DREAM tracks local parameter roots along a path from a student's original
outputs to a teacher's outputs.** Its goal is high-fidelity distillation: not
merely solving a downstream task, but reproducing the teacher's function.

Suppose a student with parameters `theta_0` produces `y_0` and a teacher
produces `y_1` on a transfer set `D`. DREAM defines the constant-speed target
path

```text
y(t) = (1 - t) y_0 + t y_1,       0 <= t <= 1,
```

and repeatedly solves

```text
J(theta_t) delta ~= y(t + dt) - F(theta_t; D).
```

After every update it evaluates the real nonlinear model, rebuilds the
Jacobian, and corrects the complete remaining defect. Failed intervals are
rolled back and retried with a smaller `dt`.

This repository is the curated implementation accompanying the master's
thesis *High-Fidelity Distillation as Root Tracking*. The full research
archive is deliberately kept separate: it contains hundreds of exploratory
drivers and roughly 226 GB of caches, checkpoints, and intermediate results.

## What the experiments establish

The experiments support three bounded conclusions:

1. Exact target-space continuations can reach the complete endpoint on small
   transfer sets in both Transformer and ResNet settings.
2. Direct endpoint optimization and continuation can select very different
   roots, even when both use the same training objective.
3. As transfer coverage grows, nearby branches become increasingly expensive
   to access. Smaller continuation steps can select much smaller and more
   transferable roots, but the dense exact solver becomes impractical.

They do **not** establish lossless distillation on unseen inputs. Exact
interpolation on a small transfer set can still generalize poorly.

## Install and run in ten minutes

Python 3.10+ and PyTorch 2.3+ are required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
dream-toy
pytest
```

The toy run is deliberately tiny and CPU-safe. It verifies the complete
target path, live-refined residual-space solve, adaptive controller, and primal
root gates.

## Repository map

```text
src/dream/          reusable method and solvers
experiments/        thin scientific entry points
configs/            frozen, human-readable example protocols
tests/              numerical invariants and end-to-end smoke tests
artifacts/          compact, sanitized result summaries
docs/               method, numerical notes, and reproduction guide
```

The central files are:

- [`continuation.py`](src/dream/continuation.py): fresh-J predictor/corrector
  loop, rollback, and adaptive target steps.
- [`solvers.py`](src/dream/solvers.py): dense minimum-norm solves with live
  iterative refinement and matrix-free LSQR.
- [`operators.py`](src/dream/operators.py): explicit and JVP/VJP Jacobian
  access.
- [`targets.py`](src/dream/targets.py): centered logits and the straight
  target-space homotopy.

## Canonical CIFAR-100 experiment

The readable CIFAR entry point consumes the frozen cache and teacher checkpoint
from the original run:

```bash
python experiments/cifar100_distillation.py \
  --teacher /path/to/teacher_state.pt \
  --cache /path/to/fixed_cifar100_cache.pt \
  --d 8 \
  --device cuda
```

The D8 experiment materializes a float64 Jacobian and therefore requires a
GPU with substantial memory. Larger exact runs are research workloads, not
installation tests. Artifact formats and provenance are documented in
[`docs/reproduction.md`](docs/reproduction.md).

## Selected results

On the Stanton-inspired CIFAR-100 setup, DREAM reached numerical roots on D8,
D16, and D64. These roots fit their transfer sets essentially exactly, but
small-D test fidelity remained poor.

| Transfer set | Endpoint train RMS | Stages | Jacobian builds | Test teacher agreement |
|---:|---:|---:|---:|---:|
| 8  | 2.56e-15 | 36 | 368 | 1.58% |
| 16 | 3.07e-15 | 24 | 224 | 2.00% |
| 64 | 1.19e-13 | 32 | 547 | 2.45% |

At the shared D64 target `t = 1/256`, taking sixteen `1/4096` steps instead of
one `1/256` step reduced parameter displacement from 36.78 to 5.49 and improved
held-out target RMS from 0.0560 to 0.0315. This is evidence that step size acts
as a root-selection rule; it is not a universal monotonicity theorem.

The sanitized source numbers are in
[`artifacts/summaries/cifar100_core_results.json`](artifacts/summaries/cifar100_core_results.json).

## Numerical discipline

DREAM accepts roots using the independently replayed nonlinear residual.
Normal-equation residuals and normal gradients are diagnostics, not success
criteria. Dense Gram solves are refined against the live rectangular
Jacobian, because forming `J J.T` squares the condition number.

See [`docs/numerical_notes.md`](docs/numerical_notes.md) before changing solver
tolerances or interpreting an iterative solver stall as root nonexistence.

## Status

This is a research reference implementation, not a production long-context
system. The public API may evolve while the archived thesis experiments remain
immutable.

## License

The code is released under the MIT License.

