# Reproduction guide

## Reproduction levels

The repository distinguishes three levels:

1. **Installation smoke test** — runs on CPU and checks the complete algorithm
   on an underdetermined linear problem.
2. **Canonical small-D run** — reconstructs the D8 CIFAR-100 continuation from
   frozen inputs and a teacher checkpoint.
3. **Archival thesis reproduction** — requires the complete private research
   archive and substantial GPU time.

This distinction keeps the public repository readable without pretending that
multi-hour dense Jacobian experiments are lightweight examples.

## CIFAR artifact format

`experiments/cifar100_distillation.py` expects:

- a `ResNet20GN1` teacher state dictionary;
- a PyTorch cache containing `train_images_u8` and `balanced_order`.

The original cache also stores labels, test images, and teacher outputs. The
public driver does not read test data while constructing the root.

The original setup used one deterministic crop-and-flip view, a class-balanced
ordering frozen before inspecting teacher outputs, float64 model operations,
and a student initialized as 25% teacher parameters plus 75% independently
initialized parameters.

## Provenance

The compact result file records values copied from authenticated run summaries.
Heavy checkpoints, Jacobians, caches, and stage-by-stage histories are excluded
from Git. Their hashes and an external artifact location can be added after an
archive is deposited on Zenodo or Hugging Face.

## Before claiming reproduction

Check all of the following:

- `t=0` is an exact root;
- target endpoints match the intended student and teacher outputs;
- centered-logit and contrast-space distances agree;
- JVP/VJP adjoint checks pass;
- failed intervals restore parameters exactly;
- root acceptance uses a fresh nonlinear replay;
- held-out data never controls the continuation.

