# Numerical notes

## Measure the primal quantity

The authoritative linear metric is `||J delta - b||`, evaluated by applying
the live rectangular operator. A small `||J.T (J delta - b)||` can coexist with
a large output residual in weakly coupled directions.

The authoritative nonlinear metric is the freshly replayed model residual.
No derived certificate replaces it.

## Dense small-residual solver

For small transfer sets, DREAM forms

```text
G = J J.T
delta = J.T G^+ b.
```

This selects a minimum-parameter-norm solution of the local linear equation.
Because `G` squares the condition number, the implementation refines using

```text
e <- b - J delta
delta <- delta + J.T G^+ e.
```

Each pass is certified with a new live `J delta`.

## Matrix-free large-residual solver

When `J` cannot be materialized, the repository supplies rectangular LSQR over
JVP/VJP access. Budget exhaustion is classified as unresolved numerical
access, not proof that a root does not exist.

## Conditioning versus relevant amplification

The worst-case condition number alone does not determine the continuation
step. The direction-specific quantity

```text
||J^+ b|| / ||b||
```

is often more informative: it measures how much parameter motion the actual
target direction requires. Larger transfer sets can expose weak directions,
increase this amplification, and force smaller steps for local branch tracking.

## Root selection

An underdetermined transfer set admits many exact roots. A minimum-norm linear
correction, an Adam endpoint, and a continuation-selected endpoint can belong
to different branches. Exact training equality therefore does not identify a
unique student function.

