# Method

## Problem

Let a student output `F(theta, x)` and let a teacher define target outputs
`T(x)` on a transfer set `D`. High-fidelity distillation asks for parameters
whose outputs match the teacher, rather than only their labels or task score.

For language models, DREAM compares vocabulary-centered logits. For the
CIFAR-100 experiment it uses an orthonormal 99-dimensional contrast basis,
which is exactly equivalent to centered-logit Euclidean distance.

## Target-space continuation

Let

```text
y_0 = F(theta_0; D)
y_1 = T(D)
y(t) = (1 - t) y_0 + t y_1.
```

Define the residual

```text
r(t, theta) = F(theta; D) - y(t).
```

The initial student is an exact root because `r(0, theta_0) = 0`. If the
parameter Jacobian `J_t = dF(theta_t; D) / dtheta` has the required local rank,
the implicit-function viewpoint suggests a nearby branch satisfying
`r(t, theta_t) = 0`. Its minimum-norm tangent obeys

```text
J_t dtheta/dt = y_1 - y_0.
```

The straight target path fixes a weakness of the earlier RoPE-peel homotopy:
equal increments in `t` always introduce equal output-space displacement.
Parameter-space difficulty can still change because `J_t` and the root branch
change along the path.

## Algorithm

At an accepted root `(t, theta)`:

1. Propose `t_next = t + dt`.
2. Evaluate the complete nonlinear residual at `t_next`.
3. Solve `J delta ~= -r` using either the exact residual-space solver or a
   matrix-free rectangular solver.
4. Replay candidate multipliers on the nonlinear model and require trusted
   primal decrease.
5. Rebuild `J` and correct the remaining defect until the root gates pass.
6. Accept the interval, or restore the source root and halve `dt`.

The endpoint has the desired meaning: `y(1) = y_1`, so an accepted root matches
the teacher on `D`.

## Claim boundary

The implicit-function argument is local. It does not ensure that:

- a branch reaches `t = 1`;
- a useful branch is numerically accessible;
- the selected root has small cumulative displacement;
- equality on finite `D` extends to unseen inputs.

Those are empirical questions and are reported separately.

