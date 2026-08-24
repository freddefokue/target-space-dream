# Experiments

The thesis separates questions that are easy to conflate.

## Does a local root exist?

Small residual spaces permit an explicit float64 Jacobian and a dense
residual-space oracle. On Pythia-14M, this established exact local roots and
showed that the nonlinear model closely followed the linear prediction. On the
CIFAR-100 ResNet setup, target-space continuation reached complete exact
endpoints for D8, D16, and D64.

## Can a practical solver access it?

The exact solver scales with the residual dimension and becomes prohibitively
expensive. Matrix-free CG, LSQR, LSMR, and several preconditioners removed
substantial residual energy on larger transfer sets but did not reproduce the
small-D oracle's exact roots within practical budgets. This is an access
result, not a nonexistence proof.

## Does the fitted root transfer?

Exact equality on a small transfer set did not imply high fidelity on held-out
inputs. CIFAR-100 roots remained close to random test behavior at D8--D64.
Earlier Pythia experiments likewise found that fitting an additional small-set
constraint could worsen many held-out prompts. Coverage and solution selection
therefore remain separate from train-set feasibility.

## Does continuation matter?

Yes, in two distinct ways:

- A direct D8 endpoint Gauss--Newton jump found no trusted update, while
  adaptive target-space continuation reached the endpoint.
- At the same D64 intermediate target, sixteen smaller steps selected a root
  with 6.7 times smaller displacement and better held-out target fidelity than
  one larger step.

## How to read the result

The work establishes a feasible local mechanism and identifies its principal
scale barrier. It does not present a production distillation system. The
remaining research problem is to make broad-coverage, locally selected roots
accessible without dense Jacobians, and to test whether those roots determine
the teacher function on unseen queries.

