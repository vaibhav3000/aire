# The gate decides with paired tests on per-case results

Status: accepted (Step 2)

A change beyond the tolerance blocks a candidate only when a paired test over
the cases says it is unlikely to be chance: McNemar's exact test for metrics
that are 0 or 1 on every case, a seeded paired bootstrap of the mean difference
for the rest. The tolerance keeps its Step 1 meaning (the smallest change that
matters); significance answers whether the eval set supports the change. A drop
beyond tolerance that is not significant is reported as `inconclusive` and only
blocks with `--fail-on-inconclusive`. Rows with fewer than ten paired values
keep the Step 1 tolerance rule, because resampling a handful of values gives no
usable interval and a gate that can never block is worse than a strict one.

## Considered options

- Point estimates only (Step 1): one flipped case on a small suite moves a
  mean by several points, so the gate blocked or passed on noise.
- Non-inferiority (block unless the interval's lower bound clears minus the
  tolerance): the strictest release rule, but on a 150 case suite four cases
  changing in opposite directions already widen a 95% interval of a 0/1
  metric past two points (about 1.96 * sqrt(4) / 150 = 0.026), so it would
  block candidates whose per-case results merely shuffle.
- Unpaired tests on the two means: ignore that both runs answered the same
  questions and need far more cases for the same power.

## Consequences

The bootstrap is seeded (seed 0, 2000 resamples by default) so committed
comparisons are byte-for-byte reproducible. Intervals describe uncertainty from
the choice of cases, not run-to-run model noise, which needs repeated runs.
