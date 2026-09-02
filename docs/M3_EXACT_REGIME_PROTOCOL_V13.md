# M3 Exact-Regime Scale Gate Protocol v13

## Holdout reset

The v2–v12 corpus (graph seeds 92000/92001, Sobol indices 3/5/7/11) was inspected repeatedly and is now development evidence, not an untouched pass/fail holdout. It cannot support a final M3 pass claim.

Before training v13, this protocol freezes a new confirmatory robustness corpus: graph seeds 93000/93001 and nonzero Sobol indices 13/17/19/23, across both families, widths 18/20/22/24, and depths 1/2/3. Thresholds remain unchanged. The corpus is evaluated only after the checkpoint is final.

## Model update

V13 starts from the v11 contextual checkpoint. Development failures consistently concentrated at p=2, so the existing teacher-forced edge-correlation auxiliary loss receives a preregistered 3x p=2 weight. Fine-tuning uses the unchanged 12,000-circuit development mixture for 20 epochs, learning rate 0.0003, and base correlation-loss weight 30. No confirmatory examples enter training.

Every main, new-confirmatory median/p90, systems, route, scale, and boundary check must pass for `M3_PASS_EXACT_REGIME`. QPU work remains unauthorized regardless until this decision is recorded and a separate QPU-only gap is defined.
