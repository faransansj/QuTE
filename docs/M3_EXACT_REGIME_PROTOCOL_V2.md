# M3 Exact-Regime Scale Gate Protocol v2

## Reason for v2

The frozen v1 gate passed, but a post-gate audit found that its two-point validation grid contained one zero-angle Sobol point. The median was therefore not a sufficient robustness test. V1 artifacts and decision remain immutable. V2 is a preregistered model iteration that adds a disjoint, nonzero-angle robustness corpus and does not weaken any v1 threshold.

## Workload and splits

- Train: random 3-regular primary plus cycle control; n={20,22,24}; p={1,2,3}; 10,000 exact-regime teacher circuits.
- Main validation: n={18,20,22,24}, p={1,2,3}, disjoint graph seeds 91000–91002 and two frozen Sobol points, retained for direct v1 comparison.
- Robustness validation: the same widths/depths/families, disjoint graph seeds {92000,92001}, and nonzero Sobol indices {3,5,7,11} from a fixed 16-point sequence. It is frozen before v2 training.
- Teacher: Aer statevector, 256 samples per train circuit and 65,536 samples per validation circuit.
- No explicit 2^n model output, per-circuit optimization, neural-route exact call, QPU job, or noisy emulation.

## Model iteration

V2 keeps the variable-width graph-conditioned autoregressive route but replaces the rank-limited dot-product pair term with a directional pair MLP, uses hidden width 64, three message-passing layers, and 24 full-corpus epochs. Sampling remains one loop over qubit positions and vectorized over shots.

## Frozen gates

Main validation retains all v1 gates:

- median energy error/edge <=0.05;
- median marginal MAE <=0.03;
- median ZZ MAE <=0.05;
- 18Q exact TVD <=0.30;
- worst warm 4,096-shot median latency <=100 ms;
- maximum incremental peak RSS <=256 MiB;
- 24Q/20Q latency ratio <=1.5;
- 24Q median energy error <=2x 18Q median;
- zero optimizer steps and exact neural-route calls;
- exact validation through 24Q.

The robustness corpus must independently satisfy both median and p90 thresholds for energy error/edge, marginal MAE, and ZZ MAE using the same numerical limits. A single failed robustness check yields `M3_NEEDS_ITERATION`.

## Decisions

- `M3_PASS_EXACT_REGIME`: full 10,000-circuit training plus every main and robustness gate passes.
- `M3_NEEDS_ITERATION`: infrastructure is valid but any model gate fails.
- `M3_BLOCKED`: teacher correctness, integrity, or instrumentation is unreliable.

A pass is simulator-side evidence only. QPU use remains unauthorized until a separate gap analysis identifies information unavailable from exact/MPS validation.
