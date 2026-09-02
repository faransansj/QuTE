# M3 Exact-Regime Scale Gate Protocol

## Role

`m3_exact_regime_scale_gate` is an exploratory, simulator-only scale study authorized by M2. It trains one variable-width amortized sampler and tests accuracy, latency, memory, and width/depth trends at widths where exact statevector or MPS validation remains available. It is not confirmatory evidence and does not authorize QPU work.

## Workload

- Primary: random 3-regular QAOA-MaxCut.
- Control: cycle QAOA-MaxCut.
- Train widths: 20, 22, 24; validation widths: 18, 20, 22, 24.
- QAOA depth: p in {1,2,3}; output shots: 1,024, 4,096, 65,536.
- Angles: deterministic unscrambled Sobol points, gamma in [0,pi], beta in [0,pi/2].
- Train and validation graph-seed namespaces are disjoint. Circuit and split hashes are frozen before teacher generation.

## Staged corpus rule

A 216-circuit calibration corpus is generated first (two families x three widths x three depths x three graph seeds x four angle points). The full 10,000-circuit exploratory corpus is generated only if the calibration model passes the viability screen: finite loss, no exact calls on inference, median validation energy error/edge <=0.15, and warm 4,096-shot latency <=250 ms. This prevents spending the full teacher budget on a structurally inadequate model. The 10,000-circuit corpus is still the M3 gate corpus; calibration alone cannot pass M3.

Teacher output is exact-regime Aer statevector sampling. Training retains a deterministic bounded sample per circuit rather than an explicit 2^n table. Validation uses independent circuits and exact statevector probabilities where the resource guard permits. MPS is cross-checked on selected cells and M2 timeout evidence remains part of the boundary record.

## Model

One graph-conditioned pairwise autoregressive sampler supports variable n without per-circuit optimization or explicit 2^n inference output. A graph encoder runs once per circuit. All teacher-forced token logits are computed in one tensor operation. Sampling has one loop over qubit positions and vectorizes over shots. No exact simulator is reachable from the neural inference route.

## Gates

### Accuracy

All are median over validation circuits, with family/width/depth breakdown also reported:

- cut-energy absolute error per edge <=0.05;
- marginal-probability MAE <=0.03;
- ZZ-correlation MAE <=0.05;
- exact-distribution TVD <=0.30 only where exact enumeration and adequate evaluation are tractable.

### Systems

- warm 4,096-shot end-to-end latency <=100 ms, batch size 1;
- incremental/model-attributable peak memory <=256 MiB;
- per-circuit optimizer steps = 0;
- exact simulator calls on neural route = 0;
- no explicit 2^n inference output.

### Scale readiness

- accuracy is reported at 18,20,22,24Q and p=1,2,3;
- no width has median energy error/edge >2x the 18Q median;
- 24Q warm latency <=1.5x 20Q warm latency;
- statevector validation is completed through 24Q or resource-guarded with MPS/reference coverage;
- MPS/statevector overlap passes on selected tractable controls.

## Decisions

- `M3_PASS_EXACT_REGIME`: full 10,000-circuit corpus used and every accuracy, systems, and scale gate passes.
- `M3_NEEDS_ITERATION`: infrastructure is valid but one or more model gates fail, or the calibration viability screen blocks the full corpus.
- `M3_BLOCKED`: teacher correctness, artifact integrity, or instrumentation is unreliable.

Decision rules are fixed before execution and are not weakened after seeing results.

## Non-goals

No QPU jobs, noisy-QPU emulation, confirmatory corpus, production BackendV2, fallback router, 120Q simulation, or fitted-only success claim.
