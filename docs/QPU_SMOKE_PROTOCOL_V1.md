# QPU Smoke Protocol v1

## Goal

Check whether the v13 local neural backend can act as a practical local QPU-like sampler for small QAOA circuits: bounded memory, no queue/pending loop for local runs, and acceptable descriptive agreement with a real QPU sample.

## Scope

This is a smoke test, not a replacement claim. It uses four small circuits from the frozen R2 workload shape:

- cycle 6Q p=1
- cycle 8Q p=2
- random 3-regular 6Q p=1
- random 3-regular 8Q p=2

Shots: 1024 per circuit.

## Procedure

1. Submit the frozen circuits to one IBM QPU backend.
2. Save backend metadata, job manifest, and raw QPU counts.
3. Run the same workloads through `m3_exact_regime_scale_gate_v13/m3_exact_regime.pt` locally.
4. Optionally run Aer statevector if installed.
5. Report TVD, energy error/edge, ZZ MAE, local latency, and local RSS.

## Decision

- `QPU_SMOKE_RECORDED`: QPU job completed and local artifacts were recorded.
- `QPU_SMOKE_BLOCKED`: credentials, dependency, backend, or job completion blocked execution.

Numerical QPU agreement is descriptive only. The main product signal is whether local execution avoids QPU queue latency and simulator memory blow-up while preserving useful workload statistics.

## Boundaries

- No PR creation.
- No broad QPU campaign.
- No claim that the model replaces physical QPUs.
- QPU counts include hardware noise; differences are not treated as model-only error.
