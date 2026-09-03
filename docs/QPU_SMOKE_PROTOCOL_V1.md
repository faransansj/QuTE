# QPU Smoke Protocol v1

## Goal

Check whether the v13 local neural backend can act as a practical local QPU-like sampler for small quantum circuits: bounded memory, no queue/pending loop for local runs, and descriptive agreement with a real QPU sample.

This is about emulating quantum-computer input/output behavior. It is not an EDA/QAOA optimization experiment.

## Scope

This is a smoke test, not a replacement claim. The first batch should cover more than one circuit style:

- Bell/GHZ-style entanglement circuit.
- Small random Clifford or Clifford+rotation circuit.
- Hardware-friendly entangling layout circuit.
- One QAOA-like structured circuit only as a benchmark case, not the project identity.

Shots: 1024 per circuit.

## Procedure

1. Submit the frozen circuits to one IBM QPU backend.
2. Save backend metadata, job manifest, queue timing, and raw QPU counts.
3. Run the same circuits through `m3_exact_regime_scale_gate_v13/m3_exact_regime.pt` locally when inside its current envelope.
4. Run Aer statevector for exact-checkable small circuits if installed.
5. Report TVD/Hellinger, selected observable error, marginal MAE, correlation/ZZ MAE where applicable, local latency, and local RSS.

## Decision

- `QPU_SMOKE_RECORDED`: QPU job completed and local artifacts were recorded.
- `QPU_SMOKE_BLOCKED`: credentials, dependency, backend, or job completion blocked execution.

Numerical QPU agreement is descriptive only. The main product signal is whether local execution avoids QPU queue latency and simulator memory blow-up while preserving useful output behavior.

## Boundaries

- No PR creation.
- No broad QPU campaign.
- No claim that the model replaces physical QPUs.
- QPU counts include hardware noise; differences are not treated as model-only error.
- QAOA is allowed as one benchmark family, not as the research goal.
