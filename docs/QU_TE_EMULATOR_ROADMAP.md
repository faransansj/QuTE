# QuTE Emulator Roadmap

## Direction

QuTE is a learned local emulator for quantum-computer behavior:

```text
quantum circuit + execution request -> QPU-like samples/counts/values
```

It is not an EDA optimizer and not a QAOA project. QAOA remains only one benchmark family because it is structured and already has artifacts in this repository.

## Target contract

A QuTE backend should eventually accept:

- a circuit description;
- shot count;
- measurement basis/request;
- optional target device/noise profile;
- an OOD/fallback policy.

It should return:

- counts or samples;
- selected expectation values when requested;
- timing and memory metadata;
- confidence/OOD metadata;
- a clear abstention/fallback result when outside the trained envelope.

## Current state

The current R2/M3 backend is a useful prototype but still QAOA-shaped internally:

- input type is `Workload` with graph family, QAOA depth, angles, and edges;
- output is counts plus backend metadata;
- v13 passes its frozen simulator-side gates;
- it does not yet accept arbitrary quantum circuits.

## Immediate blocker

Before a broad QPU smoke test, QuTE needs a minimal circuit IR. Without that, any “general quantum-computer emulator” claim would be fake: the code can only emulate the current QAOA-like workload class.

## Next minimal milestone: R3 circuit-in emulator contract

Build the smallest circuit IR that can cover smoke tests:

- qubit count;
- ordered gates;
- gate parameters;
- measurements;
- circuit hash;
- supported gate set: `h`, `x`, `rx`, `ry`, `rz`, `cx`, `cz`, `rzz`, `measure`.

Then add adapters:

1. Qiskit `QuantumCircuit` -> QuTE circuit IR.
2. QuTE circuit IR -> Qiskit circuit for Aer/QPU baselines.
3. QuTE circuit IR -> existing QAOA `Workload` only when the circuit matches the old envelope.

## Benchmark sequence

1. Exact-checkable tiny circuits
   - Bell pair
   - GHZ
   - random Clifford/rotation
   - one QAOA-like circuit

2. Simulator baselines
   - Aer statevector
   - Aer MPS/TN when available

3. Local neural backend
   - latency
   - peak RSS
   - counts/observable agreement
   - OOD/abstention behavior

4. Sparse QPU smoke
   - same tiny circuits
   - raw QPU counts
   - queue time and job wall time
   - descriptive local-vs-QPU comparison

## Claim boundary

Allowed:

> QuTE is a local learned emulator that approximates selected quantum-circuit outputs under a declared trained envelope.

Not allowed:

- universal quantum simulator;
- QPU replacement;
- QAOA/EDA optimizer;
- ideal-state correctness inferred from noisy QPU agreement.

## First implementation task

Do not train a new model yet. First make the interface honest:

1. add circuit IR;
2. add conversion tests;
3. make QPU smoke use the IR;
4. mark unsupported circuits as `ABSTAIN` instead of forcing them through the QAOA backend.
