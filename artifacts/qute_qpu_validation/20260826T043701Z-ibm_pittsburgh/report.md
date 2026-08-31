# QuTE / CC-NQE QPU Validation

## Scope

This validation layer does not modify or retrain the frozen CC-NQE model, benchmark, splits, seeds, or prior results.

## Research Question

> How closely do QuTE / CC-NQE predictions and ideal simulator outputs reproduce measurement statistics observed on a real IBM QPU?

## Experimental Setup

- Model/checkpoint: `artifacts/cc_nqe_p1_p4/checkpoints/transformer_seed11.pt` (CC-NQE circuit transformer, frozen seed 11)
- Benchmark source: `artifacts/cc_nqe_p1_p4/dataset/samples.jsonl`
- Selected splits: IID, Parameter-OOD, Composition-OOD, Depth-OOD
- Circuits: 20
- Backend: `ibm_pittsburgh`
- Shots: 4096
- Transpilation: preset pass manager level 2, seed 20260811
- Bit order: benchmark q0..q3 = displayed bitstring MSB..LSB; benchmark q maps to Qiskit qubit 3-q
- Status: **QPU_SUBMISSION_READY**

## Results

Values are mean / median TVD.

| Split | QuTE-Sim | Sim-QPU | QuTE-QPU |
|---|---:|---:|---:|
| IID | 0.505727 / 0.443419 | N/A | N/A |
| Parameter-OOD | 0.781578 / 0.846336 | N/A | N/A |
| Composition-OOD | 0.934271 / 0.931805 | N/A | N/A |
| Depth-OOD | 0.672322 / 0.671261 | N/A | N/A |

## Hardware Complexity

Per-circuit transpiled depth, 2Q gate count, native gate counts, and physical layout are in `circuit_metrics.json`. Complexity correlations are descriptive and stored in `summary.json`.

## Interpretation

- QuTE ↔ Simulator discrepancy = model approximation / generalization error.
- Simulator ↔ QPU discrepancy = hardware execution gap.
- QuTE ↔ QPU discrepancy = combined model + hardware gap.

A small QuTE-QPU distance does not imply that QuTE learned QPU noise; this model was trained against ideal simulation. The smoke sample is too small for statistical claims.

## Phase 2 Candidate

Only after Phase 1 succeeds: evaluate single-qubit X/Y/Z and selected two-qubit ZZ observables. No basis expansion is submitted by this implementation.
