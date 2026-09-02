# QuTE M1 R1 Pilot Report

**Verdict:** `PILOT_PASS`
**Role:** feasibility only; not confirmatory

## Frozen gate

| Check | Result |
|---|---|
| `counts_sum_to_shots` | PASS |
| `per_circuit_optimizer_steps_zero` | PASS |
| `exact_simulator_calls_on_inference_zero` | PASS |
| `median_iid_tvd_le_0_35` | PASS |
| `median_energy_error_per_edge_le_0_10` | PASS |
| `latency_reported` | PASS |
| `peak_memory_reported` | PASS |

## Evidence

- One checkpoint was trained on **72** circuits and evaluated on **12** unseen-graph circuits.
- Per-circuit optimizer steps: **0**.
- Exact simulator calls on the demonstrated neural route: **0**.
- Counts total: **4096 / 4096**.
- Median TVD: **0.270918** (pilot threshold `<= 0.35`).
- Median cut-energy error per edge: **0.043367** (threshold `<= 0.10`).
- Independent-bit median TVD: **0.373089**.
- Uniform median TVD: **0.372795**.
- Observable-regressor median energy error per edge: **0.045576**.

## Counts example

- `010101`: 227
- `101010`: 202
- `100100`: 111
- `000101`: 108
- `010010`: 108
- `010100`: 107
- `010001`: 102
- `001001`: 98
- `101001`: 96
- `101000`: 92
- `110101`: 89
- `001010`: 87

## Systems result

- Neural parse-to-counts median latency, 4096 shots: **159.822 ms**.
- Exact statevector-to-counts median latency: **0.896 ms**.
- Neural/exact latency ratio: **178.33x slower**.
- Combined process peak RSS: **331.23 MiB**.
- Exact six-qubit state payload: **1.00 KiB**.
- Checkpoint: **54.65 KiB**, 13153 parameters.
- Teacher generation and sampling: **0.018 s**.
- Training: **1.994 s**.
- Break-even: **none; neural execution was not faster in this six-qubit pilot**.

## Interpretation

The feasibility gate passes, but the overall research success condition does **not**: the neural path is slower, uses a larger payload than the six-qubit exact state, and has no finite break-even point. A pass permits Phase 1 profiling only. It does not establish scale, OOD reliability, or replacement of Aer/MPS.

## Known failures

- pilot is six qubits and p=1 only
- uncertainty is not calibrated; support score is a hard envelope check
- Aer and MPS are not installed in the pilot environment
- memory high-water mark combines teacher, training, exact evaluation, and inference
- Qiskit adapter trusts canonical qute_qaoa metadata rather than parsing arbitrary transpiled circuits
- pilot circuits and thresholds are not confirmatory evidence

## Next gate

Proceed only to simulator/workload profiling. Do not generate confirmatory data. Phase 4 must freeze exact thresholds, corpus hashes, seeds, baseline versions, hardware, statistics, model selection, and calibration before confirmatory generation.
