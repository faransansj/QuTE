# M2 Operating-Region Report

## Executive summary

**Decision: `PROCEED_TO_M3`.** The harness correctness gate passed, both required Aer methods were measured, and measured candidate cells exist at 22–24 qubits. This authorizes a small scale-capable model study; it is **not** evidence that QuTE already beats Aer.

The main operating region is random 3-regular QAOA at `n=22–24`, `p=1–3`, 4,096 shots. Aer statevector is the best measured baseline there. MPS exceeded the 120-second per-circuit limit at random 3-regular `n=20,p=2/3`, while statevector remained below 5.4 seconds through measured `n=24`. The cycle control remained much more MPS-friendly at low depth.

## Hardware and corpus

- Apple arm64 host, 10 physical/logical cores, 16 GiB RAM.
- Python 3.13.5, Qiskit 2.5.2, Aer 0.17.2, NumPy 2.5.2, PyTorch 2.10.0.
- Fixed one-thread policy: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, PyTorch threads 1.
- Base widths 6–20; guarded adaptive widths 22–32; depths 1–3; cycle and random 3-regular graphs.
- Three graph seeds and four deterministic Sobol parameter points per cell; all backends consumed the same manifest.
- The preregistered Erdős–Rényi stress family was not activated because MPS did not satisfy the activation rule through `n=20,p=3`.

## Correctness

`correctness_report.json` is `PASS`:

- legacy exact vs Qiskit statevector maximum probability error: below `1e-12`;
- 6Q MPS/statevector TVD, energy, and marginal checks: below `1e-10` tolerances;
- frozen pilot median TVD reproduced exactly: `0.2709180816`;
- frozen pilot median energy error/edge reproduced exactly: `0.0433674532`;
- per-circuit optimizer steps: `0`;
- neural-route exact simulator calls: `0`;
- 4,096 neural counts produced.

## Measured results

### Warm single-circuit latency, 4,096 shots

| Family | n | p | Statevector median | MPS median | Best measured baseline |
|---|---:|---:|---:|---:|---|
| cycle | 20 | 1 | 85.3 ms | 147.6 ms | statevector |
| cycle | 20 | 3 | 215.7 ms | 2,746.2 ms | statevector |
| cycle | 22 | 3 | 1,126.7 ms | 3,831.5 ms | statevector |
| cycle | 24 | 1 | 1,092.3 ms | 211.6 ms | MPS |
| cycle | 24 | 3 | 4,596.6 ms | 4,462.3 ms | MPS |
| random 3-regular | 20 | 1 | 114.7 ms | 2,999.1 ms | statevector |
| random 3-regular | 20 | 2 | 175.2 ms | timeout | statevector |
| random 3-regular | 20 | 3 | 227.7 ms | timeout | statevector |
| random 3-regular | 22 | 1 | 581.9 ms | 8,792.9 ms | statevector |
| random 3-regular | 22 | 2 | 947.3 ms | projected skip after timeout | statevector |
| random 3-regular | 22 | 3 | 968.5 ms | projected skip after timeout | statevector |
| random 3-regular | 24 | 1 | 2,314.3 ms | 3,859.1 ms | statevector |
| random 3-regular | 24 | 2 | 2,753.1 ms | projected skip after timeout | statevector |
| random 3-regular | 24 | 3 | 5,313.0 ms | projected skip after timeout | statevector |

All exact medians and dispersion statistics are in `aggregate_results.csv`.

### Practical boundaries

- **Statevector:** measured through `n=24`. At `n=24`, median latency was 1.09–5.31 seconds and peak process RSS was about 579–580 MiB. `n=26` was not run: the 1 GiB theoretical complex128 payload and 4 GiB guarded peak projection exceeded 70% of then-available host memory. This is a guarded host boundary, not an OOM boundary.
- **MPS cycle:** measured through `n=32`; medians at `n=32` were 288.8 ms (`p=1`), 745.5 ms (`p=2`), and 7,166.9 ms (`p=3`).
- **MPS random 3-regular:** `p=1` measured through `n=28` (23.76 s); `p=2/3` failed to complete the `n=20` cell within the 120-second per-circuit guard and were not extended.

### Memory

Statevector theoretical payload and process RSS remain separate. Measured statevector peak RSS rose from about 339 MiB at `n=20`, to 387 MiB at `n=22`, to 579–580 MiB at `n=24`. MPS cycle process RSS remained about 325–329 MiB through `n=32`; random 3-regular MPS reached higher cost before timeout. These values include Python/Aer process overhead.

## Current QuTE bottleneck

The frozen 6Q checkpoint was loaded without training. At 4,096 shots, the instrumented median was 152.97 ms; the independent full-route rerun median was 185.1 ms versus the frozen 159.822 ms, attributable to run/load conditions rather than a changed checkpoint.

| Component | Median at 4,096 shots |
|---|---:|
| input validation | 0.007 ms |
| one-time circuit encoding | 0.006 ms |
| tokenization / graph-feature construction | 147.88 ms |
| decoder forward | 2.13 ms |
| random sampling | 0.11 ms |
| counts aggregation | 2.81 ms |
| tensor transfer | 0.009 ms |
| Result packaging | 0.012 ms |
| model load, reported separately | 1.36 ms; 1.23 MiB RSS delta |

Feature construction inside sequential autoregressive generation is the bottleneck, not neural matrix multiplication or Qiskit-style packaging. The three-point linear shot fit produced an unphysical negative intercept, so it is retained only as a diagnostic planning fit and must not be extrapolated as performance evidence.

## Batch and shot scaling

At 4,096 shots, QuTE delivered about 5.1 circuits/s at batch 8 and 32 because the frozen backend processes circuits serially. Aer statevector random 3-regular 6Q increased from about 39.8 circuits/s at batch 1 to 461 circuits/s at batch 32; at 20Q p=1 it stayed near 7–8 circuits/s. MPS batching did not overcome the high-entanglement random 3-regular cost.

For random 3-regular statevector `n=20,p=3`, median latency was 221.9 ms at 1,024 shots, 227.7 ms at 4,096 shots, and 351.4 ms at 65,536 shots. MPS did not complete that cell. Frozen QuTE rerun medians were 43.8 ms, 185.1 ms, and 3,158.9 ms respectively, confirming near-linear sampling cost at high shots.

## Teacher-data feasibility

At 4,096 shots, 10,000 random 3-regular teacher circuits require the following serial estimates:

| Cell | Serial time | Ideal 8-worker time | Sparse-count upper-bound storage |
|---|---:|---:|---:|
| 20Q p=1/2/3 | 0.32 / 0.49 / 0.63 h | 0.04 / 0.06 / 0.08 h | 430 MiB each |
| 22Q p=1/2/3 | 1.62 / 2.63 / 2.69 h | 0.20 / 0.33 / 0.34 h | 430 MiB each |
| 24Q p=1/2/3 | 6.43 / 7.65 / 14.76 h | 0.80 / 0.96 / 1.85 h | 430 MiB each |

A 10,000-circuit **total** stratified M3 corpus across these nine cells is approximately 4.2 serial hours and 430 MiB under the same linear assumptions. Worker scenarios are idealized; no cluster availability is asserted. Projections for 10,000, 100,000, and 1,000,000 circuits are in `teacher_cost_projection.json`.

## Candidate operating region

Nine cells satisfy the preregistered 250 ms / 256 MiB planning target and at least 2× potential against the best available baseline. The primary recommendation is random 3-regular `n=22–24`, `p=1–3`, 4,096 shots. At 22Q its measured best-classical latency is 581.9–968.5 ms; at 24Q it is 2,314.3–5,313.0 ms. A 100 ms neural target implies at least 5.8× at the weakest primary candidate and over 50× at the strongest, if accuracy and memory targets are also met.

Cycle `n=22,p=3` and `n=24,p=2/3` are retained as controls, not as the main workload. This avoids selecting a region that only beats statevector while ignoring MPS.

## Limitations

- QuTE accuracy and full execution remain valid only for frozen 6Q p=1 cycle-plus-chord circuits.
- No larger-width neural output was generated; all larger-width QuTE values are budgets, not measurements.
- MPS timeout evidence supplies a lower bound, not a precise latency, for random 3-regular `p=2/3` at and above 20Q.
- Measurements are specific to this Apple CPU, Aer version, thread policy, and observed memory availability.
- RSS includes interpreter and library overhead; GPU memory is not applicable on this CPU-only run.
- Scaling fits extrapolate at most four qubits and are planning-only.
- No confirmatory corpus, large model, QPU job, or noisy-QPU experiment was run.

## M2 decision

`PROCEED_TO_M3`: benchmark correctness passed; statevector and MPS were measured; measured candidate regions exist; exact statevector or MPS validation remains available through the recommended range; a 10,000-circuit exploratory teacher corpus is feasible; and the next model has explicit 100 ms / 256 MiB engineering targets.
