# M2 Operating-Region Profiling Protocol

## Purpose

`m2_operating_region_profile` is exploratory systems profiling. It measures when direct classical QAOA simulation becomes costly enough to justify a scale-capable learned approximate backend. It does not claim QuTE speedup, scalable accuracy, or QPU advantage, and it does not alter the frozen R1 or M1 pilot evidence.

## Corpus

One immutable manifest supplies every backend. Widths are `6,8,10,12,14,16,18,20`; guarded adaptive widths are `22,24,26,28,30,32`; QAOA depth is `p=1,2,3`; graph seeds are `2026,2027,2028`; and four deterministic Sobol parameter points are used per graph. Angles are fixed to `gamma in [0, pi]` and `beta in [0, pi/2]`, with one `(gamma,beta)` pair per QAOA layer. Graph families are cycle and random 3-regular. The conditional Erdős–Rényi stress family is preregistered with `edge_probability=0.5` and `maximum_n=24`; it activates only if, through `n=20,p=3`, MPS median latency is below both 25% of statevector latency and 250 ms per circuit. Logical IDs and SHA-256 hashes cover canonical graph, parameters, and circuit structure.

## Hardware and software

The harness records OS, CPU, logical/physical cores, RAM, Python, Qiskit, Aer, NumPy, PyTorch, psutil, Git status/commit, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, PyTorch thread count, and Aer parallelism. CPU thread policy is fixed before profiling. `qiskit-aer` is an explicit profiling dependency because both required baselines use Aer; no optional tensor-network dependency is installed.

## Timing definitions

Cold start is measured in a fresh subprocess and includes backend/model initialization, preprocessing, execution, sampling, aggregation, and packaging after Python startup. Warm single-circuit latency uses an initialized backend, no result cache, and batch size one. Warm batch throughput uses batch sizes 1, 8, and 32. Shot scaling uses 1,024, 4,096, and 65,536 shots. Execution-only excludes corpus construction and common preparation. Components are stored separately where the backend exposes them. Three warmups precede 10 repetitions for `n<=16`, 5 for `n>16`, and no fewer than 3 for resource-heavy cells. Statistics are median, mean, sample standard deviation, p10, p90, min, and max. GPU runs synchronize before and after timing; this protocol's initial host is CPU-only.

## Memory definitions

Host baseline RSS, sampled peak RSS, and incremental peak RSS are separate fields. Peak RSS is sampled from the isolated child process, not inferred from state payload. GPU baseline allocated, peak allocated, and peak reserved are recorded when available. Model load time, model-load RSS delta, checkpoint bytes, and serialized model bytes are separate. Statevector payload uses Aer precision (`complex128`: `16 * 2**n` bytes) and is never presented as process peak RSS.

## Resource guard

Before every statevector run, the harness records predicted payload, available RAM, calibrated overhead, timeout projection, and verdict. A run is skipped as `SKIPPED_RESOURCE_GUARD` if payload exceeds 50% of available RAM, projected process peak exceeds 70%, or backend preflight is unsafe; as `SKIPPED_TIMEOUT_PROJECTION` if projected execution exceeds the 120-second hard timeout; and as `SKIPPED_UNSUPPORTED` when unavailable. No intentional OOM is permitted. Skips are boundary evidence, not failures. Adaptive widths execute only while this guard passes.

## Correctness

Before profiling, small-width Aer statevector probabilities are compared with the frozen pilot exact implementation. Wherever exact distributions remain practical, Aer MPS is compared with statevector using energy, marginals, ZZ correlations, and probability distance; seeded counts are checked separately. The existing 6Q checkpoint is loaded without training. Its original validation corpus is reproduced to verify zero optimizer steps, zero neural-route exact calls, counts, TVD, and energy error within declared tolerance. Existing pilot artifacts are read-only.

## Analysis

For each workload cell, the best classical baseline is the lower measured median of statevector and MPS. Neural target maps evaluate latency budgets `10,25,50,100,250,500 ms` and memory budgets `256 MiB,512 MiB,1 GiB,2 GiB,4 GiB`. Fits record family, parameters, residuals, confidence intervals, held-out error, fitted range, and at most four extrapolated qubits. Fits are planning evidence only. Teacher costs are linear projections for `10^4,10^5,10^6` circuits under serial, 8-worker, and 32-worker idealized scenarios. Amortization evaluates `M=10^3,10^4,10^5,10^6`; unknown training cost is not invented.

## Decision rules

- `PROCEED_TO_M3`: correctness passes; statevector and MPS are measured; at least one candidate has a validation path, feasible teacher generation, and explicit neural latency/memory targets.
- `PIVOT_WORKLOAD_BEFORE_M3`: statevector grows but MPS remains cheaply dominant throughout relevant QAOA cells, indicating that width-only model scaling is not justified.
- `BLOCKED`: reproducibility, instrumentation, artifacts, or baseline correctness are unreliable.
- `NO_FEASIBLE_REGION_FOUND`: no candidate exists in the measured or `largest measured n + 4` horizon and no evidence supports a workload pivot.

Rules are fixed before execution. A candidate must permit at least 2x at a listed reasonable neural budget, plausibly lower memory, retain an accuracy-validation route, have feasible small-study teacher cost, and must not merely beat statevector while losing badly to MPS.

## Non-goals

No QPU jobs, 120Q claims, noisy-QPU emulation, large variable-width training, production `BackendV2`, deliberate OOM, fitted-only speedup claims, or modification/reinterpretation of frozen pilot evidence or thresholds.
