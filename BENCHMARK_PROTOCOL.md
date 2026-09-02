# QuTE Amortized Backend Benchmark Protocol

**Protocol:** `m1_r1_benchmark_v1`
**Status:** pilot protocol frozen; confirmatory execution blocked
**Scope source:** [`R1_SCOPE.yaml`](R1_SCOPE.yaml)

## 1. Purpose

Measure the joint accuracy–runtime–memory–amortization frontier of a circuit-conditioned sample generator. Accuracy alone and model-forward latency alone cannot yield PASS.

The pilot is a feasibility check. It cannot support the paper's confirmatory claim, architecture selection, scale claim, or economic claim.

## 2. Execution modes

For every backend and shot count, report:

1. cold single-circuit latency;
2. warm single-circuit latency;
3. batched-circuit throughput;
4. small-shot and large-shot workloads;
5. parse, encode, execute/sample, and package components;
6. identical circuit corpus, seeds, hardware process, precision, and output requirement.

Compilation and model caches must be either disabled for all methods or reported as separate cold/warm modes.

## 3. Exact baselines

Required for confirmatory work:

- Qiskit Aer statevector sampling;
- Qiskit Aer matrix-product-state;
- an available tensor-network simulator selected and version-frozen in Phase 4.

Pilot baseline:

- local verified exact statevector implementation;
- Qiskit Aer cross-check when available;
- exact probability enumeration only at six qubits.

Exact outputs must agree within numerical tolerance before they are accepted as teacher or evaluation data.

## 4. Approximate and learned baselines

- uniform distribution;
- empirical or independent per-qubit marginals;
- low-order/pairwise model if implemented without changing the frozen pilot;
- task-specific QAOA-energy regressor;
- simple circuit-conditioned autoregressive sampler;
- candidate graph-aware autoregressive QuTE model;
- per-circuit NQS/statebank-style comparator if a reproducible implementation is available and its fitting cost is fully charged.

A missing optional baseline is reported as missing; it is not silently approximated by a weaker method.

## 5. Accuracy metrics

### Small exact-validation regime

- total variation distance (primary pilot distribution metric);
- Hellinger fidelity;
- Jensen–Shannon divergence;
- probability-mass coverage;
- top-`k` outcome overlap.

KL divergence is secondary and must specify smoothing or support handling.

### Physics and task utility

- expected MaxCut energy error and error per edge;
- approximation-ratio error;
- optimal-solution probability error;
- feasible-solution probability;
- single-qubit marginals;
- pairwise `ZZ` correlations;
- parameter-point ranking correlation;
- optimizer-trajectory agreement in confirmatory work.

All aggregate tables include median, interquartile range, 90th and 95th percentiles, and worst declared family. Family macro-averages accompany pooled averages.

## 6. Systems metrics

Timed boundary:

```text
input validation
+ circuit parse
+ support check
+ feature encoding
+ model inference
+ ancestral sampling
+ counts/result packaging
```

Report:

- total and component latency;
- circuits/s and shots/s;
- peak process RSS;
- peak device memory, where applicable;
- model checkpoint bytes;
- teacher corpus bytes;
- warm-up count, repetitions, batch size, thread limits, precision, and synchronization points.

GPU timings synchronize before and after measured regions. Peak memory is measured in isolated child processes for final results; combined-process high-water marks are not attributed to one backend.

## 7. Hardware record

Every run records:

- host and accelerator model;
- CPU core/thread count;
- host RAM and accelerator VRAM;
- OS, Python, NumPy, PyTorch, Qiskit, Aer, and tensor-network versions;
- precision and deterministic-algorithm settings;
- environment thread limits;
- model and source commit hashes.

The first pilot environment is Apple arm64 macOS, CPU execution, Python 3.13 venv, NumPy 2.5.2, PyTorch 2.10.0, Qiskit 2.5.2, and psutil 7.2.2. This is pilot evidence only; confirmatory hardware remains unfrozen.

## 8. Total-cost accounting

For `M` evaluated circuits with the frozen shot requirement:

```text
C_QuTE(M)  = C_corpus_generation
             + C_teacher_simulation
             + C_teacher_sampling
             + C_training
             + M * C_qute_execution

C_direct(M) = M * C_direct_execution
```

When `C_direct_execution > C_qute_execution`:

```text
M* = (C_corpus_generation + C_teacher_simulation
      + C_teacher_sampling + C_training)
     / (C_direct_execution - C_qute_execution)
```

Otherwise `M* = infinity` and the economic condition fails. Storage and energy proxy are reported separately; neither is converted into money without a frozen pricing model.

Teacher data reused by multiple experimental candidates is charged once in workload-level accounting and also reported per candidate under both equal-share and full-charge views.

## 9. Reliability protocol

The support checker runs before neural execution. Confirmatory reliability reports:

- support score and calibrated error estimate;
- uncertainty–error rank correlation;
- OOD AUROC/AUPRC;
- risk–coverage curve;
- error-threshold violation rate at declared coverage;
- abstention and fallback rates;
- latency and cost after fallback.

Calibration uses development data only. The uncertainty threshold and fallback policy freeze before confirmatory access.

## 10. Pilot gate, frozen before pilot execution

All are required:

- generated counts sum exactly to requested shots;
- zero optimizer steps occur per inference circuit;
- zero exact simulator calls occur on the neural inference path;
- median IID TVD ≤ `0.35`;
- median absolute cut-energy error per edge ≤ `0.10`;
- end-to-end latency, peak RSS, model size, teacher cost, and training time are recorded;
- direct exact baseline and known failures are reported.

No pilot speedup is required because six-qubit exact simulation is expected to be competitive. Pilot PASS means “vertical slice is credible enough for profiling,” not “QuTE is economically useful.”

## 11. Confirmatory success criteria

Exact numerical confirmatory thresholds are intentionally **not chosen in Phase 0**. Choosing them after inspecting pilot data is allowed only once, in Phase 4, before any fresh confirmatory corpus is generated. The frozen criteria must require simultaneously:

```text
accuracy condition
AND systems condition
AND finite economic condition
AND declared support envelope
```

The confirmatory registry must hash thresholds, corpus membership, seeds, tests, baseline versions, hardware, and model-selection rules. Any change creates a new protocol version and new untouched corpus; it cannot revise the existing verdict.

## 12. Statistical analysis

- circuit is the primary unit, not individual shots;
- use paired per-circuit comparisons on identical corpus members;
- bootstrap confidence intervals resample circuits within predeclared families;
- multiple OOD families use family-level reporting and a predeclared correction when formal hypothesis tests are used;
- report effect sizes and intervals, not only `p` values;
- model-selection seeds are separate from final reporting seeds;
- pilot data are excluded from confirmatory inference.

## 13. Validation beyond exact statevector range

No “accurate” claim is allowed without at least one independent check: tractable subclass, analytic observable, high-quality tensor network, QPU measurement, cross-method agreement, conserved quantity, or symmetry. Otherwise the result is labeled `UNVERIFIED_EXTRAPOLATION`.

## 14. Stop rules

Stop or narrow the support envelope if:

- teacher/training cost yields no realistic break-even;
- inference is not faster under the same output requirement;
- output tails or task rankings fail the accuracy budget;
- OOD confidence does not control risk;
- the support envelope becomes too narrow to represent repeated practical work.

A stopped study is a valid negative result. Thresholds do not move.
