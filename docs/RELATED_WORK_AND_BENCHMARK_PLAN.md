# QuTE Related Work and Benchmark Plan

## Corrected QuTE research idea

QuTE is a neural emulator of quantum-computer behavior. A quantum circuit is given to a learned local backend, and the backend returns QPU-like outputs: samples/counts, estimated measurement values, and eventually state-like summaries when the requested representation is inside the trained envelope.

The target is not EDA optimization and not QAOA-specific optimization. QAOA can be one benchmark workload, but it is not the project goal.

Research claim to test:

> A learned local backend can emulate selected quantum-circuit input/output behavior with bounded memory and low latency, trading exactness for practical local execution when statevector simulation or real QPU queueing is impractical.

Product motivation:

- Avoid `2^n` statevector memory blow-up for repeated circuit evaluation.
- Avoid real-QPU queue/pending delays during development loops.
- Provide a local “similar-to-QPU” execution path for approximate samples/values.
- Keep uncertainty, OOD detection, and fallback explicit instead of pretending this is a universal simulator.

## Related work map

### 1. Dense statevector simulators

Examples: Qiskit Aer statevector, NVIDIA cuStateVec.

- Strength: exact or near-exact reference while memory fits.
- Weakness: stores `2^n` amplitudes, so memory is the first wall.
- Role in QuTE: teacher/reference for small circuits and calibration data.

Benchmark implication: report qubits, precision, peak memory, host/GPU, and whether distributed/offloaded memory was used.

### 2. Tensor-network and MPS simulators

Examples: Qiskit Aer MPS, cuTensorNet, TensorCircuit, circuit-specific TN simulators.

- Strength: can scale far beyond statevector for low-entanglement or favorable structure.
- Weakness: cost depends on entanglement, connectivity, depth, contraction order, bond dimension, truncation error, and path-search overhead.
- Role in QuTE: main classical approximate/scalable simulator baseline.

Benchmark implication: QuTE should compare against TN/MPS on multiple circuit families, not only QAOA. The claim is practical local emulation under bounded resources, not generic superiority.

### 3. Neural quantum states and autoregressive samplers

Examples: autoregressive neural quantum states, Transformer neural states, neural Born-machine-style distribution learners.

- Strength: direct sampling without materializing the full `2^n` state; can amortize training cost over repeated circuit families.
- Weakness: generalization, OOD failure, calibration drift, and tail errors.
- Role in QuTE: closest method family. QuTE packages this as a backend-like circuit-in/counts-out emulator.

Benchmark implication: report median and tail metrics. A backend that looks good on averages can still fail rare circuits.

### 4. Circuit-output surrogate models

Examples: models that predict amplitudes, probabilities, expectation values, or measurement distributions from circuit descriptions.

- Strength: directly aligned with “circuit in, value/distribution out”.
- Weakness: many results are small-circuit, fixed-template, or expectation-only rather than full counts backend.
- Role in QuTE: the main SOTA comparison category for learned circuit emulation.

Benchmark implication: separate tasks:

- count/sample distribution emulation;
- expectation-value prediction;
- state/amplitude approximation;
- noisy-device output emulation.

### 5. QAOA-specific ML accelerators

Examples: GNNs for QAOA parameter prediction, expectation prediction, warm-starting, or solution-quality prediction.

- Strength: useful benchmark workload and adjacent evidence that graph/circuit structure can be learned.
- Weakness: usually solves optimization workflow acceleration, not general QPU-like backend emulation.
- Role in QuTE: optional benchmark family only. Do not frame QuTE as QAOA/EDA optimization.

### 6. Noisy QPU emulation

Examples: hardware-calibration-driven and ML-assisted noisy device emulators.

- Strength: can mimic a particular device/noise regime.
- Weakness: hardware drift and calibration dependence; matching noisy QPU counts does not prove ideal quantum simulation.
- Role in QuTE: external sanity check for QPU-like behavior, not a replacement claim.

Benchmark implication: separate ideal-simulator agreement from real-QPU agreement.

## Benchmark baselines

Minimum baselines:

1. Aer statevector: exact reference until memory/resource guard.
2. Aer MPS or equivalent TN: scalable simulator competitor.
3. QuTE neural backend: local learned emulator.
4. Real QPU: sparse external validation of QPU-like output behavior.

Optional later baselines:

- cuStateVec/cuTensorNet on GPU if available.
- QASMBench workloads for broader circuit coverage.
- Simple autoregressive/Born-machine baseline if a neural ablation is needed.

## Metrics

### Output agreement

- TVD / Hellinger between output count distributions.
- KL only when support smoothing is explicit.
- Fidelity/infidelity for state or probability-vector tasks where exact reference is available.
- Observable error: expectation values for requested measurements.
- Marginal MAE and correlation/ZZ MAE for bitstring samples.
- Calibration/OOD confidence and abstention rate.
- Median and p90/p95 tail metrics.

### Systems

- Warm per-circuit latency.
- End-to-end wall time.
- Samples/second.
- Peak RSS / GPU memory.
- Training data generation cost.
- Training time.
- QPU queue time and job wall time.
- Whether exact simulator calls occur on the neural route.

## Workload axes

Primary axes should cover general circuit emulation:

- Width: small exact-checkable circuits first, then memory-stress widths.
- Depth: shallow to moderately deep.
- Gate families: Clifford, Clifford+T, rotation-heavy, entangling templates.
- Connectivity: line, grid-like, all-to-all/sparse random, hardware-native layouts.
- Output task: counts, selected observables, marginal/correlation summaries.
- Splits: IID, parameter-OOD, topology-OOD, gate-family-OOD, width-OOD.

QAOA/MaxCut remains one useful structured benchmark, not the identity of the project.

External benchmark axis:

- QASMBench subset after the backend contract is stable.

## QPU smoke positioning

QPU smoke should verify:

- A small frozen circuit set can be executed on real QPU and locally through QuTE.
- Raw QPU counts and local counts are preserved.
- Local execution avoids queue wait and keeps memory bounded.
- Output agreement is descriptively reported.

QPU smoke should not claim:

- QPU replacement.
- General universal quantum simulation.
- Statistical significance from a tiny smoke set.
- Ideal-distribution correctness from noisy hardware agreement.

## Next protocol changes before QPU smoke

1. Update QPU smoke circuits so they are not only QAOA-shaped.
2. Include at least: Bell/GHZ-style entanglement, random Clifford/rotation circuit, and one QAOA-like structured circuit as a benchmark case.
3. Report three comparisons separately:
   - QuTE vs ideal simulator.
   - QPU vs ideal simulator.
   - QuTE vs QPU.
4. Add queue/wall-time fields to the QPU job manifest.
5. Decide pass/fail only on artifact completeness and local resource bounds; keep numerical agreement descriptive.

## Source families consulted

- NVIDIA cuQuantum/cuStateVec documentation and benchmark notes.
- Qiskit Aer MPS documentation.
- TensorCircuit/cuTensorNet and tensor-network benchmark discussions.
- QASMBench benchmark suite.
- Neural quantum state, autoregressive Transformer, neural Born-machine, and circuit-output surrogate literature.
- Noisy QPU emulation and sampling benchmark literature, including XEB/HOG/TVD-style metrics.
