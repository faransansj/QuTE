# QuTE Related Work and Benchmark Plan

## QuTE research idea

QuTE is an amortized neural surrogate backend for structured quantum-circuit sampling. The target is not exact universal simulation. The target is a local, bounded-memory, low-latency sampler that approximates useful output statistics for repeated workloads such as QAOA, where exact statevector memory grows as `O(2^n)` and real QPU execution can be blocked by queue/pending latency.

Research claim to test:

> For a restricted QAOA workload family, a trained neural backend can trade some distributional accuracy for much lower local memory pressure and no QPU queue wait, while preserving useful observables such as energy, marginals, and ZZ correlations.

## Related work map

### 1. Dense statevector simulators

Examples: Qiskit Aer statevector, NVIDIA cuStateVec.

- Strength: exact or near-exact reference while memory fits.
- Weakness: stores `2^n` amplitudes, so memory is the first wall.
- Role in QuTE: gold reference for small/medium circuits and teacher generation.

Benchmark implication: always report qubits, precision, peak memory, host/GPU, and whether distributed/offloaded memory was used.

### 2. Tensor-network and MPS simulators

Examples: Qiskit Aer MPS, cuTensorNet, TensorCircuit, QAOA-specific MPS work.

- Strength: can scale far beyond statevector for low-entanglement or favorable circuit structure.
- Weakness: cost depends on entanglement, graph connectivity, depth, contraction order, bond dimension, truncation error, and path-search overhead.
- Role in QuTE: main classical competitor in the memory-limited regime.

Benchmark implication: QuTE should not claim generic advantage over tensor networks. It should compare on the actual target region: random graph QAOA cells where MPS/TN latency or memory becomes impractical under fixed accuracy/resource guards.

### 3. Neural quantum states and autoregressive samplers

Examples: autoregressive neural quantum states, Transformer-style neural state models, Born-machine-style distribution learners.

- Strength: direct sampling without materializing the full `2^n` state; can amortize training cost over repeated workloads.
- Weakness: restricted generalization, OOD failure, tail errors, and heterogeneous benchmark practice.
- Role in QuTE: closest method family. QuTE is specifically a backend-style sampler, not just a wavefunction ansatz or optimizer helper.

Benchmark implication: report both median and tail metrics. QuTE v13 already showed why: median can pass while `p90_zz_mae` fails.

### 4. QAOA-specific ML accelerators

Examples: GNNs for QAOA parameter prediction, expectation prediction, warm-starting, or solution-quality prediction.

- Strength: reduces optimizer/evaluation cost.
- Weakness: usually does not produce full samples/counts as a backend.
- Role in QuTE: adjacent but not equivalent. QuTE should distinguish “sampling backend” from “parameter optimizer”.

Benchmark implication: include QAOA observables, but keep the primary artifact as counts/samples plus backend metadata.

### 5. Noisy QPU emulation

Examples: hardware-calibration-driven and ML-assisted noisy device emulators.

- Strength: can mimic a particular device/noise regime.
- Weakness: hardware drift and calibration dependence; matching a noisy QPU does not prove matching the ideal quantum distribution.
- Role in QuTE: QPU smoke is external sanity evidence, not a replacement claim.

Benchmark implication: separate three distances: local-vs-ideal simulator, QPU-vs-ideal simulator, and local-vs-QPU.

## Benchmark baselines

Minimum baselines:

1. Aer statevector: exact reference until memory/resource guard.
2. Aer MPS or equivalent TN: scalable classical competitor.
3. QuTE neural backend: local amortized sampler.
4. Real QPU: sparse external validation only.

Optional later baselines:

- cuStateVec/cuTensorNet on GPU if available.
- QASMBench workloads for broader non-QAOA coverage.
- Simple QCBM/autoregressive baseline if the paper needs a neural ablation.

## Metrics

### Accuracy / utility

- TVD between counts distributions.
- Hellinger distance where useful.
- Energy error per edge for MaxCut/QAOA.
- Marginal MAE.
- ZZ/correlation MAE.
- Low-energy or optimum-solution probability.
- Width/depth/OOD breakdown.
- Median and p90/p95 tail metrics.

### Systems

- Warm per-circuit latency.
- End-to-end wall time.
- Samples/second.
- Peak RSS / GPU memory.
- Training data generation cost.
- Training time.
- QPU queue time and job wall time.
- Whether exact calls or per-circuit optimizer steps occur on the neural route.

## Workload axes

Primary QuTE axis:

- QAOA MaxCut.
- `cycle` and `random_3_regular` graphs.
- Depth `p=1..3` first.
- Widths inside exact validation boundary, then beyond statevector memory guard for systems-only tests.
- IID, parameter-OOD, graph-seed-OOD, and width-OOD splits.

External benchmark axis:

- QASMBench subset later, only after QAOA claim is clean.

## QPU smoke positioning

QPU smoke should verify:

- QPU job can be completed and raw counts preserved.
- The same workloads run locally through QuTE v13.
- Local execution avoids queue wait and keeps memory bounded.
- Local-vs-QPU observables are descriptively reported.

QPU smoke should not claim:

- QPU replacement.
- General quantum simulation.
- Statistical significance from four circuits.
- Ideal-distribution correctness from noisy hardware agreement.

## Next protocol changes before QPU smoke

1. Add this related-work framing to the QPU smoke report.
2. In QPU smoke, report three comparisons separately:
   - QuTE vs Aer statevector.
   - QPU vs Aer statevector.
   - QuTE vs QPU.
3. Add queue/wall-time fields to the QPU job manifest.
4. Keep the first QPU batch small: four circuits, 1024 shots.
5. Decide pass/fail only on artifact completeness and local resource bounds; keep numerical agreement descriptive.

## Source families consulted

- NVIDIA cuQuantum/cuStateVec documentation and benchmark notes.
- Qiskit Aer MPS documentation.
- TensorCircuit/cuTensorNet and QAOA tensor-network benchmark discussions.
- QASMBench benchmark suite.
- Neural quantum state, autoregressive Transformer, and Born-machine literature.
- Noisy QPU emulation and sampling benchmark literature, including XEB/HOG/TVD-style metrics.
