# Literature and Provisional Gap Matrix

**Search date:** 2026-08-31
**Coverage target:** 2018–present
**Status:** structured search, not a completed systematic review

## Search method and claim boundary

Searches combined exact-title queries and topic queries for neural quantum-state circuit simulation, autoregressive wavefunctions, generative circuit-output simulation, amortized surrogates, learned tensor-network control, observable prediction, simulator selection, and Qiskit backends. Primary papers and official Qiskit documentation were checked where accessible.

No row below proves novelty. The provisional gap remains open until a reproducible systematic search records databases, full query strings, inclusion/exclusion rules, deduplication, and forward/backward citation screening. Therefore this project must not use “first,” “previously absent,” or equivalent claims.

Legend: **yes**, **no**, **partial**, **not reported (NR)**. “Per-circuit training” means optimization or fitting performed while simulating a new circuit trajectory, not one offline training run reused across unseen circuits.

## Matrix

| Work | Year | Circuit family / teacher | Per-circuit training | Input-state support | Output | Sampling / amplitude | Multi-query reuse | Scale generalization | Runtime / memory / total training cost | Backend / abstention | Direct overlap with QuTE |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| Jónsson, Bauer, Carleo, *Neural-network states for the classical simulation of quantum computing* [[arXiv](https://arxiv.org/abs/1808.05232)] | 2018 | Hadamard/Fourier-style transforms; local gate rules | **yes** for approximate non-diagonal gate updates; iterative SGD | neural wavefunction | neural state | amplitude representation; Born sampling via the NQS machinery | yes after fitting that trajectory | demonstrations beyond brute-force width for selected circuits; no unseen-circuit amortization study | no QuTE-style teacher/training break-even accounting | no / no | High on neural-state simulation; differs on per-circuit optimization and deployment contract |
| Carrasquilla et al., *Probabilistic simulation of quantum circuits using a deep-learning architecture* [[DOI](https://doi.org/10.1103/PhysRevA.104.032610)] | 2021 | GHZ, graph states to 60 qubits; 6-qubit TFIM VQE; local quasi-stochastic gate updates | **yes**; Transformer distribution updated/optimized after each gate | state represented through factorized generalized measurements | learned POVM probability representation and observables | autoregressive sampling; not a pretrained circuit-conditioned computational-basis backend | yes after trajectory fitting | width scaling demonstrated on selected families | accuracy studies; no end-to-end amortized unseen-circuit break-even | no / no | High; important per-circuit probabilistic-simulator baseline |
| Cantori, Vitali, Pilati, *Supervised learning of random quantum circuits via scalable neural networks* [[DOI](https://doi.org/10.1088/2058-9565/acc4e2)] | 2023 | classically simulated random circuits; universal or rotation gate sets | no; model reused across circuits | fixed preparation implicit in corpus | selected one-/two-qubit expectation values | no reusable counts; no amplitude API | limited to trained target observables | larger-circuit extrapolation studied | model accuracy/scaling emphasized; no complete backend cost ledger | no / no | High on amortization, low on generative output |
| Cantori and Pilati, *Challenges and opportunities in the supervised learning of quantum circuit outputs* [[arXiv](https://arxiv.org/abs/2402.04992)] | 2024 preprint; later publication | layered CNOT plus random one-qubit rotations, variational-style circuits | no | fixed circuit preparation | selected expectation values | no / no | target-specific | width/depth extrapolation; hard regime with inter-layer angle variation | reports supervised-learning cost growth; not a counts backend benchmark | no / no | High on boundary analysis; scalar-output limitation remains |
| Cantori et al., *Synergy between noisy quantum computers and scalable classical deep learning* [[arXiv](https://arxiv.org/abs/2404.07802)] | 2024 | Ising/Trotter circuits; simulated noisy expectations plus circuit descriptors | no | fixed preparation | ideal expectation estimates / error mitigation | no reusable counts / no amplitude | target-specific | trained at 6–10 qubits, tested to 16 in reported configuration | no standalone ideal-backend break-even | no / no | Adjacent; noisy-measurement-assisted mitigation has a different objective |
| Lange et al., *From architectures to applications: a review of neural quantum states* [[arXiv](https://arxiv.org/abs/2402.09402)] | 2024 | review of NQS architectures and applications | varies | varies | wavefunctions, observables, samples | varies | varies | reviews expressivity, optimization, and scaling limits | review, not a backend benchmark | no / no | Establishes that NQS and autoregressive state representations are mature prior art |
| Nietner et al., *On the average-case complexity of learning output distributions of quantum circuits* [[arXiv](https://arxiv.org/abs/2305.05765)] | 2023 preprint; 2025 journal | brickwork random-circuit distributions | n/a | fixed preparation | output distribution learning task | distribution learning | n/a | hardness for sufficiently deep random circuits in the statistical-query model | theoretical lower-bound context | no / no | Motivates restriction to structured families; rules out broad empirical generalization claims |
| Wang and Fu, *Simulating quantum circuits with a neural statebank* [[arXiv](https://arxiv.org/abs/2606.08707)] | 2026 | HCZCH, QAOA, Clifford+T, Clifford | **yes**; one checkpoint trained per layer from the previous checkpoint | autoregressive complex wavefunction | amplitude queries and independent Born samples | **yes / yes** | yes after fitting that circuit trajectory | reported to 34 qubits on main benchmark and wider selected tests; depth error accumulation analyzed | compares end-to-end representation compute and memory; not train-once/unseen-circuit amortization | no Qiskit backend / no OOD fallback | Closest rich-output simulator; primary distinction to test is cross-circuit amortization |
| TensorCircuit, *A Quantum Software Framework for the NISQ Era* [[Quantum](https://quantum-journal.org/papers/q-2023-02-02-912/)] | 2023 | general tensor-network/statevector simulation framework | no learned per-circuit fitting required | broad circuit/state support | exact/approximate simulator outputs | simulator dependent | yes | structure/hardware dependent | strong systems baseline; not a learned output surrogate | framework integration / no learned abstention | Baseline and integration prior art, not the same learned model |
| Qiskit `BackendV2` and custom backend guide [[API](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.providers.BackendV2)] [[guide](https://quantum.cloud.ibm.com/docs/guides/custom-backend)] | current docs checked at Qiskit 2.5 | execution interface, not a simulation method | n/a | `QuantumCircuit` input | `Job`/result contract | backend-defined | yes | n/a | interface only | **yes** / policy left to provider | Required interface prior art; wrapper alone is not a contribution |
| Transformer-GNN full-output repository [[GitHub](https://github.com/Prof-it/trans-gnn-QuantumCircuitPrediction)] | current repository, publication status not established | 2–5 qubit noisy/noiseless datasets; reported extrapolation to 6 | apparently amortized across corpus | fixed dataset assumptions | explicit full probability vectors | sampling derivable, but output is exponential | partial | small-width extrapolation | no validated backend cost accounting found | no / no | Warning baseline: full-distribution prediction exists, but violates QuTE's no-`2^n` output rule |

## Provisional taxonomy

### 1. Amortized, task-specific predictors

```text
circuit -> selected observable
```

These reuse one supervised model over many circuits but do not produce reusable measurement samples.

### 2. Rich neural-state simulators with circuit-specific optimization

```text
circuit trajectory -> layer/gate-wise neural fitting -> amplitudes and/or samples
```

These can provide reusable state queries but pay optimization cost on each new trajectory.

### 3. Conventional exact/approximate simulator frameworks

These provide execution interfaces and broad semantics but are not pretrained learned surrogates. Their cost depends on state size, entanglement, contraction width, magic, topology, and hardware.

## Gap to test, not claim

> Does a single pretrained circuit-conditioned autoregressive model support previously unseen circuits from one frozen structured workload family, without per-circuit optimization or an explicit `2^n` inference object, while producing reusable computational-basis samples and a measurable accuracy–runtime–memory–amortization frontier with selective fallback?

The current search found direct partial overlaps, especially Cantori-style amortized predictors and Wang–Fu/Carrasquilla-style generative neural simulators. It did **not** establish that the full QuTE combination is absent. The next literature gate must add citation chaining around the two closest generative simulator papers and searches for “conditional,” “meta-learned,” “amortized,” and “surrogate” circuit execution.

## Positioning sentence under review

Prior work demonstrates amortized prediction of selected circuit observables and circuit-specific neural representations that support richer state queries. QuTE tests a different systems point: train one circuit-conditioned generator once, execute many unseen circuits in a declared workload without circuit-specific optimization, emit reusable counts under an explicit error budget, and route unsupported inputs to conventional simulators.
