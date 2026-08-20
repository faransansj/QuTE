# QuTE Roadmap v1.0

## Short term — 0–12 months
- **ST-1:** publish/archive P4.5–P4.8 around scaling, allocation, operator-first structure, hard constraints, ablations, multi-seed validation, one-time sealed evaluation, and preserved negative findings—not universal replacement.
- **ST-2:** restricted Workload Emulator MVP for one high-reuse VQE/QAOA/QML/control workload: `(C, theta, state descriptor, observable) -> observable/samples/gradients/uncertainty`; no full-state requirement.
- **ST-3:** Benchmark v2 with new IID, State-, Parameter-, Composition-, Depth-, circuit-equivalence-, and cross-decomposition-OOD families; large families and macro reporting.
- **ST-4:** 6–12 qubit compressed/query prototype using observables, RDM, PTM, MPO, local channels, or latent propagators. Qubit count alone is not KPI.
- **ST-5:** parser adapter, quantum-compatible contract, inference, OOD, fallback, provenance, backend-like API.
- **ST-6 targets (not claims):** predeclared workload tolerance; repeated-query speed initially ≥10× teacher latency where feasible; memory reduction; high-recall fallback; reproducible provenance; realistic break-even. Freeze targets before results.

## Medium term — 1–3 years
- **MT-1:** QuTE Base + Workload Adapter + Hardware Adapter + Query Head.
- **MT-2:** compare PTM, MPO, local channels, light cones, and latent propagators without full exponential objects.
- **MT-3:** QPU digital twin for queue-free preview, noise-aware optimization, compiler search, drift monitoring, and QPU-call reduction.
- **MT-4:** confidence/error/output/cost/latency-aware router among QuTE, tensor-network, exact simulator, and QPU.

## Long term — 3–7 years
- **LT-1:** commercial runtime: SDK/API, Qiskit/OpenQASM-compatible backend, compiler, registry, certification, OOD, router, QPU connectors, accounting.
- **LT-2:** workload compiler from customer family/domain/query/accuracy/hardware declaration through teacher generation, training, certification, deployment.
- **LT-3:** primary KPI: **Verified simulator/QPU-call reduction at a fixed task-level error**, plus coverage, fallback, latency, cost, memory, provenance.
- **LT-4:** add Hamiltonian/time, noise/device, richer state/query context only after core execution representation stabilizes.

## Immediate coordinated program
- **R1 Operator-Semantic Benchmark:** rewrite equivalence, commuting reorderings, inverse cancellation, equivalent rotations, cross-decomposition OOD, and new untouched families. Establish semantic correctness.
- **M1 Workload Emulator MVP:** `(parameterized circuit, state descriptor, observable) -> expectation / gradient / uncertainty / fallback`. Establish practical utility.

Neither track replaces the other.
