# M1 Amortized QAOA Backend — Research Proposal Gate

**Owner:** QuTE research
**Date:** 2026-08-31
**Status:** `APPROVED_PILOT_ONLY_CONFIRMATORY_BLOCKED`
**Classification:** `ALIGNED_CORE`

1. **Capability enabled:** one pretrained model executes many supported unseen QAOA circuits and returns reusable counts without circuit-specific optimization.
2. **Restricted workload:** ideal noiseless QAOA MaxCut; pilot is six qubits, `p=1`, connected cycle-plus-chord graphs, terminal computational-basis measurement.
3. **Contract improved:** supported circuit plus shots and seed maps to samples, counts, diagonal observables, support metadata, route, and provenance.
4. **Workload value:** replace repeated direct simulation calls only inside a validated envelope; measure complete latency, memory, teacher/training cost, and break-even.
5. **Output scaling:** autoregressive samples; no full state, unitary, or `2^n` probability vector on the inference path.
6. **Physical structure:** normalized autoregressive probability by construction; graph/circuit parameters are explicit; exact quantum semantics are supplied only by training teachers and small-width validation.
7. **OOD/fallback:** deterministic syntax/envelope check in the pilot; calibrated uncertainty, abstention, and Aer fallback are mandatory before confirmatory backend claims.
8. **Metrics:** TVD, Hellinger, task errors, tail/optimum probability, end-to-end latency, throughput, RSS/VRAM, storage, teacher cost, training cost, break-even, risk–coverage, and fallback rate.
9. **Untouched sets:** pilot graph/circuit hashes are development-only and permanently ineligible for confirmatory verdicts. Fresh confirmatory membership and seeds freeze before generation.
10. **Boundary:** core learned execution study. The pilot includes only the smallest backend-style adapter needed to test the core contract; production runtime packaging is a later gate.

## Automatic drift warnings

- [x] no larger model without a workload/generalization hypothesis
- [x] no arbitrary full-state/full-unitary scale-out
- [x] no test-set-driven architecture change
- [x] fallback required before deployment claims
- [x] not fidelity-only
- [x] no language feature as execution core
- [x] no QPU replacement claim
- [x] learned-runtime purpose retained
- [x] primary output is quantum execution counts

## Authorization boundary

Authorized now:

- Phase 0 documents and literature matrix;
- six-qubit feasibility corpus;
- one simple conditional autoregressive model;
- local exact teacher and evaluation at pilot width;
- backend-style `run -> job -> result -> counts` path;
- pilot metrics and report.

Not authorized:

- confirmatory corpus generation or access;
- paper-level claim acceptance;
- post-pilot threshold revision on an existing confirmatory corpus;
- QPU execution;
- universal circuit support;
- production release.

## Gate decision

`ALIGNED_CORE`. The existing R1 Operator-Semantic Benchmark remains separate and frozen. Pilot execution may proceed under [`R1_SCOPE.yaml`](../R1_SCOPE.yaml) and [`BENCHMARK_PROTOCOL.md`](../BENCHMARK_PROTOCOL.md).
