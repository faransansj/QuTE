# R1 Operator-Semantic Benchmark — Research Proposal Gate

**Owner:** QuTE research

**Date:** 2026-08-27

**Status:** `APPROVED_PROTOCOL_FROZEN_NOT_RUN`

**Classification:** `ALIGNED_BENCHMARK`

1. **Capability enabled:** measure whether QuTE learns reusable quantum transformation semantics instead of memorizing circuit syntax.
2. **Restricted workload:** initially four-qubit unitary circuits over the existing `H`, `X`, `RX`, `RY`, `RZ`, and `CNOT` alphabet, plus explicitly reviewed equivalent decompositions.
3. **Contract improved:** circuit/context input maps to an operator/action representation, queried state action or measurement statistics, semantic confidence, OOD confidence, fallback decision, and provenance.
4. **Workload value:** identify representations that can reuse one learned transformation across equivalent circuit forms and reduce repeated simulator calls after certification. This proposal makes no latency or call-reduction claim yet.
5. **Output scaling:** exact full-unitary comparison is allowed only as a four-qubit oracle. Candidate-facing and future-scale outputs must support probe actions, observables, samples, or compressed/query representations.
6. **Physical structure:** equivalence up to global phase, linear state action, norm preservation, and architecture-specific constraints such as exact unitarity where applicable. No architecture is mandated.
7. **OOD/fallback:** rewrite-family, cross-decomposition, parameter, depth, and state/probe OOD labels; confidence and fallback metrics are required. A simulator remains the oracle/fallback.
8. **Metrics:** semantic correctness, equivalent-pair consistency, non-equivalent-control rejection, family-level macro results, worst-family results, physical-validity error, OOD/fallback quality, latency, memory, and teacher-call accounting.
9. **Untouched sets:** base-circuit identities, rewrite instances, rewrite families, decomposition families, and probe states must be separated before generation. Final test access is logged and forbidden during development.
10. **Boundary:** R1 is Benchmark work. It evaluates QuTE Core candidates but does not itself select a product workload, implement M1, create a Hardware Twin, or claim QPU replacement.

## Automatic drift warnings

- [x] No larger-model-without-hypothesis proposal
- [x] No arbitrary full-state/full-unitary scale-out
- [x] No test-set-driven architecture change
- [x] Fallback required
- [x] Not fidelity-only
- [x] No language feature as execution core
- [x] No QPU replacement claim
- [x] Learned-runtime purpose retained
- [x] Primary output remains quantum execution semantics

## Evidence provenance

QPU Validation v1 is motivation only: Composition-OOD mean Model–Sim TVD was `0.93427`, while mean Sim–QPU TVD was `0.03148` on the frozen 20-circuit `ibm_pittsburgh` workload. This does not change R1 rewrite rules, thresholds, splits, or candidate selection post hoc.

## Gate decision

`ALIGNED_BENCHMARK`. Protocol decisions are frozen in `artifacts/r1_operator_semantic_benchmark/protocol.json`. This gate authorizes the benchmark direction, not execution: no data generation, model training, final-test access, or QPU execution occurred or is authorized in this protocol-only task.
