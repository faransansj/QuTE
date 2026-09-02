# QuTE Migration Map

**Source branch audited:** `research/r1-corpus-generator` at `1cea49d`
**Target branch:** `research/r2-amortized-neural-backend`
**New artifact namespace:** `artifacts/r2_amortized_backend/`

## 1. Audit summary

The repository contains four distinct asset groups:

1. CC-NQE P1–P4.8 four-qubit state/operator experiments;
2. QPU Validation v1 frozen hardware evidence;
3. R1 Operator-Semantic Benchmark protocol, corpus governance, and pilot artifacts;
4. project-wide North Star, roadmap, governance, and evidence ledger.

The requested direction is already compatible with North Star v1.0 and its M1 program. No frozen anchor or historical verdict needs rewriting.

## 2. Reuse directly

| Asset | Reuse | Boundary |
|---|---|---|
| `r1_corpus.py`: `canonical_json_bytes`, `content_sha256`, namespaced IDs, pair/content hashes | canonical manifests and artifact identities | copy/adapt utility behavior; do not import the operator-semantic corpus as QAOA data |
| `r1_corpus.py`: lifecycle, determinism, intersection, and namespace audits | train/pilot/confirmatory isolation | existing R1 namespaces remain untouched |
| `cc_nqe.py`: circuit serialization, IDs, structural signatures | canonical circuit records and leak checks | QAOA needs a new semantic schema |
| P4.5–P4.8 split discipline | IID/parameter/depth/composition/topology OOD design | no old sealed member becomes development data |
| `tests/` provenance and hash regression style | new contract tests | do not weaken old tests |
| benchmark timing conventions | warm-up, repetitions, full-path versus cached timing | new benchmark must add sampling, packaging, teacher/training cost, and isolated memory |
| phase-invariant fidelity and observable utilities | small-width diagnostics | secondary only; not the success criterion |

## 3. Reuse with adaptation

| Existing asset | Required adaptation |
|---|---|
| `Gate`, `generate_circuit`, `encode_circuits` in `cc_nqe.py` | variable width, graph edges, repeated QAOA layers, canonical bit order, and transpiler metadata |
| `CircuitTransformer` / recurrent P4.7 encoder | remove 4-qubit constants and 32-real full-state head; retain only encoder ideas after a baseline comparison |
| `build_dataset` and manifests | teacher bitstrings plus selected statistics; no stored full probability table except small evaluation data |
| State/Parameter/Composition/Depth OOD code | add graph hash, graph-family, degree, topology, width, and parameter-cell axes |
| `benchmark()` | compare complete counts production under equal shots; add CPU/GPU synchronization, isolated RSS/VRAM, and total-cost ledger |
| QPU counts/result handling | standard bitstrings and counts examples | QPU data remain sparse validation, not a pilot teacher by default |

## 4. Replace for M1

| Replace | Reason | Replacement |
|---|---|---|
| `StateOnly`, `FlatMLP`, and 32-real statevector heads | exponential semantic output and fixed 4-qubit shape | conditional autoregressive bit sampler |
| arbitrary input-state tensor in runtime | outside R1 support contract | in-circuit `|0> -> H^n` preparation |
| full-state/action fidelity as primary endpoint | does not establish usable sample execution | TVD plus task metrics plus systems and economic gates |
| single-circuit/cached-unitary timing | not a repeated unseen-circuit backend workload | parse-to-counts cold/warm and batch timing |
| unconditional neural execution | unsafe outside support | support score, abstention, and fallback route |
| old artifact directories for new outputs | risks historical contamination | `artifacts/r2_amortized_backend/<phase>/<run_id>/` |

## 5. Frozen artifacts: read-only

- `artifacts/cc_nqe_p1_p4/`
- `artifacts/cc_nqe_p4_5/`
- `artifacts/cc_nqe_p4_6/`
- `artifacts/cc_nqe_p4_7/`
- `artifacts/cc_nqe_p4_8/`
- `artifacts/cc_nqe_ab_confirmatory/`
- `artifacts/qute_qpu_validation/`
- `artifacts/r1_operator_semantic_benchmark/`
- `docs/R1_OPERATOR_SEMANTIC_BENCHMARK_PROTOCOL.md`
- `docs/R1_CORPUS_PROTOCOL_COMPLIANCE_AUDIT.md`
- `PROJECT_ANCHOR.md`, `docs/QU_TE_NORTH_STAR.md`, and anchor manifests

The original worktree also had untracked Gate 4 v3 files. They were not copied, modified, staged, or interpreted as part of this pivot.

## 6. Minimal new structure

Pilot only:

```text
RESEARCH_PIVOT.md
LITERATURE_GAP_MATRIX.md
R1_SCOPE.yaml
BENCHMARK_PROTOCOL.md
BACKEND_CONTRACT.md
MIGRATION_MAP.md
docs/M1_AMORTIZED_BACKEND_RESEARCH_PROPOSAL_GATE.md
qute_r2_pilot.py
run_qute_r2_pilot.py
tests/test_qute_r2_pilot.py
artifacts/r2_amortized_backend/pilot_v1/   # generated
```

Do not scaffold a larger package before the pilot passes.

If it passes, the smallest likely package is:

```text
qute/
  circuits.py      # canonical supported circuit record
  model.py         # circuit-conditioned autoregressive sampler
  backend.py       # run/job/result and Qiskit adapter
  teacher.py       # development-only exact/Aer path
  benchmark.py     # accuracy, systems, total-cost accounting
```

`teacher.py` must not be imported by the deployable backend package.

## 7. Branch and experiment naming

- branch: `research/r2-amortized-neural-backend`;
- study namespace: `m1_r1_amortized_qaoa`;
- pilot: `artifacts/r2_amortized_backend/pilot_v1/`;
- later development runs: `.../development/<protocol_hash>/<run_id>/`;
- confirmatory: `.../confirmatory/<frozen_protocol_hash>/`;
- model names: `qute-qaoa-r1-<protocol8>-<model8>`.

A pilot namespace can never be promoted to confirmatory. Content hashes remain namespace-independent so overlap audits can detect relabeling.

## 8. Migration sequence

1. freeze these Phase 0 documents and proposal gate;
2. implement the six-qubit vertical slice with a single model and zero per-circuit training;
3. record exact-path exclusion, counts, accuracy, latency, memory, and failures;
4. profile Aer statevector/MPS and choose the practical training envelope;
5. add OOD calibration only after the IID sampler is credible;
6. freeze fresh confirmatory data and success criteria;
7. run confirmatory benchmark once;
8. package full Qiskit `BackendV2` only if the model-system gate is supported.

## 9. Explicitly skipped

- porting all historical model classes;
- creating a generic circuit IR before one workload works;
- adding diffusion/flow models in the pilot;
- implementing every baseline before the vertical slice;
- changing North Star v1.0.

Add these only when a measured pilot limitation requires them.
