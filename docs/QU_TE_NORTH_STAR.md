# QuTE North Star v1.0

## Canonical statements
QuTE는 양자컴퓨터를 물리적으로 복제하는 프로젝트가 아니다.
QuTE는 제한되고 검증된 양자 workload를 물리 제약이 반영된
신경 연산자로 컴파일하여, 고전 AI 하드웨어에서 양자 백엔드처럼
실행하고, 불확실하거나 범위 밖인 요청은 시뮬레이터 또는 QPU로
전달하는 AI-native quantum runtime을 만드는 프로젝트다.

QuTE compiles restricted and verified quantum workloads into
physics-constrained neural execution backends running on classical
AI hardware, while routing uncertain or out-of-domain requests to
classical simulators or physical QPUs.

**Compile quantum workloads into verified neural execution backends.**

## Immutable principles
1. **P1 — Quantum-compatible execution contract.** Accept circuit, parameters, state descriptor, Hamiltonian/evolution, requested observables, and device/noise context where applicable. Return samples, expectations, selected state/channel data, RDMs, gradients, uncertainty, OOD confidence, fallback decision, and provenance as applicable.
2. **P2 — Restricted and verified domain.** Every model declares workload/circuit family, parameter and state domains, depth/complexity range, outputs, measured error bounds, failures, and fallback.
3. **P3 — Operator/channel-first representation.** Prefer context encoder → reusable operator/channel → query/state action. Full states/unitaries are small-qubit diagnostics; scale uses compressed/query forms.
4. **P4 — Physical structure by construction.** Prefer architectural representation of unitarity, norm, linearity, Hermiticity, trace preservation, complete positivity, locality, causality, and valid symmetries. Do not freeze Cayley, recurrence, Transformer, or another current architecture.
5. **P5 — No exponential output requirement at scale.** Use observables, samples, selected amplitudes, correlations, RDMs, PTMs, MPOs, local channels, latent propagators, or query-conditioned outputs.
6. **P6 — OOD detection and fallback are product requirements.** Production is neural fast path + detector + simulator/tensor-network/QPU fallback.
7. **P7 — Workload-level value.** Measure task error, latency, throughput, memory, call reduction, teacher/training cost, break-even, coverage, OOD, fallback, and physical validity—not fidelity alone.
8. **P8 — Honest scientific evaluation.** Preserve validation/test separation, untouched final tests, predeclared selection, hashes, justified multi-seed checks, negative results, and no post-test selection. Opened sealed sets never become development sets.
9. **P9 — Implementation flexibility.** Encoders, architectures, losses, representations, model size, data strategy, objective, hardware, compression, and API may change.
10. **P10 — Project boundaries.** Keep Core, Runtime, Hardware Twin, Feynman, QuDDPM/QCNN, and Benchmark separate. Feynman is not QuTE Core.

## Non-goals
- Efficient exact simulation of arbitrary quantum circuits; claiming a classical neural network is a physical quantum computer.
- Indefinite full-state/full-unitary scale-out; model size as the primary strategy.
- Optimization against an opened final test; neural deployment without OOD/fallback.
- Language explanation as execution core; quantum advantage from four-qubit fitting.
- Hidden teacher/training/fallback cost; universal claims from restricted workloads.

## Evidence ledger
| Phase | Canonical evidence | Source commit and artifacts |
|---|---|---|
| P4.5 | Data scaling materially improved generalization; insufficient-data model scaling overfit; raw parameter scaling was not dominant. The historical report retains an earlier XPU-blocked conclusion; the later committed screening metrics are the source for these empirical comparisons. | `e0dcd05`; `artifacts/cc_nqe_p4_5/screening_results.json`, `REPORT.md` |
| P4.6 | Mixed circuit/probe data effect; operator factorization beat direct state regression; exact-unitary structure beat soft regularization; exact-U supervision alone did not explain all gains. | `f979907`; `artifacts/cc_nqe_p4_6/P4_6_FINAL_REPORT.md`, `scientific_verdict.json` |
| P4.7 | Recurrent encoding supported on three-seed validation; composition self-consistency gained on Composition-OOD with balanced trade-off; privileged prefix supervision unsupported. | `cd77fde`; `artifacts/cc_nqe_p4_7/P4_7_FINAL_REPORT.md`, `scientific_verdict.json` |
| P4.8 | Untouched sealed result: recurrent qualified; composition unsupported; overall partially supported; roles unchanged; no formal significance claim. | `163a4f7`; `artifacts/cc_nqe_p4_8/P4_8_FINAL_REPORT.md`, `scientific_verdict.json`, `sealed_access_log.json` |

Resulting interpretation: explicit operator representation and hard physical structure are the strongest surviving direction; recurrence is useful but misses the strongest sealed criterion; the soft composition loss is not verified; an architecture/loss failure is not North-Star failure.
