# QuTE Project Anchor v1.0

## North Star
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
Quantum-compatible contracts; restricted verified domains; operator/channel-first representations; physical structure by construction; no exponential output requirement at scale; mandatory OOD/fallback; workload-level value; honest evaluation; implementation flexibility; and explicit project boundaries. See [North Star](docs/QU_TE_NORTH_STAR.md).

## Current evidence
P4.5 supports data over raw model scaling; P4.6 supports operator-first and hard physical structure; P4.7 supports recurrent validation evidence and a composition-specific trade-off; P4.8 is **P4.8-SEALED-PARTIALLY-SUPPORTED**, with **SEALED-RECURRENT-QUALIFIED** and **SEALED-COMPOSITION-NOT-SUPPORTED**. Candidate roles remained frozen; no formal significance is claimed.

## Immediate programs
- **R1 Operator-Semantic Benchmark:** establish semantic correctness across equivalent circuit forms.
- **M1 Workload Emulator MVP:** establish practical utility for one restricted repeated workload.
Neither track replaces the other.

## Non-goals
No arbitrary exact quantum-simulation claim, physical-QPU identity claim, indefinite full-state/unitary scaling, opened-test optimization, neural-only deployment, four-qubit quantum-advantage claim, hidden costs, or universalization of restricted results.

## Decision gate
Every new phase must complete the [Research Proposal Gate](docs/templates/RESEARCH_PROPOSAL_GATE.md). `REJECTED_DIRECTION_DRIFT` work must not start under QuTE.

## Canonical documents
[North Star](docs/QU_TE_NORTH_STAR.md) · [Roadmap](docs/QU_TE_ROADMAP.md) · [Research Governance](docs/QU_TE_RESEARCH_GOVERNANCE.md) · [ADR-0001](docs/ADR/0001-qute-north-star.md)

Anchor version: **1.0.0**. Hash manifest: [`governance/qute_anchor_manifest.json`](governance/qute_anchor_manifest.json).
