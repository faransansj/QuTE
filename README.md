# QuTE

QuTE (Quantum Transformation Emulator) explores reusable neural representations of quantum transformations.

## Project direction

QuTE compiles restricted, verified quantum workloads into physics-constrained neural execution backends on classical AI hardware, with uncertainty/OOD detection and simulator or QPU fallback. It is not an arbitrary quantum simulator or a physical quantum computer.

**Compile quantum workloads into verified neural execution backends.**

Start with the [Project Anchor](PROJECT_ANCHOR.md) and [full North Star](docs/QU_TE_NORTH_STAR.md). Current sealed status is **P4.8 partially supported**: recurrent evidence is qualified, composition consistency is not supported, candidate roles remained frozen, and no formal significance is claimed. Immediate coordinated programs are **R1 Operator-Semantic Benchmark** and **M1 Workload Emulator MVP**. Every new phase must pass the [Research Proposal Gate](docs/templates/RESEARCH_PROPOSAL_GATE.md).

## R1 operator-semantic benchmark

R1 v1 evaluates representation consistency across syntactically different but equivalent four-qubit circuits and basic discrimination against clearly non-equivalent matched controls. **R1 v1 does not certify fine-grained near-operator discrimination.** A hard-negative benchmark requires a separate protocol version.

The execution order is capacity → cross-process reproducibility → pilot → evaluation/power freeze → full planner → scientific-development corpus. Pilot, scientific-development, and sealed payload generation require separate authorization.

Four canonical preflight entry points live under `artifacts/r1_operator_semantic_benchmark/preflight/`:

- `capacity_report.json`
- `reproducibility_report.json`
- `pilot_viability_report.json`
- `preflight_evidence.json`

Detailed coordinate ledgers remain subordinate to those reports.

## CC-NQE P1–P4 controlled feasibility experiment

This repository currently contains a CPU-scale, 4-qubit experiment only. It does **not** claim arbitrary quantum simulation or asymptotic advantage.

```bash
uv sync --dev
uv run pytest
uv run python run_experiment.py
```

The run writes machine-readable provenance/results and the derived 16-section report to `artifacts/cc_nqe_p1_p4/`.

## CC-NQE P4.5 scaling study

P1–P4 artifacts are frozen. P4.5 requires a native PyTorch Intel XPU runtime and never silently falls back to CPU for training.

```bash
uv run python run_p4_5.py preflight
uv run python run_p4_5.py status
uv run python run_p4_5.py run-all
```

New artifacts are isolated under `artifacts/cc_nqe_p4_5/`. After an infrastructure block, rerun `run-all` once native XPU support is available.

## CC-NQE P4.6 compositional OOD study

P4.6 screened why generalization degrades on unseen circuit compositions and greater depths. Track A found a mixed data-allocation effect, selecting 62,500 circuits × 16 probes/circuit at seed 2026. Track B found explicit exact-unitary operator learning (B3) to be the strongest supervision-matched candidate. These are screening-only, one-seed results; P4.6 did not access sealed tests.

See the [P4.6 final report](artifacts/cc_nqe_p4_6/P4_6_FINAL_REPORT.md).

## CC-NQE P4.7 compositional architecture study

P4.7 tested recurrent circuit encoding, composition consistency, and privileged prefix supervision. Three-seed validation confirmation selected C1 as general-purpose; C2 showed a composition-specific gain with a balanced-score trade-off; privileged C3 was archived as negative.

See the [P4.7 final report](artifacts/cc_nqe_p4_7/P4_7_FINAL_REPORT.md).

## CC-NQE P4.8 sealed evaluation

The one-time sealed transaction is complete (`access_count = 1`). Overall: **P4.8-SEALED-PARTIALLY-SUPPORTED**; recurrent: **SEALED-RECURRENT-QUALIFIED**; composition: **SEALED-COMPOSITION-NOT-SUPPORTED**. Candidate roles were unchanged and no formal significance is claimed.

See the [P4.8 final report](artifacts/cc_nqe_p4_8/P4_8_FINAL_REPORT.md).
