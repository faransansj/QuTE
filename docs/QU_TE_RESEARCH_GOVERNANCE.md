# QuTE Research Governance v1.0

## Mandatory proposal gate
Every phase must answer all ten questions in the [proposal template](templates/RESEARCH_PROPOSAL_GATE.md), choose exactly one classification, freeze workload-specific targets, and identify untouched validation/test families before work begins. `REJECTED_DIRECTION_DRIFT` must not start under QuTE.

## Evaluation controls
Maintain validation/test separation, predeclared candidate selection, provenance hashes, justified multi-seed checks, negative findings, and no post-test architecture selection. A consumed sealed set is permanently ineligible as development data. P4.8 sealed access remains completed once (`access_count = 1`); governance verification reads only published reports, verdicts, hashes, and provenance.

## Change control
Architecture or experiment failure does not change the North Star. A change requires a versioned ADR/proposal and substantial evidence that the fundamental product hypothesis fails, such as: three materially different workloads failing call reduction/break-even; compressed/query outputs failing beyond small qubits; OOD+fallback failing to control task error; or costs failing realistic amortization.

Every revision requires an explicit proposal, evidence table, impact analysis, replacement canonical statement, semantic version bump, changelog entry, and new hashes. v1.0 is never silently edited.

## Boundary ownership
Core = execution model; Runtime = API/OOD/routing/fallback/certification; Hardware Twin = QPU response; Benchmark = certification. Feynman and QuDDPM/QCNN are adjacent separate projects unless a proposal concerns only their explicit interface with QuTE.
