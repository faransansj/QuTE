# ADR-0001: Establish QuTE North Star v1.0

- Status: Accepted
- Date: 2026-08-20
- Base commit: `163a4f7`

## Context
P4.5–P4.8 changed implementation evidence without changing the durable product hypothesis. A fixed, verifiable boundary is needed before future work.

## Decision
Adopt anchor `qute-project-anchor` version `1.0.0`, its ten immutable principles, non-goals, R1/M1 program, proposal gate, and change-control policy. Architectures, representations, losses, training methods, and hardware remain mutable hypotheses.

## Consequences
Every phase is classified before execution; OOD/fallback and workload/system metrics are mandatory product concerns; opened tests stay closed to development; direction changes require a new versioned ADR and evidence. This ADR does not alter any prior scientific result.
