# QuTE Scientific Evidence Ledger

This mutable ledger extends the historical evidence table frozen in QuTE North Star v1.0. It does not modify the North Star statement, immutable principles, or R1 protocol.

| Phase | Evidence | Canonical source |
|---|---|---|
| P4.5 | Data scaling materially improved generalization; scaling the model without sufficient data overfit; raw parameter scaling was not the dominant solution. | `e0dcd05`; `artifacts/cc_nqe_p4_5/` |
| P4.6 | Operator-first factorization and exact physical structure were promising; circuit/probe allocation had a mixed effect; exact operator supervision did not explain every gain. | `f979907`; `artifacts/cc_nqe_p4_6/` |
| P4.7 | Recurrent encoding had multi-seed validation evidence; composition consistency showed a Composition-OOD gain with balanced-score trade-offs; privileged prefix supervision was unsupported. | `cd77fde`; `artifacts/cc_nqe_p4_7/` |
| P4.8 | Recurrent evidence qualified on untouched sealed evaluation; current composition consistency was not sealed-supported; candidate roles remained frozen; no formal significance was claimed. | `163a4f7`; `artifacts/cc_nqe_p4_8/` |
| QPU Validation v1 | In the frozen 20-circuit, four-qubit `ibm_pittsburgh` run, mean Sim–QPU TVD was 0.02195 while mean Model–Sim TVD was 0.72347. Hardware discrepancy was small relative to current model approximation error. Verdict: `MODEL_APPROXIMATION_DOMINANT`; QPU role: `SPARSE_EXTERNAL_VALIDATION`; no formal significance claim. | IBM job `da76r2e0ukec7382tgcg`; `artifacts/qute_qpu_validation/20260826T043718Z-ibm_pittsburgh/QPU_VALIDATION_V1_REPORT.md`; `scientific_verdict.json`; `error_budget.json` |

## Current interpretation

> The primary short-term limitation of QuTE is learned quantum transformation accuracy and semantic/OOD generalization rather than the measured hardware discrepancy of the current small QPU validation workload.

Composition-OOD mean Model–Sim TVD (`0.93427`) greatly exceeded mean Sim–QPU TVD (`0.03148`). This supports the motivation for R1 Operator-Semantic Benchmark, but does not demonstrate a solution or authorize post-hoc changes to its design.

The two-qubit-gate (`r = 0.695`) and transpiled-depth (`r = 0.484`) hardware correlations are `DESCRIPTIVE_ONLY` at `n = 20` and are retained as future hypotheses, not causal findings.
