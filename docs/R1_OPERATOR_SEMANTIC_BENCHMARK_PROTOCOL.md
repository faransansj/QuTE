# R1 Operator-Semantic Benchmark Protocol

**Version:** v1

**Status:** `FROZEN_NOT_RUN`

**Execution:** not authorized by this protocol-only task

**Classification:** `ALIGNED_BENCHMARK`

## Research question

> Can QuTE learn quantum transformation semantics that remain correct across syntactically different but physically equivalent circuit representations?

R1 tests semantics rather than syntax. It is motivated by QPU Validation v1, where Composition-OOD mean Model–Sim TVD (`0.93427`) greatly exceeded mean Sim–QPU TVD (`0.03148`). That evidence identifies a model/generalization problem in the frozen workload; it neither proves an R1 solution nor authorizes post-hoc protocol changes.

## Scope

The initial benchmark is a four-qubit diagnostic over the existing `H`, `X`, `RX`, `RY`, `RZ`, and `CNOT` alphabet plus human-reviewed equivalent decompositions. Exact full-unitary comparison is allowed only as the small-system oracle. It is not a required model output or a scale strategy.

A candidate interface must support transformation context, operator/action or query-conditioned representation, queried state action or measurement statistics, semantic and OOD confidence, fallback decision, and provenance. The protocol does not freeze a Transformer, recurrent model, Cayley map, or any other architecture.

## Positive equivalence families

Seen during training and development:

1. **Inverse cancellation:** `H H`, `X X`, repeated identical CNOT, and `R_axis(theta) R_axis(-theta)`.
2. **Commuting reorder:** adjacent gates on disjoint supports and reviewed distinct-qubit `RZ` operations.
3. **Rotation fusion/split:** `R_axis(a) R_axis(b) <-> R_axis((a+b) mod 2π)`.

Held out from training and development:

4. **Identity insertion/removal:** `H X X H` and the reviewed disjoint-support CNOT/H identity template.
5. **Cross-decomposition:** `CNOT(c,t) <-> H(c) H(t) CNOT(t,c) H(c) H(t)`.
6. **Nonlocal operator semantics:** the two exact three-CNOT decompositions of SWAP.

Generator implementations must validate every symbolic precondition and pass the independent oracle checks before writing an example.

## Matched negative controls

Every semantic stratum requires non-equivalent controls. Examples include a parameter perturbation outside tolerance, uncompensated control-target reversal, or an operator-changing gate deletion/insertion. Match depth and token count where possible.

These controls prevent a constant-output or syntax-invariant model from appearing semantically correct.

## Oracle contract

Primary oracle: exact four-qubit unitary equivalence up to global phase.

Independent checks:

- deterministic informationally sufficient probe-action agreement;
- predeclared measurement-distribution agreement;
- symbolic verification that each rewrite precondition holds.

Any disagreement among oracle checks rejects the generated pair. Frozen complex128 tolerances are `1e-10` for phase-aligned relative Frobenius error, maximum probe L2 error, and probability TVD; state normalization uses `1e-12`.

## Split design

| Split | Meaning |
|---|---|
| Semantic IID | Unseen base circuits and rewrite instances from families seen during development |
| Rewrite-instance OOD | Unseen templates or parameterizations within seen families |
| Rewrite-family OOD | Entire rewrite families absent from training and development |
| Cross-decomposition OOD | Held-out reviewed decomposition family |
| Parameter-OOD | Held-out parameter regions crossed with semantic pairs |
| Depth-OOD | Held-out base and rewritten depths |
| State/probe-OOD | Held-out probes used to query the same transformations |
| Non-equivalent control | Matched negative pairs for every semantic stratum |

### Leakage rules

- Base-circuit identities are disjoint across train, development, and final test.
- Canonical operator hashes cannot cross partitions except within an explicitly linked positive pair.
- Rewrite instances are partition-disjoint.
- Rewrite-family-OOD and cross-decomposition-OOD families are absent from training and development.
- Probe/state IDs are disjoint where probe OOD is claimed.
- Opened final-test examples never become development data.

## Metrics

### Primary

- semantic action correctness against the ideal oracle;
- consistency between predictions for equivalent circuits;
- rejection of non-equivalent controls.

### Required reporting

- example-level results;
- rewrite-family and split macro results;
- worst-family result;
- parameter and depth strata;
- norm, linearity, and candidate-appropriate structural-validity diagnostics;
- OOD discrimination, risk-coverage, failure recall, accepted-set error, and fallback rate;
- latency, peak memory, repeated-workload teacher calls, and break-even where applicable.

Existing action/state fidelity and Z-basis TVD remain diagnostics where applicable. R1 must not collapse to one global fidelity score.

## Corpus allocation

Every positive pair receives one matched negative.

- Train: 1,024 pairs per seen family/depth over depths 2, 4, and 6; 9,216 positives and 9,216 negatives.
- Development: 256 pairs per seen family/depth; 2,304 positives and 2,304 negatives.
- Final test: 21,504 positives and 21,504 matched negatives across Semantic-IID, rewrite-instance OOD, rewrite-family OOD, cross-decomposition OOD, nonlocal semantics OOD, parameter-OOD, and depth-OOD.
- Base-circuit identities are disjoint across all partitions.
- Train/development parameters reuse the frozen P1 ranges; interpolation and extrapolation use the existing held-out ranges.
- Depth-OOD uses canonical base depths 8 and 10. Expanded rewrite depth is reported separately.

## Candidate and final-test policy

- Candidate design and selection are architecture-agnostic.
- Seeds are `2026`, `2027`, and `2028`.
- Selection uses the family-first macro harmonic mean of semantic action fidelity, equivalent-pair consistency, and non-equivalent-control rejection.
- Within `0.001`, choose lower measured latency, then fewer trainable parameters; if still tied, retain both without test-based selection.
- The current CC-NQE model is a baseline, not a privileged candidate.
- Final-test data receive a hash manifest and access log.
- Development loaders reject final-test artifacts.
- No post-test model or protocol selection is allowed.

## Claim boundaries

If supported, R1 may claim generalization only across its frozen reviewed semantic families, splits, domains, and candidate set. It may compare predeclared representations and quantify OOD/fallback behavior.

R1 cannot establish universal circuit understanding, arbitrary-qubit generalization, QPU replacement, or hardware-noise learning. Descriptive candidate comparisons are not causal architecture evidence.

## Out of scope for this draft

- data generation;
- model training or candidate selection;
- QPU jobs;
- M1 implementation;
- hardware twin or adapter work;
- North Star changes;
- final-test-driven redesign.

## Frozen support thresholds

- Semantic action fidelity macro: at least `0.99`.
- Equivalent-pair consistency macro: at least `0.995`.
- Non-equivalent-control rejection: at least `0.95`.
- Worst-family semantic action fidelity: at least `0.97`.
- Accepted-set semantic failure rate: at most `0.01`.
- Physical-validity error: at most `1e-4`.
- OOD AUROC: at least `0.80`.
- Maximum regression versus the current baseline: `0.01`.

Report paired 95% stratified-bootstrap intervals over base-circuit identities using 10,000 resamples and seed `47011`. These are benchmark support rules, not formal significance claims.

## Freeze checklist

- [x] Proposal gate approved
- [x] Rewrite/decomposition catalog approved
- [x] Split and leakage contract approved
- [x] Oracle self-check specification approved
- [x] Sample counts and domains approved
- [x] Candidate/seed/selection policy approved
- [x] Statistical reporting approved
- [x] Final-test access policy approved
- [x] Protocol JSON and Markdown hashes recorded

The protocol is frozen but not run. A separate explicit execution task is required before data generation. No R1 data, training, final-test access, or hardware execution occurred while producing v1.
