# R1 Corpus Protocol Compliance Audit

**Date:** 2026-08-27

**Audited branch:** `research/r1-corpus-generator`

**Audited HEAD:** `4a64e83f259ee748cba4e6462ba9ad3ac59161f9`

**Frozen protocol:** `qute-r1-operator-semantic-benchmark-v1`

**Scope:** pre-full-generation compliance review only

**Verdict:** `FAIL_FULL_GENERATION_BLOCKED`

## Executive decision

Full corpus generation must not run from the audited implementation. Equivalence-class partitioning is implemented and passes the smoke audit, but rewrite-template coverage is incomplete and the smoke final records collide exactly with future full final records. The negative controls satisfy the frozen numerical floor but do not establish a hard-negative difficulty contract.

## 1. Frozen rewrite coverage

**Verdict:** `NON_COMPLIANT`

All six frozen family labels appear in the smoke corpus, but family-label coverage is not template/semantic coverage.

| Frozen family | Implemented evidence | Compliance |
|---|---|---|
| Inverse cancellation | `H H`, `X X`, repeated CNOT, `RY(theta) RY(-theta)` | Partial: generic `R_axis` is implemented only for `RY`; `RX` and `RZ` are absent |
| Commuting reorder | disjoint `H` and `RZ`; disjoint CNOT and `H` | Partial: generic disjoint-support case exists; frozen reviewed `RZ`/`RZ` distinct-qubit template is absent |
| Rotation fusion/split | `RZ(a) RZ(b) <-> RZ(a+b)` | Partial: generic `R_axis` is implemented only for `RZ`; `RX` and `RY` are absent |
| Identity insertion/removal | `H X X H` | Partial: frozen disjoint-support `CNOT H H CNOT` identity template is absent |
| Cross decomposition | reverse-direction CNOT decomposition | Pass |
| Nonlocal operator semantics | equivalent three-CNOT SWAP decompositions | Pass |

Smoke evidence contains 11 implementation template IDs across all six labels. It does not cover every frozen template or every axis implied by `R_axis`.

**Required remediation:** implement a frozen template matrix with one machine-checkable row per protocol template and axis, then require 100% matrix coverage in smoke and full-plan preflight.

## 2. Negative distance and hard-negative difficulty

**Verdict:** `FROZEN_FLOOR_PASS_HARD_NEGATIVE_NOT_ESTABLISHED`

### Metric definition

The stored `phase_aligned_relative_frobenius` distance is

\[
d(U,V)=\frac{\lVert U-e^{-i\arg\langle U,V\rangle_F}V\rVert_F}{\lVert U\rVert_F},
\qquad \langle U,V\rangle_F=\operatorname{tr}(U^\dagger V).
\]

It is a global-phase-optimized relative Frobenius distance between equal-size ideal unitaries. For equal-norm unitaries its range is `[0, sqrt(2)]`.

### Smoke result

- Minimum: `0.765366864730179`
- Median: `0.7653668647301796`
- Mean: `0.7653668647301796`
- Maximum: `0.7653668647301798`
- Frozen minimum floor: `0.1`
- Same right-side depth/token count as positive: `27/27`
- Control types: 24 parameter perturbations, 3 gate substitutions

The controls satisfy the frozen requirements of one negative per positive, distance at least `0.1`, and matched token/depth where possible. However, every smoke negative has effectively the same large distance. The protocol has no upper bound, target band, nearest-valid-negative rule, or required control-type mixture. Therefore `0.76537` proves non-equivalence, not hard-negative difficulty.

**Required remediation:** either rename the contract to `matched_non_equivalent_control`, or freeze a hard-negative band/selection rule and required control-type coverage before full generation. Do not infer “hard” from the existing floor.

## 3. Equivalence-class split and independence

**Verdict:** `PASS_WITH_AUDITABILITY_CONCERN`

Smoke positive classes:

| Partition | Positive pairs | Distinct operator hashes | Distinct base IDs |
|---|---:|---:|---:|
| Train | 3 | 3 | 3 |
| Development | 3 | 3 | 3 |
| Final smoke | 21 | 21 | 21 |

Cross-partition operator-hash overlap is zero for every partition pair.

The generator enforces:

- globally unique base-circuit IDs;
- phase-canonicalized operator hashes;
- no positive or negative operator hash crossing partitions;
- linked positive/negative records remaining in the same partition;
- retry on detected class leakage.

Thus the current split is equivalence-class-based, not merely pair-count-based. Each positive pair currently forms one class containing its two equivalent circuit realizations.

Concern: `operator_hash` is a phase canonicalization followed by decimal rounding to 10 places. The smoke audit reports only leakage hashes and pair counts, not an explicit equivalence-class manifest or class-cardinality distribution. This is adequate for smoke but weak evidence for 33,024 full classes.

**Required remediation:** emit `equivalence_class_id`, class membership/counts per partition, and an explicit cross-partition class-intersection report in the full preflight audit.

## 4. Smoke final namespace and future sealed contamination

**Verdict:** `FAIL`

The smoke artifact exposes:

- the `final_test` partition name;
- all held-out final split names;
- held-out family names;
- exact template IDs;
- concrete held-out circuit pairs.

More importantly, smoke and full generation share the same seed inputs. Generation mode/namespace is absent from `_record_pair` seed derivation. Reconstructing the future full `local_index=0` candidates showed:

- Smoke final positive records checked: `21`
- Exact left/right circuit overlap with future full candidates: `21/21`

Therefore running full generation without a code change would place already opened smoke records into the scientific final corpus. `final_test_scientific=false` in the smoke manifest does not remove this contamination.

The frozen protocol already documents held-out family/template concepts, so names alone are not the primary failure. Exact future record reuse is.

**Required remediation:**

1. Add an immutable generation namespace to seed and ID derivation, at minimum `smoke-v1` versus `scientific-full-v1`.
2. Regenerate smoke under the smoke-only namespace.
3. Add a preflight assertion that smoke and planned full base IDs, circuit IDs, operator hashes, probe IDs, and pair IDs have zero intersection.
4. Mark the current `smoke_v1` artifact as retired/non-sealable evidence; never reuse its records in a scientific split.
5. Avoid naming smoke records `final_test`; use a smoke-only coverage namespace while retaining split-intent metadata separately.

## Full-generation gate

`BLOCKED` until all conditions below pass:

- [ ] Frozen rewrite template/axis matrix coverage is 100%.
- [ ] Negative controls are accurately classified as matched controls or a real hard-negative contract is frozen.
- [ ] Explicit equivalence-class manifest and intersection audit pass.
- [ ] Smoke/full generation namespaces are disjoint.
- [ ] Smoke-to-full IDs and operator classes have zero overlap.
- [ ] Replacement smoke evidence passes oracle, checksum, determinism, and leakage audits.

No full corpus generation, scientific final-test generation/access, model training, or QPU execution occurred during this audit.
