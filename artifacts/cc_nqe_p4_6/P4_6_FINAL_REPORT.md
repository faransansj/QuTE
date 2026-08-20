# CC-NQE P4.6 — Scientific Closure

## Scope and verdict

P4.6 asked why CC-NQE generalizes poorly to unseen circuit compositions and greater circuit depths. This archive closes a **seed-2026 screening milestone**, not a multi-seed confirmation or final test evaluation.

**Verdict: `P4.6-OPERATOR-INDUCTIVE-BIAS-SUPPORTED`.** Within the frozen four-qubit circuit distribution, OOD behavior is strongly affected by data allocation and operator-level inductive structure. Explicit exact-unitary operator learning is a promising architecture direction for CC-NQE.

All reported evaluations are validation-only. Sealed-test access remained **0**.

## Artifact audit

Track A: A1–A5 are 5/5 `COMPLETED`; each used seed 2026, 10,000 optimizer updates, and 10,240,000 pair exposures. Circuit nesting, probe nesting, and distribution controls pass. The canonical winner is A3 and the verdict is `MIXED-DATA-EFFECT`.

Track B: B0–B5 are 6/6 `COMPLETED` scientific runs; each used seed 2026, 10,000 updates, and 10,240,000 samples. Parameter counts and supervision metadata match the frozen protocol: B0 has 1,001,472 parameters; B1–B5 have 1,073,312. B0–B3 use action-only `(C, psi_in, psi_out)` supervision; B4/B5 additionally use privileged `exact_U`, and B5 adds composition consistency. B3 is the action-only exact-unitary basic-Cayley variant. The runtime status/protocol-summary files retain pre-closure mutable/stale labels, so completion is established from the canonical per-run metric JSON files rather than those runtime files.

Historical `P4_6A_REPORT.md` and `P4_6B_REPORT.md` were not rewritten.

## Track A — circuit coverage versus probes per circuit

At approximately fixed 1M-pair budget, neither maximizing unique circuit coverage nor maximizing probes per circuit alone was optimal. A3—62,500 circuits × 16 probes/circuit—was the validation-selected screening winner.

> Within the frozen four-qubit distribution and seed-2026 screening, useful generalization required a balance between circuit coverage and repeated state-action probing.

This does not establish that 16 probes are theoretically sufficient, that each sampled circuit is a distinct mathematical unitary, or that A3 is universally optimal.

## Track B — canonical comparison

The table reports each run's **primary best-balanced checkpoint**, not its final checkpoint. Full primary/final separation and final-only operator diagnostics are in `screening_summary.json`.

| Run | Parameters | Supervision/model | Best balanced | Step | IID | State OOD | Parameter OOD | Composition OOD | Depth OOD |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 1,001,472 | action-only direct state | 0.3159 | 3,500 | 0.5357 | 0.3243 | 0.5542 | 0.1742 | 0.2377 |
| B1 | 1,073,312 | action-only unconstrained operator | 0.4755 | 7,500 | 0.6768 | 0.6338 | 0.6572 | 0.3199 | 0.4298 |
| B2 | 1,073,312 | action-only soft-unitarity operator | 0.4695 | 5,000 | 0.6569 | 0.5872 | 0.6981 | 0.3076 | 0.4439 |
| B3 | 1,073,312 | action-only exact-unitary Cayley | 0.5904 | 8,500 | 0.7428 | 0.6464 | 0.6352 | 0.4930 | 0.5355 |
| B4 | 1,073,312 | privileged exact-U operator | 0.4801 | 8,000 | 0.7228 | 0.6770 | 0.6980 | 0.3126 | 0.4050 |
| B5 | 1,073,312 | privileged exact-U + composition consistency | 0.5335 | 7,500 | 0.7556 | 0.6656 | 0.5702 | 0.3676 | 0.4771 |

The metric JSON files expose operator diagnostics under `latest_validation`; therefore those values are explicitly archived as **final-checkpoint diagnostics** and are not silently substituted for primary-checkpoint diagnostics.

### Required contrasts

- **B0 → B1:** explicit operator factorization substantially improves Composition-OOD (0.1742→0.3199) and Depth-OOD (0.2377→0.4298).
- **B1 → B2:** soft unitarity sharply lowers final unitarity error (for example IID 0.5320→0.1853), but does not improve balanced or Composition-OOD performance; Depth-OOD is only slightly higher at the selected checkpoint.
- **B1/B2 → B3:** structural exact unitarity gives the strongest supervision-matched result, including Composition-OOD 0.4930 and Depth-OOD 0.5355. Final B3 unitarity errors are approximately 2.3–2.7e-7.
- **B3 vs B4:** privileged exact-U supervision does not outperform B3 overall. B4 is not supervision-matched to B3.
- **B4 → B5:** composition consistency improves Composition-OOD (0.3126→0.3676) and Depth-OOD (0.4050→0.4771). B4/B5 are not supervision-matched to B0–B3.

## Screening interpretation

The canonical metrics support the following one-seed interpretation:

1. Direct joint regression `(C, psi) -> psi_out` is not the strongest representation tested.
2. Factoring prediction as `C -> U_hat_C`, then `U_hat_C psi -> psi_out`, supplies a strong inductive bias.
3. Softly encouraging unitarity is not equivalent to structurally constraining predictions to the unitary manifold.
4. Exact-unitary parameterization is the strongest supervision-matched Track-B evidence.
5. Privileged exact-U supervision alone does not explain the gain.
6. Explicit operator-composition consistency supplies an additional positive Composition-OOD signal.

These are seed-2026 screening results, not multi-seed confirmation. Exact-unitary heads do not solve Composition-OOD, and B3 superiority is not established as seed-robust.

## Limitations and unresolved questions

- Four qubits only, with a restricted circuit distribution.
- Simulator-generated noiseless targets; no hardware/QPU data.
- One scientific seed and no final sealed-test evaluation.
- Explicit U(16) representation scales poorly with qubit count; no efficient large-qubit scaling claim is made.
- Composition/Depth OOD remain substantially below perfect fidelity.
- Basic Cayley parameterization has chart limitations, including the −1-eigenvalue boundary.
- Track C recursive/prefix architectures were not evaluated.
- It remains unresolved whether the observed ranking is multi-seed robust or transfers beyond this distribution.
- The study does not establish that privileged operator supervision is generally unnecessary.

## Future work — P4.7 Compositional Architecture

Archive, without execution, the original Track-C hypotheses: C1 monolithic state model; C2 shared gate-wise recurrent state transition; C3 recurrent plus prefix supervision. Track B motivates an additional future hypothesis: combine an exact-unitary operator representation with a recursive/compositional circuit encoder.

P4.7 should ask whether shared recursive processing improves Depth-OOD, whether prefix supervision improves gate-level semantics, whether recursive encoding combines effectively with B3, and whether operator-composition bias can improve B3 further. Multi-seed confirmation and sealed tests remain separate later gates.
