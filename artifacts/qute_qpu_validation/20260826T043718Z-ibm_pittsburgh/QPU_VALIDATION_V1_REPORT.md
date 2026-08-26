# QuTE QPU Validation v1 — Closure Report

## Status

**FROZEN.** This report closes the completed IBM hardware-validation run. It does not change QuTE North Star v1.0, launch a new QPU job, retrain a model, or alter R1 scientific design.

## Frozen execution

| Field | Value |
|---|---|
| IBM Runtime job | `da76r2e0ukec7382tgcg` |
| Backend | `ibm_pittsburgh` v1.0.16 |
| Status | `COMPLETED` |
| Workload | 20 frozen four-qubit circuits |
| Shots | 4096 per circuit; 81,920 total |
| QPU charge time | 23 seconds |
| Raw counts | 20/20 saved |
| Original checksum ledger | PASS |
| Bit-order validation | PASS |
| Formal significance claimed | No |

The charge time comes from IBM Runtime metrics retrieved for the completed job. The pre-closure local `job_manifest.json` records completion but not charge time. All other execution fields above are present in the local canonical artifact set.

## Evidence integrity and provenance

The frozen workload uses the existing CC-NQE P1–P4 circuit-conditioned Transformer checkpoint, seed 11, and its frozen four-qubit dataset. `config.json` records the checkpoint, dataset, payload, selection, and transpiler hashes and seeds. `selected_circuits.json` records the deterministic selection rule and all 20 samples. `circuit_metrics.json` records reconstruction, bit order, transpilation, native gates, and physical layouts. `per_circuit_results.json` preserves model, canonical simulator, transpiled simulator, and QPU distributions plus per-circuit TVDs. `raw_counts/` preserves all 20 IBM count payloads unchanged.

The pre-existing `checksums.sha256` ledger verifies every pre-closure execution artifact and all raw-count files. `artifact_hashes.json` adds a closure-wide SHA-256 inventory without modifying the original checksum ledger or raw counts.

## TVD results

Values are split means; parenthesized values are medians.

| Split | Model–Sim TVD | Sim–QPU TVD | Model–QPU TVD |
|---|---:|---:|---:|
| IID | 0.50573 (0.44342) | 0.01933 (0.01880) | 0.51028 (0.45099) |
| Parameter-OOD | 0.78158 (0.84634) | 0.01343 (0.01343) | 0.77594 (0.83873) |
| Composition-OOD | 0.93427 (0.93181) | 0.03148 (0.03198) | 0.90802 (0.89714) |
| Depth-OOD | 0.67232 (0.67126) | 0.02358 (0.02295) | 0.66139 (0.65116) |
| **Overall** | **0.72347 (0.83111)** | **0.02195 (0.02047)** | **0.71391 (0.82794)** |

The canonical-to-transpiled ideal mean TVD is approximately `2.82e-16`, consistent with measurement-equivalent transpilation in this workload.

## Error budget

**Verdict: `MODEL_APPROXIMATION_DOMINANT`.**

Mean Model–Sim TVD is `0.72347`; mean Sim–QPU TVD is `0.02195`. The observed model discrepancy is approximately 32.95 times the measured hardware discrepancy.

> In this frozen 20-circuit four-qubit validation workload on ibm_pittsburgh, the ideal simulator and physical QPU produced similar output distributions relative to the much larger discrepancy between the current CC-NQE/QuTE model and the ideal simulator. The dominant observed error source is therefore the learned model approximation, not hardware noise, within this limited experiment.

This conclusion is not generalized to other IBM backends, QPU architectures, circuit families, large qubit counts, deep circuits, or other hardware-noise regimes.

## Composition-OOD diagnostic

Composition-OOD has mean Model–Sim TVD `0.93427` and mean Sim–QPU TVD `0.03148`. The severe current-model failure therefore cannot primarily be attributed to the hardware discrepancy measured here.

This is empirical motivation for R1 Operator-Semantic Benchmark work on equivalent-circuit semantics, cross-decomposition OOD, rewrite-family OOD, nonlocal operator semantics, semantic representation, and architecture-level physical structure. It does not show that an R1 solution works, and it does not authorize post-hoc changes to R1 rewrite rules, thresholds, splits, model selection, or protocol.

## Hardware correlations

The observed Pearson correlations are:

- transpiled two-qubit gate count vs Sim–QPU TVD: `r = 0.69478`;
- transpiled depth vs Sim–QPU TVD: `r = 0.48391`.

**Label: `DESCRIPTIVE_ONLY`.** With `n = 20`, these values establish neither formal significance nor causality.

Future hypotheses, not conclusions:

- **H_HW1:** More two-qubit gates may increase physical hardware discrepancy.
- **H_HW2:** Greater transpiled depth may increase physical hardware discrepancy.

They may inform later hardware-adapter features, routing confidence, noise-aware execution, and a predeclared QPU Validation v2. No follow-up hardware experiment is launched by this closure.

## QPU role and teacher pipeline

Current program role: **`QPU_ROLE = SPARSE_EXTERNAL_VALIDATION`**.

Use physical QPUs for simulator-to-hardware sanity checks, hardware realism validation, final external validation of improved models, hardware-noise characterization, later hardware-adapter research, and transfer checks. Do not use large quantities of QPU time as the primary strategy for correcting current model error.

Current teacher pipeline:

```text
Ideal simulator -> large-scale training/development data
Physical QPU    -> sparse validation/calibration
```

After model approximation error becomes sufficiently small, a simulator-trained QuTE model plus a hardware-specific adapter or calibration layer may form a QPU-aware runtime.

## Research priorities

1. **QuTE model representation and semantic generalization:** reusable operators/channels, operator equivalence, cross-decomposition semantics, physical constraints by construction, compressed/query representations, and OOD detection.
2. **R1 Operator-Semantic Benchmark:** test transformation semantics rather than circuit syntax.
3. **M1 Workload Emulator MVP:** test restricted-workload latency, memory, and teacher-call reduction.
4. **QPU Validation v2:** only after a materially improved candidate exists.
5. **Hardware-specific digital twin/noise adapter:** deferred until model error is comparable to hardware discrepancy or a product workload explicitly requires hardware behavior.

## Claim boundaries

Allowed conclusions are limited to this frozen experiment: current model error is much larger than measured Sim–QPU discrepancy; sparse external QPU validation is justified; the simulator remains a reasonable large-scale teacher for this restricted workload; Composition-OOD is primarily a model/generalization problem here; and the hardware correlations are hypotheses.

Not allowed: hardware noise is negligible in general; ideal simulation always predicts QPUs; QPU validation is unnecessary; 20 circuits establish universal behavior; the correlations prove causality; current QuTE approximates the QPU well; or current QuTE can replace a physical QPU.

## QPU Validation v2 entry criteria

No v2 study is scheduled. A future v2 may begin only after an improved candidate is selected before QPU access and shows substantial Model–Sim reduction, R1 semantic improvement, and no severe IID/Parameter/Depth regression. Its protocol must be frozen before submission and use materially more than 20 circuits with predeclared sampling, OOD strata, backend, shots, transpilation, calibration metadata, statistical reporting, and test-access policy. This closure selects no final numerical threshold.

## North Star compatibility

The evidence strengthens but does not redefine: **“Compile quantum workloads into verified neural execution backends.”** QuTE North Star v1.0 remains unchanged. The result changes research priority, not project identity.
