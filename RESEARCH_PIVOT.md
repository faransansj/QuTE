# QuTE Research Pivot

**Branch:** `research/r2-amortized-neural-backend`
**Program:** M1 Workload Emulator MVP
**Study namespace:** `m1_r1_amortized_qaoa`
**Status:** Phase 0 audit complete; pilot-only scope approved
**Date:** 2026-08-31

## Decision

QuTE's primary experimental track moves from general quantum-transformation emulation

```text
(circuit C, input state ψ) -> learned operator/state -> ψ_out
```

to an amortized learned approximate execution backend for declared structured workloads:

```text
QuantumCircuit C
  -> support check
  -> circuit-conditioned autoregressive sampler
  -> computational-basis samples / counts
  -> diagonal observables
  -> abstain or exact fallback when unsupported
```

This is **not** a rewrite of QuTE North Star v1.0. The frozen anchor already requires restricted verified workloads, non-exponential outputs at scale, workload-level cost accounting, OOD detection, and fallback. This pivot makes the existing **M1 Workload Emulator MVP** the primary implementation track. The frozen **R1 Operator-Semantic Benchmark** remains a separate benchmark track and is not renamed or repurposed.

## Why the previous primary direction is insufficient

1. A full statevector or full unitary has exponential output size and cannot be the scalable runtime contract.
2. Arbitrary circuits plus arbitrary input states are too broad for an initial falsifiable systems study.
3. Prior work already includes observable predictors and circuit-specific neural-state simulators.
4. Existing QuTE evidence centers on fidelity and semantic/OOD behavior, not end-to-end latency, memory, teacher cost, training cost, or break-even.
5. A prediction model is not a usable backend unless it has a supported input contract, counts output, provenance, rejection, and fallback.

## New main research question

> Can one pretrained circuit-conditioned generative model execute unseen circuits from a frozen structured workload family, without per-circuit optimization or explicit statevector simulation at inference, while meeting a predeclared error budget and reducing end-to-end runtime and memory after amortizing teacher-data and training cost?

Subquestions cover distribution accuracy, zero-shot per-circuit execution, runtime and memory, break-even count, OOD boundaries, selective reliability, and QAOA task utility.

## R1 workload inside M1

Primary workload: ideal noiseless QAOA MaxCut with computational-basis terminal measurement. The smallest pilot uses six-qubit, depth-`p=1` connected graphs. Width, depth, topology, and parameter OOD axes expand only after the pilot gate.

A small 1D TFIM Trotter control is deferred until the QAOA vertical slice passes. It is not needed to answer the first feasibility question.

## Retained assets

- deterministic generation, manifests, canonical JSON, content hashes, provenance, lifecycle and split-audit patterns from `r1_corpus.py`;
- circuit serialization, parameter handling, structural signatures, PyTorch training utilities, OOD split methodology, and regression tests from `cc_nqe.py` and P4.5–P4.8;
- phase-invariant fidelity and observable diagnostics as secondary small-width checks;
- P4.5 data-scaling evidence, P4.6 operator-first evidence, P4.7 recurrent evidence, and P4.8 sealed positive and negative findings;
- QPU Validation v1 as frozen evidence that current learned-model error dominated measured hardware discrepancy in its limited four-qubit sample.

## Replaced for the new execution track

- 32-real-value full-state output heads;
- explicit 4-qubit/fixed-width model contracts;
- statevector-as-runtime-output APIs;
- fidelity-only success criteria;
- timing that excludes parse, encoding, sample generation, packaging, teacher generation, or training;
- assumptions that one circuit or one cached operator defines the workload.

## Archived, not deleted or reinterpreted

All existing artifacts under `artifacts/cc_nqe_*`, `artifacts/qute_qpu_validation/`, and `artifacts/r1_operator_semantic_benchmark/` remain byte-preserved historical evidence. Their original verdicts and claim boundaries remain authoritative. New artifacts use `artifacts/r2_amortized_backend/` only.

Existing results are repositioned only as:

- architecture baselines;
- OOD diagnostics;
- negative evidence about composition/depth generalization;
- motivation for a narrower supported-family backend.

They are not treated as confirmatory evidence for the new RQ.

## Final target

```text
QuantumCircuit
  -> support / uncertainty estimator
     -> QuTE neural route
     -> Aer / MPS / tensor-network fallback
  -> standard counts/result + route and provenance metadata
```

Success requires all three:

1. acceptable predeclared error;
2. measurable systems advantage under the same output requirement and hardware;
3. a finite declared support envelope with reliable rejection outside it.

A negative result—including no realistic break-even point—is valid.

## Non-goals

- universal or exact simulation of arbitrary circuits;
- full statevector generation at scale;
- arbitrary entangled input-state APIs;
- mid-circuit measurement, reset, control flow, or noise in R1;
- arbitrary-basis tomography;
- QPU replacement or quantum-advantage claims;
- accuracy claims beyond an independently validated regime;
- changing thresholds after confirmatory data access;
- using pilot circuits in a confirmatory verdict.

## Phase boundary

This branch authorizes documentation and the smallest pilot only. Confirmatory corpus generation and confirmatory model selection remain blocked until the Phase 4 protocol freeze records hashes for corpus membership, seeds, metrics, thresholds, statistical tests, baseline versions, hardware, and model-selection rules.
