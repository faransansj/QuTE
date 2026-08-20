# QuTE

QuTE (Quantum Transformation Emulator) explores reusable neural representations of quantum transformations.

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

P4.6 screened why generalization degrades on unseen circuit compositions and greater depths. Track A found a mixed data-allocation effect, selecting 62,500 circuits × 16 probes/circuit at seed 2026. Track B found explicit exact-unitary operator learning (B3) to be the strongest supervision-matched candidate. These are screening-only, one-seed results; sealed tests remain untouched. The recommended next phase is P4.7 Compositional Architecture, not yet started.

See the [P4.6 final report](artifacts/cc_nqe_p4_6/P4_6_FINAL_REPORT.md).
