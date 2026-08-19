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
