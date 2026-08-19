# CC-NQE P4.5 Scale-Up and Operator Learnability Report

## 1. Starting repository state
- Branch: `research/cc-nqe-p4-5-scaling`; starting HEAD: `2a2c97d0a1e64d4642f6e3abf97a5d87c0b2a289`; inherited P1–P4 working tree was uncommitted.
- Baseline integrity: **PASS**, 30/30 hashes verified; frozen namespace `artifacts/cc_nqe_p1_p4` was not modified.

## 2. Environment and XPU validation
- CPU: AMD Ryzen 5 5600X 6-Core Processor; RAM: 16665006080 bytes; OS: Linux-7.1.8-1-cachyos-x86_64-with-glibc2.44; Python: 3.12.13; PyTorch: 2.13.0+cu130.
- OS-visible graphics: `['0a:00.0 VGA compatible controller: Intel Corporation Battlemage G21 [Arc B580]']`. XPU available/count: False/0; PyTorch device names: `[]`; FP32 preflight: **XPU-BLOCKED**.
- Forward/backward/optimizer/parity/device-residency results could not run because native PyTorch exposes no XPU device. No CPU fallback was used.

## 3. Dataset construction
- Not run: G1 precedes large dataset generation in `run-all`; no scale data were generated after XPU-BLOCKED. The implemented generator uses deduplicated circuit/state tables, one unitary per circuit, memory-mapped sharded pairs/targets, fixed evaluation sets, and prefix-nested manifests.

## 4. Dataset audit
- Not run on a P4.5 master dataset. Audit code covers counts, normalization, unitary error, duplicates, circuit/structure/parameter/composition/depth leakage, nested subsets, and evaluation immutability.

## 5. Training protocol
- Frozen proposed recipe: `{"confirmatory_seeds": [2026, 2027, 2028], "dtype": "float32", "effective_batch_size": 1024, "learning_rate": 0.0003, "maximum_updates": 10000, "optimizer": "AdamW", "scheduler": "cosine", "screening_seed": 2026, "stopping_rule": "fixed optimizer updates", "validation_interval": 500}`. It was not executed.

## 6. Model configurations
- 60k: 58,256 state-model parameters; `{'width': 48, 'ff': 96, 'layers': 2, 'heads': 4}`; FP32 on XPU required.
- 250k: 247,136 state-model parameters; `{'width': 88, 'ff': 176, 'layers': 3, 'heads': 4}`; FP32 on XPU required.
- 1m: 1,001,472 state-model parameters; `{'width': 160, 'ff': 320, 'layers': 4, 'heads': 8}`; FP32 on XPU required.
- 5m: 5,257,088 state-model parameters; `{'width': 336, 'ff': 672, 'layers': 5, 'heads': 8}`; FP32 on XPU required.

## 7. Screening results
- All seven grid points: **NOT RUN — XPU-BLOCKED**.

## 8. Confirmatory results
- No confirmatory seeds ran.

## 9. State-transformation scaling
- No data-scaling or model-scaling inference is possible.

## 10. Operator learning
- No operator training ran. Shared circuit encoder, Hilbert–Schmidt fidelity, phase-aligned matrix error, and raw unitarity metrics are implemented and tested.

## 11. Failure localization
- Not measured; direct-state and predicted-operator application cannot be compared without valid trained checkpoints.

## 12. Composition results
- Not measured; phase-invariant `U(C2∘C1)` versus `U(C2)U(C1)` ordering diagnostic is implemented and tested.

## 13. Runtime and XPU utilization
- Dataset and training runtime: not measured. XPU memory and throughput: unavailable because no native XPU device was exposed.

## 14. Scaling conclusion
- More data: unknown. Larger model: unknown. Relative effect: unknown. Saturation: unknown. Bottleneck localization: unknown.

## 15. Final verdict
**INCONCLUSIVE**
- Infrastructure status: `XPU-BLOCKED`. This is not a scientific NO-GO.

## 16. Minimal next experiment
- Install/use a native Intel-XPU-enabled PyTorch runtime, then run `uv run python run_p4_5.py run-all`. G1 will rerun all numerical and device-residency checks before dataset generation or training.

## 17. Repository changes
- Added the P4.5 module, CLI, focused tests, and blocked provenance artifacts under `artifacts/cc_nqe_p4_5/`. No commit or push was performed; P1–P4 artifacts remain frozen.
