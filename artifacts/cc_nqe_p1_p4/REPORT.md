# CC-NQE P1–P4 Controlled Feasibility Report

## 1. Starting repository state

- Branch: `research/cc-nqe-p1-p4`
- Starting HEAD: `2a2c97d0a1e64d4642f6e3abf97a5d87c0b2a289`
- Starting working tree: clean; repository contained only `README.md`.
- Environment: system default Python 3.14.7; experiment venv Python 3.12.13; Linux-7.1.8-1-cachyos-x86_64-with-glibc2.44; AMD Ryzen 5 5600X 6-Core Processor; 16665006080 bytes RAM; NumPy 2.5.2; PyTorch 2.13.0+cu130; CPU-only.
- Existing relevant infrastructure: none (no simulator, models, tests, package config, utilities, or provenance conventions).

## 2. P1 dataset implementation

- Dataset size: 1200 samples; per split: `{"composition_ood": 64, "depth_ood": 64, "iid": 64, "parameter_extrapolation": 64, "parameter_interpolation": 64, "state_ood": 64, "train": 768, "validation": 48}`.
- Circuit counts: `{"composition_ood": 8, "depth_ood": 8, "iid": 8, "parameter_extrapolation": 8, "parameter_interpolation": 8, "state_ood": 8, "train": 48, "validation": 6}`.
- State counts: `{"composition_ood": 8, "depth_ood": 8, "iid": 8, "parameter_extrapolation": 8, "parameter_interpolation": 8, "state_ood": 8, "train": 16, "validation": 8}`.
- State-family distribution: `{"Haar-random": 300, "entangled": 300, "product": 300, "random-local": 300}`.
- Depth distribution: `{"2": 432, "4": 368, "6": 336, "8": 64}`.
- Gate distribution: `{"CNOT": 600, "H": 616, "RX": 1008, "RY": 824, "RZ": 1160, "X": 656}`.
- Generation seed: `20260811`. Teacher: `cc_nqe.numpy_exact_statevector_v1`; storage dtype `complex128`.
- Maximum input/target normalization errors: 2.220e-16 / 4.441e-16 (tolerance 1.0e-12).
- Artifacts: `artifacts/cc_nqe_p1_p4/dataset/samples.npz`, `artifacts/cc_nqe_p1_p4/dataset/samples.jsonl`, `artifacts/cc_nqe_p1_p4/dataset/manifest.json`. SHA-256 values are in `artifacts/cc_nqe_p1_p4/artifact_hashes.json`.
- Resource limitation: this is a compact 1,200-sample, 30-epoch, CPU-only experiment rather than a performance-tuned scale study.

State families are generated as follows: `product` is a tensor product sampled from {|0>,|1>,|+>,|+i>}; `random-local` is a tensor product of independently sampled Bloch-sphere states; `entangled` applies a CNOT chain and a local RY to a random-local state; `Haar-random` is a normalized iid complex-Gaussian vector.

## 3. Split contract

- State-OOD: state IDs disjoint from training while reusing eight learned train circuit contexts.
- Parameter-OOD interpolation: every parameter is in `((2.0943951023931953, 3.141592653589793),)`; training excludes it.
- Parameter-OOD extrapolation: every parameter is in `((5.235987755982989, 6.283185307179586),)`; training excludes it.
- Composition-OOD: every held-out circuit contains directed `CNOT(0,1)`; training excludes that motif. Ordered parameter-free gate/qubit signatures are also disjoint.
- Depth-OOD: training depths are 2/4/6 only; held-out depth is 8 only.
- Machine-readable contract: `artifacts/cc_nqe_p1_p4/dataset/split_manifest.json`.

## 4. Leakage and integrity audit

- PASS — `state_ood_no_state_leakage`
- PASS — `all_ood_no_state_leakage`
- PASS — `heldout_circuit_ids_disjoint`
- PASS — `heldout_structures_disjoint`
- PASS — `parameter_interpolation_contract`
- PASS — `parameter_extrapolation_contract`
- PASS — `training_excludes_parameter_holdouts`
- PASS — `composition_contract`
- PASS — `depth_contract`
- PASS — `no_duplicate_circuit_serializations`
- PASS — `state_normalization`
- PASS — `target_normalization`
- PASS — `deterministic_regeneration`
- Overall: **PASS**; duplicate sample count: 0. No repairs were required.

## 5. P2 architectures

- State-only: 32 real/imag state inputs → two ReLU hidden layers (128) → 32 raw outputs; 24,864 parameters; no circuit conditioning.
- Flat MLP: state plus 8 padded gates, each represented by gate one-hot, source/target qubit one-hots, and sin/cos/parameter mask → two ReLU hidden layers (128) → output; 42,272 parameters.
- Transformer: structural gate-type + qubit + continuous-parameter + position embeddings; two-layer, four-head circuit encoder; separate state encoder; fused prediction head; 59,728 parameters. `encode_context` is reusable.
- Output: 32 raw float32 real/imag components. Raw norms are recorded; predictions are explicitly L2-normalized for physical evaluation. The phase-invariant objective and fidelity ignore global phase.
- Training: `{"batch_size": 64, "epochs": 30, "learning_rate": 0.001, "objective": "1 - pure-state fidelity", "optimizer": "Adam", "schedule": "constant", "stopping_rule": "fixed 30 epochs"}`; seeds `[11, 23, 37]`; no test tuning. Config/checkpoint paths and SHA-256 hashes are machine-readable under `artifacts/cc_nqe_p1_p4`.

## 6. Training results

- state_only, seed 11: final train loss 0.795435; validation loss 0.912935; checkpoint `artifacts/cc_nqe_p1_p4/checkpoints/state_only_seed11.pt`; SHA-256 `84d8e2ba172e9bf80f29144556694c1cdbf30df474822b066c27eda380858520`.
- state_only, seed 23: final train loss 0.794196; validation loss 0.932799; checkpoint `artifacts/cc_nqe_p1_p4/checkpoints/state_only_seed23.pt`; SHA-256 `2bd8bb92474033467ae38e91aa3fbddeff95808d839debe02863c0eed3358365`.
- state_only, seed 37: final train loss 0.794250; validation loss 0.935115; checkpoint `artifacts/cc_nqe_p1_p4/checkpoints/state_only_seed37.pt`; SHA-256 `543be8faa8aafed975e919a14c9e7bdae1dedadb0444d506a3026ca0ebf05648`.
- flat_mlp, seed 11: final train loss 0.374487; validation loss 0.937134; checkpoint `artifacts/cc_nqe_p1_p4/checkpoints/flat_mlp_seed11.pt`; SHA-256 `375041f4bd0804ced9ba27a9f90652e14ca4b593b866c184c5949683cef2784e`.
- flat_mlp, seed 23: final train loss 0.345294; validation loss 0.928733; checkpoint `artifacts/cc_nqe_p1_p4/checkpoints/flat_mlp_seed23.pt`; SHA-256 `4c16f201661b7a54d1ed22b0219c4375bcc942d1ff2384fbab03e91629fdbc30`.
- flat_mlp, seed 37: final train loss 0.364873; validation loss 0.934864; checkpoint `artifacts/cc_nqe_p1_p4/checkpoints/flat_mlp_seed37.pt`; SHA-256 `23662eb1bc4c8d998ef3999f2edb0a21ff7f5ad08b5a77343b529dbcdb4dfe3e`.
- transformer, seed 11: final train loss 0.555004; validation loss 0.937886; checkpoint `artifacts/cc_nqe_p1_p4/checkpoints/transformer_seed11.pt`; SHA-256 `a36a19cf44fc88170e76fbd48921a4448d3bce79dcf04230114b38218b14e9e1`.
- transformer, seed 23: final train loss 0.584330; validation loss 0.938288; checkpoint `artifacts/cc_nqe_p1_p4/checkpoints/transformer_seed23.pt`; SHA-256 `7d74b138ea45605d59e5ebea187055b099167035c25d0e957aa1e027bce0e44e`.
- transformer, seed 37: final train loss 0.587289; validation loss 0.934669; checkpoint `artifacts/cc_nqe_p1_p4/checkpoints/transformer_seed37.pt`; SHA-256 `4d23930928b844d0e4de3f836920f5df55844e7e5f248a44700a36990b4ad981`.
- state_only aggregate final loss: train 0.794627 ± 0.000572; validation 0.926950 ± 0.009955.
- flat_mlp aggregate final loss: train 0.361552 ± 0.012147; validation 0.933577 ± 0.003548.
- transformer aggregate final loss: train 0.575541 ± 0.014572; validation 0.936948 ± 0.001620.
- Cross-seed aggregation uses every predeclared seed; no best-seed selection. See `per_seed_aggregates.json` and `cross_seed_aggregates.json`.

## 7. P3 fidelity results

Each line reports cross-seed mean of the per-seed distribution statistics (source per-example records are retained).
- **state_only**
  - iid: mean=0.174570, median=0.125069, std=0.191397, P05=0.005134, P95=0.649396, P01=0.000746, minimum=0.000543
  - state_ood: mean=0.089978, median=0.073477, std=0.075440, P05=0.005017, P95=0.228755, P01=0.002037, minimum=0.001624
  - parameter_interpolation: mean=0.069371, median=0.047403, std=0.072361, P05=0.004570, P95=0.200677, P01=0.001501, minimum=0.000639
  - parameter_extrapolation: mean=0.107675, median=0.067283, std=0.120143, P05=0.007351, P95=0.364171, P01=0.002455, minimum=0.001099
  - composition_ood: mean=0.067090, median=0.050491, std=0.059725, P05=0.003575, P95=0.196128, P01=0.001172, minimum=0.000717
  - depth_ood: mean=0.062727, median=0.041619, std=0.060054, P05=0.004593, P95=0.178425, P01=0.002090, minimum=0.000981
  - raw pre-normalization norm error across OOD/IID split summaries: mean=0.764833, mean P95=0.804320; fidelity uses explicitly normalized predictions.
- **flat_mlp**
  - iid: mean=0.087401, median=0.064148, std=0.085781, P05=0.004244, P95=0.246963, P01=0.001640, minimum=0.000398
  - state_ood: mean=0.103551, median=0.062160, std=0.109528, P05=0.004073, P95=0.318374, P01=0.000929, minimum=0.000640
  - parameter_interpolation: mean=0.068797, median=0.054024, std=0.060937, P05=0.003643, P95=0.190607, P01=0.000876, minimum=0.000758
  - parameter_extrapolation: mean=0.077350, median=0.056290, std=0.073931, P05=0.004218, P95=0.226933, P01=0.002362, minimum=0.001446
  - composition_ood: mean=0.066872, median=0.048930, std=0.066477, P05=0.003388, P95=0.200161, P01=0.000914, minimum=0.000347
  - depth_ood: mean=0.058312, median=0.046015, std=0.054919, P05=0.002305, P95=0.163826, P01=0.000721, minimum=0.000306
  - raw pre-normalization norm error across OOD/IID split summaries: mean=0.891835, mean P95=0.920940; fidelity uses explicitly normalized predictions.
- **transformer**
  - iid: mean=0.092728, median=0.055996, std=0.097180, P05=0.006906, P95=0.269649, P01=0.001138, minimum=0.000273
  - state_ood: mean=0.100570, median=0.064360, std=0.095243, P05=0.004564, P95=0.304079, P01=0.000681, minimum=0.000390
  - parameter_interpolation: mean=0.061871, median=0.046598, std=0.053866, P05=0.006049, P95=0.164866, P01=0.001557, minimum=0.000962
  - parameter_extrapolation: mean=0.071797, median=0.053020, std=0.064596, P05=0.005690, P95=0.202376, P01=0.001582, minimum=0.000967
  - composition_ood: mean=0.062969, median=0.044560, std=0.056599, P05=0.006478, P95=0.180431, P01=0.002511, minimum=0.001898
  - depth_ood: mean=0.067159, median=0.045772, std=0.065407, P05=0.006189, P95=0.211080, P01=0.002639, minimum=0.001132
  - raw pre-normalization norm error across OOD/IID split summaries: mean=0.570938, mean P95=0.726724; fidelity uses explicitly normalized predictions.

## 8. Observable results

Absolute observable-error mean/tails below average the corresponding per-observable, per-split, per-seed statistics; full X_i, Z_i, and Z_iZ_j distributions are in per-seed aggregates and raw JSONL.
- state_only: X mean/P95/P99/max=0.417814/0.953520/1.242693/1.329388; Z mean/P95/P99/max=0.459105/1.118353/1.325257/1.377726; ZZ mean/P95/P99/max=0.358796/0.918714/1.115046/1.195366.
- flat_mlp: X mean/P95/P99/max=0.367685/0.915787/1.128158/1.215041; Z mean/P95/P99/max=0.436737/1.057483/1.257751/1.343853; ZZ mean/P95/P99/max=0.350114/0.885188/1.102735/1.201712.
- transformer: X mean/P95/P99/max=0.376159/0.927125/1.141847/1.225857; Z mean/P95/P99/max=0.445159/1.075734/1.285811/1.358984; ZZ mean/P95/P99/max=0.363642/0.916414/1.128258/1.209708.

## 9. Linearity diagnostic

Method: for a fixed held-out circuit, form a normalized complex superposition of two held-out states. Align each separate branch prediction to its exact branch target to remove the fidelity loss's arbitrary branch phase, then compare the normalized direct prediction with the normalized same-coefficient combination using phase-invariant fidelity.
- state_only: mean across seeds 0.235029 ± 0.064985 (32 combinations/seed).
- flat_mlp: mean across seeds 0.252823 ± 0.049897 (32 combinations/seed).
- transformer: mean across seeds 0.411049 ± 0.032035 (32 combinations/seed).
- Transformer minus flat-MLP linearity fidelity: +0.158227; this compact run does show more operator-like behavior by this diagnostic.

## 10. Context-utility verdict

- Does circuit context help? IID mean fidelity: state-only 0.174570, flat 0.087401, transformer 0.092728.
- Does structured encoding help over flat? Transformer-minus-flat IID +0.005327; composition/depth average +0.002471.
- Does advantage survive OOD? Strongest-OOD averages: state-only 0.064909, flat 0.062592, transformer 0.065064.
- Parameter counts are reported above: flat and Transformer are of the same order, while the state-only negative control is smaller.

## 11. P4 exact-simulator benchmark

- Hardware/software: `{"cpu": "AMD Ryzen 5 5600X 6-Core Processor", "device": "CPU", "dtype": "float32 neural; complex128 exact", "gpu": "none/driver unavailable", "logical_cpus": 12, "native_threadpools": [{"architecture": "Haswell", "filepath": "/home/midori/Develop/Quantum_research/QuTE/.venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-61654e39.so", "internal_api": "openblas", "num_threads": 1, "prefix": "libscipy_openblas", "threading_layer": "pthreads", "user_api": "blas", "version": "0.3.34.0.0"}, {"filepath": "/home/midori/Develop/Quantum_research/QuTE/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1", "internal_api": "openmp", "num_threads": 1, "prefix": "libgomp", "user_api": "openmp", "version": null}], "numpy": "2.5.2", "platform": "Linux-7.1.8-1-cachyos-x86_64-with-glibc2.44", "psutil": "7.2.2", "python": "3.12.13", "ram_bytes": 16665006080, "thread_environment": {"MKL_NUM_THREADS": null, "OMP_NUM_THREADS": null, "OPENBLAS_NUM_THREADS": null}, "torch": "2.13.0+cu130", "torch_threads": 1}`.
- Exact uncached single-query gate simulation: mean=0.108597, median=0.076449, std=0.048326, P05=0.071650, P95=0.182834, P01=0.071514, minimum=0.071494 ms; exact cached-unitary application: mean=0.001947, median=0.001573, std=0.002229, P05=0.001538, P95=0.001943, P01=0.001527, minimum=0.001523 ms.
- Neural full/cold single-query: mean=0.704591, median=0.693985, std=0.061595, P05=0.685987, P95=0.716849, P01=0.628667, minimum=0.575402 ms; cached forward: mean=0.062960, median=0.062066, std=0.003653, P05=0.060184, P95=0.069286, P01=0.059748, minimum=0.059512 ms.
- Exact unitary preparation: mean=1.945092, median=1.649871, std=0.615264, P05=1.311867, P95=2.987718, P01=1.240038, minimum=1.193533 ms. Neural median circuit preprocessing/context encoding/state preprocessing/normalization-only estimate: 0.026274/0.523429/0.004603/0.024566 ms.
- Batch throughput: `[{"batch_size": 1, "median_ms": 0.034505, "states_per_sec": 28981.307056948266}, {"batch_size": 10, "median_ms": 0.059547, "states_per_sec": 167934.57269047978}, {"batch_size": 100, "median_ms": 0.1227905, "states_per_sec": 814395.2504468994}, {"batch_size": 1000, "median_ms": 0.5429805, "states_per_sec": 1841686.7640734797}]`.
- Process RSS at measurement: 731205632 bytes; combined-run rusage high-water mark: 717196 KiB. Neither attributes memory separately to exact and neural methods, so no comparative memory result is established.

## 12. Repeated-context/cache benchmark

CPU exact and neural timings use the same host and one native/Torch thread. Exact reports direct gate simulation and cached 16×16-unitary BLAS application, selecting the faster measured exact path at each N; neural inference encodes once and batches by up to 1024. Each N uses five post-warmup measurements (component medians).
- N=1: exact direct 0.188 ms; preparation/apply/cached-total 3.018/0.002/3.020 ms; selected `direct_gate_simulation` total 0.188 ms (0.188224 ms/query, 5312.8 states/s). Neural circuit preprocessing/context encoding/state preprocessing/cached inference/normalization 0.029/0.387/0.012/0.055/0.021 ms, total 0.505 ms (0.504981 ms/query, 1980.3 states/s).
- N=10: exact direct 0.916 ms; preparation/apply/cached-total 1.552/0.003/1.555 ms; selected `direct_gate_simulation` total 0.916 ms (0.091606 ms/query, 10916.3 states/s). Neural circuit preprocessing/context encoding/state preprocessing/cached inference/normalization 0.028/0.352/0.012/0.066/0.019 ms, total 0.478 ms (0.047821 ms/query, 20911.4 states/s).
- N=100: exact direct 8.690 ms; preparation/apply/cached-total 2.996/0.010/3.006 ms; selected `cached_unitary` total 3.006 ms (0.030062 ms/query, 33264.8 states/s). Neural circuit preprocessing/context encoding/state preprocessing/cached inference/normalization 0.039/0.543/0.016/0.150/0.030 ms, total 0.779 ms (0.007792 ms/query, 128331.2 states/s).
- N=1000: exact direct 135.398 ms; preparation/apply/cached-total 1.614/0.058/1.672 ms; selected `cached_unitary` total 1.672 ms (0.001672 ms/query, 598160.2 states/s). Neural circuit preprocessing/context encoding/state preprocessing/cached inference/normalization 0.030/0.388/0.034/0.541/0.038 ms, total 1.032 ms (0.001032 ms/query, 969454.3 states/s).
- N=10000: exact direct 1258.042 ms; preparation/apply/cached-total 1.563/0.448/2.012 ms; selected `cached_unitary` total 2.012 ms (0.000201 ms/query, 4970843.5 states/s). Neural circuit preprocessing/context encoding/state preprocessing/cached inference/normalization 0.033/0.404/1.270/5.134/0.562 ms, total 7.403 ms (0.000740 ms/query, 1350735.1 states/s).
- First observed crossover N: `10`; neural-faster tested N: `[10, 100, 1000]`; sustained break-even through the largest tested N: `None`.
- Timing boundary: full neural query includes circuit preprocessing, context encoding, state conversion, forward, and normalization; cached forward excludes all but forward; repeated exact total is the faster measured direct-gate or unitary-cache path; neural total includes preprocessing, encoding, conversion, forward, and normalization.

## 13. Accuracy–compute tradeoff

- State-only has the lowest parameter count but no reusable operator context. Flat/Transformer fidelity must be read alongside the exact-vs-neural timings above. The observed frontier is not compressed into a score: strongest-OOD fidelity (state/flat/transformer) is 0.064909/0.062592/0.065064, while neural repeated-query throughput ranges from 1980.3 to 1350735.1 states/s.

## 14. Final gate

**NO-GO**

The controlled run does not support all three required conditions. The measured fidelity and strongest-OOD results do not establish reliable reusable transformation learning, regardless of whether batching/caching gives computational throughput benefit. A larger valid training set and/or model may be tested next, but multimodal scale-up is not justified by this run.

## 15. Scientific interpretation

- **Demonstrated:** deterministic leakage-audited 4-qubit data generation, phase-invariant evaluation, structural context encoding, raw physicality/observable/linearity diagnostics, and a same-CPU exact-vs-neural context-cache harness.
- **Suggested:** restricted neural surrogates may amortize computation in some batched repeated-context regimes; any suggestion is conditional on the measured low approximation fidelity.
- **Not demonstrated:** accurate generalization to unseen compositions/depths, replacement of arbitrary quantum simulation, asymptotic complexity or memory advantage, QPU relevance, or Hamiltonian/cross-modal learning.

## 16. Repository changes

- Added: `.gitignore`, `pyproject.toml`, `uv.lock`, `cc_nqe.py`, `run_experiment.py`, `tests/test_cc_nqe.py`, and generated `artifacts/cc_nqe_p1_p4/**`.
- Modified: `README.md`.
- Commits created: none.
- Final HEAD: `2a2c97d0a1e64d4642f6e3abf97a5d87c0b2a289` (unchanged from starting HEAD).
- Working tree: intentionally uncommitted research implementation/artifacts; no staged files.

Machine-readable result hashes: `artifacts/cc_nqe_p1_p4/artifact_hashes.json`; uncommitted source hashes: `artifacts/cc_nqe_p1_p4/experiment_config.json`. Report generated from recorded artifacts, not used as their source of truth.
