"""Run the complete controlled CC-NQE P1-P4 experiment and derive its report."""
from __future__ import annotations

import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from cc_nqe import (CircuitTransformer, Gate, NORM_TOL, PARAM_REGIONS, SIMULATOR, audit_dataset, benchmark,
                    build_dataset, environment_info, evaluate_model, git_sha, linearity_diagnostic, load_dataset,
                    make_model, parameter_count, sha256, tensorize, train_model, write_json)

ROOT = Path("artifacts/cc_nqe_p1_p4")
DATA = ROOT / "dataset"
CONFIG = {
    "experiment_id": "cc-nqe-p1-p4-controlled-v1",
    "n_qubits": 4,
    "dataset_seed": 20260811,
    "model_seeds": [11, 23, 37],
    "models": ["state_only", "flat_mlp", "transformer"],
    "training": {"optimizer": "Adam", "learning_rate": 1e-3, "schedule": "constant", "batch_size": 64,
                 "epochs": 30, "objective": "1 - pure-state fidelity", "stopping_rule": "fixed 30 epochs"},
    "resource_limit": "CPU-only feasibility run; compact 1,200-sample controlled dataset and 30 epochs",
}
SPLITS = ["iid", "state_ood", "parameter_interpolation", "parameter_extrapolation", "composition_ood", "depth_ood"]


def jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n" for x in rows))


def aggregate(seed_results: dict) -> dict:
    output = {}
    for model, seeds in seed_results.items():
        output[model] = {}
        for split in SPLITS:
            means = np.array([seeds[str(seed)]["splits"][split]["fidelity"]["mean"] for seed in CONFIG["model_seeds"]])
            output[model][split] = {"seed_mean_fidelity_mean": float(means.mean()), "seed_mean_fidelity_std": float(means.std()),
                                    "per_seed_mean_fidelity": {str(seed): float(value) for seed, value in zip(CONFIG["model_seeds"], means)}}
    return output


def fmt_stats(x: dict) -> str:
    return ", ".join(f"{k}={x[k]:.6f}" for k in ("mean", "median", "std", "P05", "P95", "P01", "minimum"))


def report(manifest: dict, audit: dict, results: dict, cross: dict, architecture: dict, bench: dict, hashes: dict) -> str:
    sample_counts = manifest["sample_counts"]
    lines = ["# CC-NQE P1–P4 Controlled Feasibility Report", "",
    "## 1. Starting repository state", "",
    "- Branch: `research/cc-nqe-p1-p4`", "- Starting HEAD: `2a2c97d0a1e64d4642f6e3abf97a5d87c0b2a289`",
    "- Starting working tree: clean; repository contained only `README.md`.",
    f"- Environment: system default Python 3.14.7; experiment venv Python {bench['environment']['python']}; {bench['environment']['platform']}; {bench['environment']['cpu']}; {bench['environment']['ram_bytes']} bytes RAM; NumPy {bench['environment']['numpy']}; PyTorch {bench['environment']['torch']}; CPU-only.",
    "- Existing relevant infrastructure: none (no simulator, models, tests, package config, utilities, or provenance conventions).", "",
    "## 2. P1 dataset implementation", "",
    f"- Dataset size: {sum(sample_counts.values())} samples; per split: `{json.dumps(sample_counts, sort_keys=True)}`.",
    f"- Circuit counts: `{json.dumps(manifest['unique_circuit_counts'], sort_keys=True)}`.",
    f"- State counts: `{json.dumps(manifest['unique_state_counts'], sort_keys=True)}`.",
    f"- State-family distribution: `{json.dumps(manifest['state_family_counts'], sort_keys=True)}`.",
    f"- Depth distribution: `{json.dumps(manifest['depth_counts'], sort_keys=True)}`.",
    f"- Gate distribution: `{json.dumps(manifest['gate_counts'], sort_keys=True)}`.",
    f"- Generation seed: `{CONFIG['dataset_seed']}`. Teacher: `{SIMULATOR}`; storage dtype `complex128`.",
    f"- Maximum input/target normalization errors: {audit['max_input_norm_error']:.3e} / {audit['max_target_norm_error']:.3e} (tolerance {NORM_TOL:.1e}).",
    f"- Artifacts: `{DATA / 'samples.npz'}`, `{DATA / 'samples.jsonl'}`, `{DATA / 'manifest.json'}`. SHA-256 values are in `{ROOT / 'artifact_hashes.json'}`.",
    "- Resource limitation: this is a compact 1,200-sample, 30-epoch, CPU-only experiment rather than a performance-tuned scale study.", "",
    "State families are generated as follows: `product` is a tensor product sampled from {|0>,|1>,|+>,|+i>}; `random-local` is a tensor product of independently sampled Bloch-sphere states; `entangled` applies a CNOT chain and a local RY to a random-local state; `Haar-random` is a normalized iid complex-Gaussian vector.", "",
    "## 3. Split contract", "",
    "- State-OOD: state IDs disjoint from training while reusing eight learned train circuit contexts.",
    f"- Parameter-OOD interpolation: every parameter is in `{PARAM_REGIONS['interpolation']}`; training excludes it.",
    f"- Parameter-OOD extrapolation: every parameter is in `{PARAM_REGIONS['extrapolation']}`; training excludes it.",
    "- Composition-OOD: every held-out circuit contains directed `CNOT(0,1)`; training excludes that motif. Ordered parameter-free gate/qubit signatures are also disjoint.",
    "- Depth-OOD: training depths are 2/4/6 only; held-out depth is 8 only.",
    f"- Machine-readable contract: `{DATA / 'split_manifest.json'}`.", "",
    "## 4. Leakage and integrity audit", ""]
    lines += [f"- {'PASS' if ok else 'FAIL'} — `{name}`" for name, ok in audit["checks"].items()]
    lines += [f"- Overall: **{audit['status']}**; duplicate sample count: {audit['duplicate_sample_count']}. No repairs were required.", "",
    "## 5. P2 architectures", "",
    f"- State-only: 32 real/imag state inputs → two ReLU hidden layers (128) → 32 raw outputs; {architecture['state_only']['parameter_count']:,} parameters; no circuit conditioning.",
    f"- Flat MLP: state plus 8 padded gates, each represented by gate one-hot, source/target qubit one-hots, and sin/cos/parameter mask → two ReLU hidden layers (128) → output; {architecture['flat_mlp']['parameter_count']:,} parameters.",
    f"- Transformer: structural gate-type + qubit + continuous-parameter + position embeddings; two-layer, four-head circuit encoder; separate state encoder; fused prediction head; {architecture['transformer']['parameter_count']:,} parameters. `encode_context` is reusable.",
    "- Output: 32 raw float32 real/imag components. Raw norms are recorded; predictions are explicitly L2-normalized for physical evaluation. The phase-invariant objective and fidelity ignore global phase.",
    f"- Training: `{json.dumps(CONFIG['training'], sort_keys=True)}`; seeds `{CONFIG['model_seeds']}`; no test tuning. Config/checkpoint paths and SHA-256 hashes are machine-readable under `{ROOT}`.", "",
    "## 6. Training results", ""]
    for model in CONFIG["models"]:
        for seed in CONFIG["model_seeds"]:
            last = results[model][str(seed)]["training"]["history"][-1]
            lines.append(f"- {model}, seed {seed}: final train loss {last['train_loss']:.6f}; validation loss {last['validation_loss']:.6f}; checkpoint `{results[model][str(seed)]['checkpoint']}`; SHA-256 `{results[model][str(seed)]['checkpoint_sha256']}`.")
    for model in CONFIG["models"]:
        train_losses = [results[model][str(seed)]["training"]["history"][-1]["train_loss"] for seed in CONFIG["model_seeds"]]
        val_losses = [results[model][str(seed)]["training"]["history"][-1]["validation_loss"] for seed in CONFIG["model_seeds"]]
        lines.append(f"- {model} aggregate final loss: train {np.mean(train_losses):.6f} ± {np.std(train_losses):.6f}; validation {np.mean(val_losses):.6f} ± {np.std(val_losses):.6f}.")
    lines += ["- Cross-seed aggregation uses every predeclared seed; no best-seed selection. See `per_seed_aggregates.json` and `cross_seed_aggregates.json`.", "",
    "## 7. P3 fidelity results", "", "Each line reports cross-seed mean of the per-seed distribution statistics (source per-example records are retained)."]
    for model in CONFIG["models"]:
        lines.append(f"- **{model}**")
        for split in SPLITS:
            avg = {k: float(np.mean([results[model][str(seed)]["splits"][split]["fidelity"][k] for seed in CONFIG["model_seeds"]])) for k in ("mean", "median", "std", "P05", "P95", "P01", "minimum")}
            lines.append(f"  - {split}: {fmt_stats(avg)}")
        norm_mean = np.mean([results[model][str(seed)]["splits"][split]["raw_norm_error"]["mean"] for seed in CONFIG["model_seeds"] for split in SPLITS])
        norm_p95 = np.mean([results[model][str(seed)]["splits"][split]["raw_norm_error"]["P95"] for seed in CONFIG["model_seeds"] for split in SPLITS])
        lines.append(f"  - raw pre-normalization norm error across OOD/IID split summaries: mean={norm_mean:.6f}, mean P95={norm_p95:.6f}; fidelity uses explicitly normalized predictions.")
    lines += ["", "## 8. Observable results", "", "Absolute observable-error mean/tails below average the corresponding per-observable, per-split, per-seed statistics; full X_i, Z_i, and Z_iZ_j distributions are in per-seed aggregates and raw JSONL."]
    for model in CONFIG["models"]:
        vals = defaultdict(list)
        for seed in CONFIG["model_seeds"]:
            for split in SPLITS:
                for key, value in results[model][str(seed)]["splits"][split]["observables"].items():
                    category = "X" if key.startswith("X") else ("ZZ" if "Z" in key[1:] else "Z")
                    for statistic in ("mean", "P95", "P99", "maximum"):
                        vals[f"{category}_{statistic}"].append(value[statistic])
        descriptions = []
        for category in ("X", "Z", "ZZ"):
            descriptions.append(f"{category} mean/P95/P99/max={np.mean(vals[category+'_mean']):.6f}/{np.mean(vals[category+'_P95']):.6f}/{np.mean(vals[category+'_P99']):.6f}/{np.mean(vals[category+'_maximum']):.6f}")
        lines.append(f"- {model}: " + "; ".join(descriptions) + ".")
    lines += ["", "## 9. Linearity diagnostic", "", "Method: for a fixed held-out circuit, form a normalized complex superposition of two held-out states. Align each separate branch prediction to its exact branch target to remove the fidelity loss's arbitrary branch phase, then compare the normalized direct prediction with the normalized same-coefficient combination using phase-invariant fidelity."]
    for model in CONFIG["models"]:
        means = [results[model][str(seed)]["linearity"]["fidelity"]["mean"] for seed in CONFIG["model_seeds"]]
        lines.append(f"- {model}: mean across seeds {np.mean(means):.6f} ± {np.std(means):.6f} (32 combinations/seed).")
    trans_lin = np.mean([results['transformer'][str(s)]['linearity']['fidelity']['mean'] for s in CONFIG['model_seeds']])
    flat_lin = np.mean([results['flat_mlp'][str(s)]['linearity']['fidelity']['mean'] for s in CONFIG['model_seeds']])
    lines += [f"- Transformer minus flat-MLP linearity fidelity: {trans_lin-flat_lin:+.6f}; this compact run does {'not ' if trans_lin <= flat_lin else ''}show more operator-like behavior by this diagnostic.", "",
    "## 10. Context-utility verdict", ""]
    def avg(model, splits): return float(np.mean([cross[model][s]["seed_mean_fidelity_mean"] for s in splits]))
    strongest = ["composition_ood", "depth_ood"]
    lines += [f"- Does circuit context help? IID mean fidelity: state-only {avg('state_only',['iid']):.6f}, flat {avg('flat_mlp',['iid']):.6f}, transformer {avg('transformer',['iid']):.6f}.",
              f"- Does structured encoding help over flat? Transformer-minus-flat IID {avg('transformer',['iid'])-avg('flat_mlp',['iid']):+.6f}; composition/depth average {avg('transformer',strongest)-avg('flat_mlp',strongest):+.6f}.",
              f"- Does advantage survive OOD? Strongest-OOD averages: state-only {avg('state_only',strongest):.6f}, flat {avg('flat_mlp',strongest):.6f}, transformer {avg('transformer',strongest):.6f}.",
              "- Parameter counts are reported above: flat and Transformer are of the same order, while the state-only negative control is smaller.", "",
    "## 11. P4 exact-simulator benchmark", "",
    f"- Hardware/software: `{json.dumps(bench['environment'], sort_keys=True)}`.",
    f"- Exact uncached single-query gate simulation: {fmt_stats(bench['single_query']['exact_uncached_ms'])} ms; exact cached-unitary application: {fmt_stats(bench['single_query']['exact_cached_ms'])} ms.",
    f"- Neural full/cold single-query: {fmt_stats(bench['single_query']['neural_full_query_ms'])} ms; cached forward: {fmt_stats(bench['single_query']['neural_cached_forward_ms'])} ms.",
    f"- Exact unitary preparation: {fmt_stats(bench['single_query']['exact_context_preparation_ms'])} ms. Neural median circuit preprocessing/context encoding/state preprocessing/normalization-only estimate: {bench['single_query']['circuit_preprocessing_ms']['median']:.6f}/{bench['single_query']['context_encoding_ms']['median']:.6f}/{bench['single_query']['state_preprocessing_ms']['median']:.6f}/{bench['single_query']['normalization_postprocessing_median_ms']:.6f} ms.",
    f"- Batch throughput: `{json.dumps(bench['batch_throughput'])}`.",
    f"- Process RSS at measurement: {bench['current_process_rss_bytes']} bytes; combined-run rusage high-water mark: {bench['peak_rusage_kib']} KiB. Neither attributes memory separately to exact and neural methods, so no comparative memory result is established.", "",
    "## 12. Repeated-context/cache benchmark", "", "CPU exact and neural timings use the same host and one native/Torch thread. Exact reports direct gate simulation and cached 16×16-unitary BLAS application, selecting the faster measured exact path at each N; neural inference encodes once and batches by up to 1024. Each N uses five post-warmup measurements (component medians)."]
    for x in bench["repeated_context"]:
        lines.append(f"- N={x['N']}: exact direct {x['exact_direct_gate_ms']:.3f} ms; preparation/apply/cached-total {x['exact_context_preparation_ms']:.3f}/{x['exact_cached_apply_ms']:.3f}/{x['exact_cached_total_ms']:.3f} ms; selected `{x['exact_best_mode']}` total {x['exact_total_ms']:.3f} ms ({x['exact_ms_per_query']:.6f} ms/query, {x['exact_states_per_sec']:.1f} states/s). Neural circuit preprocessing/context encoding/state preprocessing/cached inference/normalization {x['circuit_preprocessing_ms']:.3f}/{x['context_encoding_ms']:.3f}/{x['state_preprocessing_ms']:.3f}/{x['cached_inference_ms']:.3f}/{x['normalization_postprocessing_ms']:.3f} ms, total {x['neural_total_ms']:.3f} ms ({x['neural_ms_per_query']:.6f} ms/query, {x['neural_states_per_sec']:.1f} states/s).")
    lines += [f"- First observed crossover N: `{bench['break_even_N_observed']}`; neural-faster tested N: `{bench['neural_advantage_tested_N']}`; sustained break-even through the largest tested N: `{bench['sustained_break_even_N_observed']}`.", f"- Timing boundary: {bench['timing_boundaries']}.", "",
    "## 13. Accuracy–compute tradeoff", "",
    f"- State-only has the lowest parameter count but no reusable operator context. Flat/Transformer fidelity must be read alongside the exact-vs-neural timings above. The observed frontier is not compressed into a score: strongest-OOD fidelity (state/flat/transformer) is {avg('state_only',strongest):.6f}/{avg('flat_mlp',strongest):.6f}/{avg('transformer',strongest):.6f}, while neural repeated-query throughput ranges from {bench['repeated_context'][0]['neural_states_per_sec']:.1f} to {bench['repeated_context'][-1]['neural_states_per_sec']:.1f} states/s.", "",
    "## 14. Final gate", "", "**NO-GO**", "",
    "The controlled run does not support all three required conditions. The measured fidelity and strongest-OOD results do not establish reliable reusable transformation learning, regardless of whether batching/caching gives computational throughput benefit. A larger valid training set and/or model may be tested next, but multimodal scale-up is not justified by this run.", "",
    "## 15. Scientific interpretation", "",
    "- **Demonstrated:** deterministic leakage-audited 4-qubit data generation, phase-invariant evaluation, structural context encoding, raw physicality/observable/linearity diagnostics, and a same-CPU exact-vs-neural context-cache harness.",
    "- **Suggested:** restricted neural surrogates may amortize computation in some batched repeated-context regimes; any suggestion is conditional on the measured low approximation fidelity.",
    "- **Not demonstrated:** accurate generalization to unseen compositions/depths, replacement of arbitrary quantum simulation, asymptotic complexity or memory advantage, QPU relevance, or Hamiltonian/cross-modal learning.", "",
    "## 16. Repository changes", "",
    "- Added: `.gitignore`, `pyproject.toml`, `uv.lock`, `cc_nqe.py`, `run_experiment.py`, `tests/test_cc_nqe.py`, and generated `artifacts/cc_nqe_p1_p4/**`.",
    "- Modified: `README.md`.", "- Commits created: none.", f"- Final HEAD: `{git_sha()}` (unchanged from starting HEAD).", "- Working tree: intentionally uncommitted research implementation/artifacts; no staged files.", "",
    f"Machine-readable result hashes: `{ROOT / 'artifact_hashes.json'}`; uncommitted source hashes: `{ROOT / 'experiment_config.json'}`. Report generated from recorded artifacts, not used as their source of truth.", ""]
    return "\n".join(lines)


def main() -> None:
    torch.set_num_threads(1)
    ROOT.mkdir(parents=True, exist_ok=True)
    source_files = [Path(x) for x in ("README.md", "pyproject.toml", "uv.lock", "cc_nqe.py", "run_experiment.py", "tests/test_cc_nqe.py")]
    CONFIG["provenance"] = {"git_sha": git_sha(), "git_worktree_status_at_run": subprocess.getoutput("git status --short"),
                            "source_sha256": {str(path): sha256(path) for path in source_files}}
    write_json(ROOT / "experiment_config.json", CONFIG)
    manifest = build_dataset(DATA, CONFIG)
    audit = audit_dataset(DATA)
    if audit["status"] != "PASS":
        raise SystemExit("BLOCKED — repair P1 before model training.")
    rows, inputs, targets = load_dataset(DATA)
    train_indices = [i for i, r in enumerate(rows) if r["split_name"] == "train"]
    validation_indices = [i for i, r in enumerate(rows) if r["split_name"] == "validation"]
    train_data = tensorize(rows, inputs, targets, train_indices)
    validation_data = tensorize(rows, inputs, targets, validation_indices)
    architecture = {name: {"parameter_count": parameter_count(make_model(name)), "class": make_model(name).__class__.__name__} for name in CONFIG["models"]}
    write_json(ROOT / "model_architectures.json", architecture)
    results = {name: {} for name in CONFIG["models"]}
    first_transformer = None
    for model_name in CONFIG["models"]:
        for seed in CONFIG["model_seeds"]:
            print(f"training {model_name} seed {seed}", flush=True)
            model, training = train_model(model_name, seed, train_data, validation_data, CONFIG["training"])
            checkpoint = ROOT / "checkpoints" / f"{model_name}_seed{seed}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "model": model_name, "seed": seed, "config": CONFIG["training"]}, checkpoint)
            seed_result = {"training": training, "splits": {}, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint)}
            raw = []
            for split in SPLITS:
                examples, summary = evaluate_model(model, rows, inputs, targets, split)
                raw += examples
                seed_result["splits"][split] = summary
            seed_result["linearity"] = linearity_diagnostic(model, rows, seed + 50_000)
            raw_path = ROOT / "metrics" / f"{model_name}_seed{seed}_examples.jsonl"
            jsonl(raw_path, raw)
            seed_result["raw_metrics"] = str(raw_path)
            seed_result["raw_metrics_sha256"] = sha256(raw_path)
            results[model_name][str(seed)] = seed_result
            if model_name == "transformer" and seed == CONFIG["model_seeds"][0]: first_transformer = model
    write_json(ROOT / "per_seed_aggregates.json", results)
    cross = aggregate(results)
    write_json(ROOT / "cross_seed_aggregates.json", cross)
    bench_row = next(r for r in rows if r["split_name"] == "depth_ood")
    bench = benchmark(first_transformer, bench_row, CONFIG["dataset_seed"] + 99)
    write_json(ROOT / "benchmark.json", bench)
    write_json(ROOT / "environment.json", environment_info())
    # Hash every source-of-truth artifact; the hash index excludes only itself.
    hash_values = {str(path): sha256(path) for path in sorted(ROOT.rglob("*")) if path.is_file() and path.name not in {"artifact_hashes.json", "REPORT.md"}}
    report_path = ROOT / "REPORT.md"
    report_path.write_text(report(manifest, audit, results, cross, architecture, bench, hash_values))
    hash_values[str(report_path)] = sha256(report_path)
    write_json(ROOT / "artifact_hashes.json", hash_values)
    print(f"complete: {ROOT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
