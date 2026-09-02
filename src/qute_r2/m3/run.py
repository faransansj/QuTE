from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from torch import nn

from qute_r2.m3.model import GraphAutoregressiveSampler, M3Backend, graph_tensors, save_checkpoint
from qute_r2.profiling.corpus import Workload, canonical_hash, graph_edges, parameter_points, qiskit_circuit

ROOT = Path(__file__).resolve().parents[3]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def workload(family: str, n: int, p: int, graph_seed: int, point_index: int, points) -> Workload:
    edges = graph_edges(family, n, graph_seed)
    gammas, betas = points[point_index % len(points)]
    graph_hash = canonical_hash({"family": family, "n_qubits": n, "edges": edges})
    payload = {"family": family, "n_qubits": n, "p": p, "edges": edges, "gammas": gammas[:p], "betas": betas[:p], "graph_seed": graph_seed, "point_index": point_index}
    circuit_hash = canonical_hash(payload)
    logical_id = f"m3:{family}:n{n}:p{p}:g{graph_seed}:a{point_index}"
    return Workload(logical_id, n, p, family, graph_seed, point_index, edges, gammas[:p], betas[:p], graph_hash, circuit_hash)


def fixed_corpus(config: dict[str, Any], split: str) -> list[Workload]:
    widths = config["train_widths"] if split == "calibration" else config["validation_widths"]
    seeds = config["calibration_graph_seeds"] if split == "calibration" else config["validation_graph_seeds"]
    count = config["calibration_parameter_points"] if split == "calibration" else config["validation_parameter_points"]
    points = parameter_points(count, 3)
    return [workload(family, n, p, seed, point, points) for family in config["families"] for n in widths for p in config["depths"] for seed in seeds for point in range(count)]


def robustness_corpus(config: dict[str, Any]) -> list[Workload]:
    points = parameter_points(16, 3)
    return [workload(family, n, p, seed, point, points) for family in config["families"]
            for n in config["validation_widths"] for p in config["depths"]
            for seed in config["robustness_graph_seeds"] for point in config["robustness_parameter_indices"]]


def targeted_corpus(config: dict[str, Any]) -> list[Workload]:
    cells = [(n, p) for n in config["train_widths"] for p in (2, 3)]
    points = parameter_points(1024, 3); result = []
    for index in range(config["targeted_circuits"]):
        n, p = cells[index % len(cells)]; block = index // len(cells)
        result.append(workload("random_3_regular", n, p, config["targeted_seed_base"] + block, block % len(points), points))
    return result


def full_corpus(config: dict[str, Any]) -> list[Workload]:
    cells = [(family, n, p) for family in config["families"] for n in config["train_widths"] for p in config["depths"]]
    points = parameter_points(1024, 3)
    result = []
    for index in range(config["full_train_circuits"]):
        family, n, p = cells[index % len(cells)]
        block = index // len(cells)
        seed = 41000 + block if family == "random_3_regular" else 41000
        result.append(workload(family, n, p, seed, block % len(points), points))
    return result


def workload_dict(row: Workload) -> dict[str, Any]:
    return {"logical_id": row.logical_id, "n_qubits": row.n_qubits, "p": row.p, "graph_family": row.graph_family,
            "graph_seed": row.graph_seed, "parameter_index": row.parameter_index, "edges": row.edges,
            "gammas": row.gammas, "betas": row.betas, "graph_hash": row.graph_hash, "circuit_hash": row.circuit_hash}


def workload_from_dict(row: dict[str, Any]) -> Workload:
    return Workload(row["logical_id"], row["n_qubits"], row["p"], row["graph_family"], row["graph_seed"], row["parameter_index"],
                    tuple(map(tuple, row["edges"])), tuple(row["gammas"]), tuple(row["betas"]), row["graph_hash"], row["circuit_hash"])


def counts_to_outcomes(counts: dict[str, int]) -> np.ndarray:
    return np.concatenate([np.full(count, int(bitstring, 2), dtype=np.uint32) for bitstring, count in sorted(counts.items())])


def teacher_dataset(rows: list[Workload], shots: int, path: Path, manifest_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    if path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["circuit_hashes"] == [row.circuit_hash for row in rows] and manifest["shots"] == shots:
            return {"status": "REUSED", **manifest}
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    simulator = AerSimulator(method=config["teacher_method"], precision=config["statevector_precision"], max_parallel_threads=config["threads"], max_parallel_experiments=1)
    outcomes = np.zeros((len(rows), shots), dtype=np.uint32)
    timings = []
    for index, row in enumerate(rows):
        circuit = transpile(qiskit_circuit(row), simulator, optimization_level=0, seed_transpiler=config["seed"])
        start = time.perf_counter_ns()
        result = simulator.run(circuit, shots=shots, seed_simulator=config["seed"] + index).result()
        timings.append((time.perf_counter_ns() - start) / 1e6)
        values = counts_to_outcomes(result.get_counts())
        if len(values) != shots:
            raise RuntimeError("teacher counts do not sum to shots")
        np.random.default_rng(config["seed"] + index).shuffle(values)
        outcomes[index] = values
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, outcomes=outcomes)
    manifest = {"schema_version": config["schema_version"], "shots": shots, "count": len(rows),
                "circuit_hashes": [row.circuit_hash for row in rows], "workloads": [workload_dict(row) for row in rows],
                "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "teacher_method": config["teacher_method"],
                "latency_ms": {"median": statistics.median(timings), "mean": statistics.mean(timings), "maximum": max(timings), "all": timings}}
    write_json(manifest_path, manifest)
    return {"status": "GENERATED", **manifest}


def bits_from_outcomes(outcomes: np.ndarray, max_qubits: int = 24) -> torch.Tensor:
    shifts = np.arange(max_qubits, dtype=np.uint32)
    return torch.from_numpy(((outcomes[..., None] >> shifts) & 1).astype(np.float32))


def train_model(rows: list[Workload], dataset_path: Path, config: dict[str, Any], full: bool, *, model: GraphAutoregressiveSampler | None = None, epochs_override: int | None = None) -> tuple[GraphAutoregressiveSampler, list[dict[str, float]]]:
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.set_num_threads(config["threads"])
    model = model or GraphAutoregressiveSampler(**config["model"])
    if full and config.get("initial_checkpoint") and not epochs_override:
        payload = torch.load(ROOT / config["initial_checkpoint"], map_location="cpu", weights_only=True)
        current = model.state_dict()
        compatible = {key: value for key, value in payload["state_dict"].items() if key in current and current[key].shape == value.shape}
        model.load_state_dict(compatible, strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    outcomes = np.load(dataset_path)["outcomes"]
    batch_size = config["training"]["batch_circuits"]
    sample_count = min(config["training"]["samples_per_circuit"], outcomes.shape[1])
    epochs = epochs_override if epochs_override is not None else (config["training"]["epochs_full"] if full else config["training"]["epochs_calibration"])
    history = []
    rng = np.random.default_rng(config["seed"] + int(full))
    for epoch in range(epochs):
        order = rng.permutation(len(rows)); losses = []
        for start in range(0, len(rows), batch_size):
            indices = order[start:start + batch_size]
            selected = [rows[int(index)] for index in indices]
            feature, adjacency, mask = graph_tensors(selected)
            sample_indices = rng.integers(0, outcomes.shape[1], size=(len(indices), sample_count))
            sampled = np.take_along_axis(outcomes[indices], sample_indices, axis=1)
            bits = bits_from_outcomes(sampled)
            logits = model(feature, adjacency, mask, bits)
            loss_rows = nn.functional.binary_cross_entropy_with_logits(logits, bits, reduction="none") * mask[:, None, :]
            loss = loss_rows.sum() / (mask.sum() * sample_count)
            if config["training"].get("zz_loss_weight", 0):
                correlation_losses = []
                signed = bits * 2 - 1
                conditional_mean = torch.tanh(logits / 2)
                for circuit_index, circuit in enumerate(selected):
                    for a, b in circuit.edges:
                        earlier, later = min(a, b), max(a, b)
                        teacher_correlation = (signed[circuit_index, :, earlier] * signed[circuit_index, :, later]).mean()
                        predicted_correlation = (signed[circuit_index, :, earlier] * conditional_mean[circuit_index, :, later]).mean()
                        weight = config["training"].get("zz_loss_p2_weight", 1.0) if circuit.p == 2 else 1.0
                        correlation_losses.append(weight * (predicted_correlation - teacher_correlation).square())
                loss = loss + config["training"]["zz_loss_weight"] * torch.stack(correlation_losses).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch + 1, "mean_loss": statistics.mean(losses), "median_loss": statistics.median(losses)})
    model.eval()
    return model, history


def outcome_metrics(teacher: np.ndarray, predicted: np.ndarray, row: Workload) -> dict[str, float]:
    n = row.n_qubits
    shifts = np.arange(n, dtype=np.uint32)
    teacher_bits = ((teacher[:, None] >> shifts) & 1).astype(np.float32)
    predicted_bits = ((predicted[:, None] >> shifts) & 1).astype(np.float32)
    teacher_energy = np.zeros(len(teacher), dtype=np.float32); predicted_energy = np.zeros(len(predicted), dtype=np.float32)
    teacher_zz, predicted_zz = [], []
    for a, b in row.edges:
        teacher_energy += teacher_bits[:, a] != teacher_bits[:, b]
        predicted_energy += predicted_bits[:, a] != predicted_bits[:, b]
        teacher_zz.append(np.mean((1 - 2 * teacher_bits[:, a]) * (1 - 2 * teacher_bits[:, b])))
        predicted_zz.append(np.mean((1 - 2 * predicted_bits[:, a]) * (1 - 2 * predicted_bits[:, b])))
    return {"energy_error_per_edge": float(abs(teacher_energy.mean() - predicted_energy.mean()) / len(row.edges)),
            "marginal_mae": float(np.mean(np.abs(teacher_bits.mean(0) - predicted_bits.mean(0)))),
            "zz_mae": float(np.mean(np.abs(np.asarray(teacher_zz) - np.asarray(predicted_zz))))}


def exact_tvd_18(model: GraphAutoregressiveSampler, row: Workload) -> float:
    from qiskit.quantum_info import Statevector

    exact = Statevector.from_instruction(qiskit_circuit(row, measure=False)).probabilities()
    feature, adjacency, mask = graph_tensors([row])
    probabilities = np.empty(1 << row.n_qubits, dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(probabilities), 8192):
            values = np.arange(start, min(start + 8192, len(probabilities)), dtype=np.uint32)
            bits = bits_from_outcomes(values[None, :])[:, :, :row.n_qubits]
            logits = model(feature, adjacency, mask, bits_from_outcomes(values[None, :]))[0, :, :row.n_qubits]
            logp = -nn.functional.binary_cross_entropy_with_logits(logits, bits[0], reduction="none").sum(-1)
            probabilities[start:start + len(values)] = torch.exp(logp).numpy()
    probabilities /= probabilities.sum()
    return float(0.5 * np.abs(exact - probabilities).sum())


def validate(model: GraphAutoregressiveSampler, rows: list[Workload], teacher_path: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    outcomes = np.load(teacher_path)["outcomes"]
    checkpoint_temp = ROOT / config["artifact_root"] / ".validation_model.pt"
    save_checkpoint(model, checkpoint_temp); backend = M3Backend.from_pretrained(checkpoint_temp)
    result = []
    for index, row in enumerate(rows):
        counts, metadata = backend.sample(row, config["validation_teacher_shots"], config["seed"] + 100000 + index)
        predicted = counts_to_outcomes(counts)
        metrics = outcome_metrics(outcomes[index], predicted, row)
        tvd = exact_tvd_18(model, row) if row.n_qubits == 18 and row.parameter_index == 0 else None
        result.append({"logical_id": row.logical_id, "circuit_hash": row.circuit_hash, "graph_family": row.graph_family,
                       "n_qubits": row.n_qubits, "p": row.p, **metrics, "exact_tvd": tvd,
                       "neural_total_ms_65536": metadata["timing_ms"]["total"]})
    checkpoint_temp.unlink(missing_ok=True)
    return result


def benchmark_worker(payload_path: Path, output_path: Path) -> None:
    payload = json.loads(payload_path.read_text()); row = workload_from_dict(payload["workload"])
    process = psutil.Process(); before_load = process.memory_info().rss
    backend = M3Backend.from_pretrained(payload["checkpoint"])
    after_load = process.memory_info().rss
    for seed in range(3): backend.sample(row, payload["shots"], seed)
    timings = []
    for seed in range(10):
        start = time.perf_counter_ns(); counts, metadata = backend.sample(row, payload["shots"], 100 + seed)
        timings.append((time.perf_counter_ns() - start) / 1e6); assert sum(counts.values()) == payload["shots"]
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != "Darwin": peak *= 1024
    write_json(output_path, {"latency_ms": timings, "baseline_rss_bytes": before_load, "model_load_rss_delta_bytes": max(0, after_load - before_load),
                             "peak_rss_bytes": peak, "incremental_peak_rss_bytes": max(0, peak - before_load), "metadata": metadata})


def systems_benchmark(checkpoint: Path, rows: list[Workload], config: dict[str, Any], out: Path) -> list[dict[str, Any]]:
    selected = [next(row for row in rows if row.graph_family == family and row.n_qubits == n and row.p == p and row.parameter_index == 0)
                for family in config["families"] for n in config["validation_widths"] for p in config["depths"]]
    result = []
    env = os.environ | {"PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    for index, row in enumerate(selected):
        payload = out / f".benchmark-{index}.json"; output = out / f".benchmark-{index}-out.json"
        write_json(payload, {"checkpoint": str(checkpoint), "workload": workload_dict(row), "shots": config["primary_shots"]})
        subprocess.run([sys.executable, "-m", "qute_r2.m3.run", "--benchmark-worker", str(payload), str(output)], cwd=ROOT, env=env, check=True, timeout=120)
        measured = json.loads(output.read_text()); values = measured["latency_ms"]
        result.append({"graph_family": row.graph_family, "n_qubits": row.n_qubits, "p": row.p, "shots": config["primary_shots"],
                       "median_latency_ms": statistics.median(values), "mean_latency_ms": statistics.mean(values),
                       "p90_latency_ms": float(np.percentile(values, 90)), "incremental_peak_rss_bytes": measured["incremental_peak_rss_bytes"],
                       "model_load_rss_delta_bytes": measured["model_load_rss_delta_bytes"],
                       "per_circuit_optimizer_steps": measured["metadata"]["per_circuit_optimizer_steps"],
                       "exact_calls": measured["metadata"]["exact_simulator_calls_on_neural_route"]})
        payload.unlink(missing_ok=True); output.unlink(missing_ok=True)
    return result


def aggregate_validation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ("energy_error_per_edge", "marginal_mae", "zz_mae")
    overall = {f"median_{key}": statistics.median(row[key] for row in rows) for key in metrics}
    tvd = [row["exact_tvd"] for row in rows if row["exact_tvd"] is not None]
    overall["median_exact_tvd_18q"] = statistics.median(tvd) if tvd else None
    breakdown = {}
    for n in sorted({row["n_qubits"] for row in rows}):
        selected = [row for row in rows if row["n_qubits"] == n]
        breakdown[str(n)] = {f"median_{key}": statistics.median(row[key] for row in selected) for key in metrics}
    return {"overall": overall, "by_width": breakdown}


def gate_decision(stage: str, validation: dict[str, Any], systems: list[dict[str, Any]], config: dict[str, Any], robustness: dict[str, Any] | None = None) -> tuple[str, dict[str, bool]]:
    gate = config["gates"]; overall = validation["overall"]
    by_width = validation["by_width"]
    latency_by_width = {n: statistics.median(row["median_latency_ms"] for row in systems if row["n_qubits"] == n) for n in config["validation_widths"]}
    checks = {
        "full_10000_circuit_corpus": stage == "full",
        "energy_error_per_edge": overall["median_energy_error_per_edge"] <= gate["energy_error_per_edge"],
        "marginal_mae": overall["median_marginal_mae"] <= gate["marginal_mae"],
        "zz_mae": overall["median_zz_mae"] <= gate["zz_mae"],
        "tvd_18q": overall["median_exact_tvd_18q"] is not None and overall["median_exact_tvd_18q"] <= gate["tvd"],
        "latency": max(row["median_latency_ms"] for row in systems) <= gate["latency_ms"],
        "memory": max(row["incremental_peak_rss_bytes"] for row in systems) <= gate["incremental_peak_mib"] * 2**20,
        "optimizer_steps_zero": all(row["per_circuit_optimizer_steps"] == 0 for row in systems),
        "exact_calls_zero": all(row["exact_calls"] == 0 for row in systems),
        "latency_width_trend": latency_by_width[24] <= gate["latency_24_over_20"] * latency_by_width[20],
        "energy_width_trend": by_width["24"]["median_energy_error_per_edge"] <= gate["energy_width_ratio"] * max(by_width["18"]["median_energy_error_per_edge"], 1e-9),
        "validation_boundary_24q": "24" in by_width,
    }
    if robustness is not None:
        for key in ("energy_error_per_edge", "marginal_mae", "zz_mae"):
            checks[f"robust_median_{key}"] = robustness["overall"][f"median_{key}"] <= gate[key]
            checks[f"robust_p90_{key}"] = robustness["overall"][f"p90_{key}"] <= gate[key]
    return ("M3_PASS_EXACT_REGIME" if all(checks.values()) else "M3_NEEDS_ITERATION"), checks


def report(out: Path, decision: str, checks: dict[str, bool], validation: dict[str, Any], robustness: dict[str, Any] | None, systems: list[dict[str, Any]], stage: str) -> str:
    text = f"""# M3 Exact-Regime Scale Gate Report

**Decision:** `{decision}`
**Training stage reached:** `{stage}`
**Role:** exploratory, simulator-only; no QPU authorization

## Gate checks

| Gate | Result |
|---|---|
{chr(10).join(f'| `{key}` | {"PASS" if value else "FAIL"} |' for key, value in checks.items())}

## Accuracy

```json
{json.dumps(validation, indent=2, sort_keys=True)}
```

## Confirmatory robustness

```json
{json.dumps(robustness, indent=2, sort_keys=True) if robustness else 'not run'}
```

## Systems

- Worst warm 4,096-shot median latency: {max(row['median_latency_ms'] for row in systems):.3f} ms.
- Maximum incremental peak RSS: {max(row['incremental_peak_rss_bytes'] for row in systems)/2**20:.2f} MiB.
- Per-circuit optimizer steps and neural-route exact calls remained zero.

## Boundary

Exact statevector teacher/validation completed through 24Q. M2 remains the classical boundary reference: statevector was resource-guarded at 26Q; MPS cycle reached 32Q, random 3-regular p=1 reached 28Q, and random 3-regular p=2/3 hit the 120-second guard at 20Q.

## Interpretation

A failed model gate means iterate inside M3; it does not justify QPU data. A pass means only that the exact-regime model is ready for the next simulator-side gate.
"""
    (ROOT / "docs/M3_EXACT_REGIME_REPORT.md").write_text(text); (out / "report.md").write_text(text)
    return text


def execute(config_path: Path) -> None:
    config = json.loads(config_path.read_text()); out = ROOT / config["artifact_root"]; out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["threads"]); os.environ["OMP_NUM_THREADS"] = str(config["threads"]); os.environ["MKL_NUM_THREADS"] = str(config["threads"])
    calibration = fixed_corpus(config, "calibration"); validation_rows = fixed_corpus(config, "validation")
    protocol_path = ROOT / config.get("protocol", "docs/M3_EXACT_REGIME_PROTOCOL.md")
    write_json(out / "protocol_freeze.json", {"config": config, "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "protocol_path": str(protocol_path.relative_to(ROOT)), "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "calibration_hashes": [row.circuit_hash for row in calibration], "validation_hashes": [row.circuit_hash for row in validation_rows]})
    write_json(out / "environment.json", {"platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(), "ram_bytes": psutil.virtual_memory().total, "threads": config["threads"]})
    calibration_path = out / "calibration_teacher.npz"
    calibration_manifest = teacher_dataset(calibration, config["teacher_shots_per_circuit"], calibration_path, out / "calibration_manifest.json", config)
    model, history = train_model(calibration, calibration_path, config, full=False)
    calibration_checkpoint = out / "m3_calibration.pt"; save_checkpoint(model, calibration_checkpoint)
    validation_path = out / "validation_teacher.npz"
    teacher_dataset(validation_rows, config["validation_teacher_shots"], validation_path, out / "validation_manifest.json", config)
    validation_metrics = validate(model, validation_rows, validation_path, config); validation_summary = aggregate_validation(validation_metrics)
    calibration_systems = systems_benchmark(calibration_checkpoint, validation_rows, config, out)
    viability = (math.isfinite(history[-1]["mean_loss"]) and validation_summary["overall"]["median_energy_error_per_edge"] <= config["viability"]["energy_error_per_edge"]
                 and max(row["median_latency_ms"] for row in calibration_systems) <= config["viability"]["latency_ms"])
    stage = "calibration"
    if viability:
        full_rows = full_corpus(config)
        full_path = out / "full_teacher_10000.npz"
        teacher_dataset(full_rows, config["teacher_shots_per_circuit"], full_path, out / "full_manifest_10000.json", config)
        model, full_history = train_model(full_rows, full_path, config, full=True); history.extend({"stage": "full", **row} for row in full_history)
        if config.get("targeted_circuits"):
            target_rows = targeted_corpus(config); target_path = out / "targeted_teacher.npz"
            teacher_dataset(target_rows, config["targeted_teacher_shots"], target_path, out / "targeted_manifest.json", config)
            if config.get("targeted_mix_with_full"):
                common = min(np.load(full_path)["outcomes"].shape[1], np.load(target_path)["outcomes"].shape[1])
                mixed_path = out / "mixed_teacher.npz"
                np.savez_compressed(mixed_path, outcomes=np.concatenate((np.load(full_path)["outcomes"][:, :common], np.load(target_path)["outcomes"][:, :common])))
                fine_tune_rows = full_rows + target_rows
            else:
                mixed_path = target_path; fine_tune_rows = target_rows
            model, target_history = train_model(fine_tune_rows, mixed_path, config, full=True, model=model, epochs_override=config["targeted_epochs"])
            history.extend({"stage": "targeted_mixed" if config.get("targeted_mix_with_full") else "targeted", **row} for row in target_history)
        checkpoint = out / "m3_exact_regime.pt"; save_checkpoint(model, checkpoint)
        validation_metrics = validate(model, validation_rows, validation_path, config); validation_summary = aggregate_validation(validation_metrics)
        systems = systems_benchmark(checkpoint, validation_rows, config, out); stage = "full"
    else:
        checkpoint = calibration_checkpoint; systems = calibration_systems
    robustness_summary = None
    if config.get("robustness_graph_seeds"):
        robust_rows = robustness_corpus(config); robust_path = out / "robustness_teacher.npz"
        teacher_dataset(robust_rows, config["validation_teacher_shots"], robust_path, out / "robustness_manifest.json", config)
        robust_metrics = validate(model, robust_rows, robust_path, config); robustness_summary = aggregate_validation(robust_metrics)
        for key in ("energy_error_per_edge", "marginal_mae", "zz_mae"):
            values = np.asarray([row[key] for row in robust_metrics])
            robustness_summary["overall"][f"p90_{key}"] = float(np.percentile(values, 90))
            robustness_summary["overall"][f"maximum_{key}"] = float(values.max())
        write_csv(out / "robustness_metrics.csv", robust_metrics); write_json(out / "robustness_summary.json", robustness_summary)
    decision_value, checks = gate_decision(stage, validation_summary, systems, config, robustness_summary)
    write_json(out / "training_history.json", {"stage": stage, "viability_screen_pass": viability, "history": history})
    write_csv(out / "validation_metrics.csv", validation_metrics); write_json(out / "validation_summary.json", validation_summary)
    write_csv(out / "systems_benchmark.csv", systems)
    write_json(out / "validation_boundary.json", {"exact_statevector_validated_through_qubits": 24, "m2_statevector_resource_guard_qubits": 26,
        "m2_mps_cycle_measured_through_qubits": 32, "m2_mps_random_p1_measured_through_qubits": 28,
        "m2_mps_random_p2_p3_timeout_qubits": 20})
    write_json(out / "M3_DECISION.json", {"decision": decision_value, "stage": stage, "checks": checks, "qpu_authorized": False,
        "checkpoint": checkpoint.name, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
    report(out, decision_value, checks, validation_summary, robustness_summary, systems, stage)
    print(json.dumps({"decision": decision_value, "stage": stage, "viability": viability, "validation": validation_summary["overall"],
                      "worst_latency_ms": max(row["median_latency_ms"] for row in systems)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, default=ROOT / "configs/r2/m3_exact_regime_scale_gate.yaml")
    parser.add_argument("--benchmark-worker", nargs=2); args = parser.parse_args()
    if args.benchmark_worker: benchmark_worker(Path(args.benchmark_worker[0]), Path(args.benchmark_worker[1]))
    else: execute(args.config)


if __name__ == "__main__": main()
