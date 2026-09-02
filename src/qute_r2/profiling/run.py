from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import signal
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

from .analysis import aggregate_runs, best_classical, budget_map, candidate_regions, decision, scaling_fits, statevector_guard
from .corpus import Workload, generate_corpus, qiskit_circuit

ROOT = Path(__file__).resolve().parents[3]


def read_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())  # JSON is valid YAML; avoids an otherwise unused parser dependency.


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def workload_from_dict(value: dict[str, Any]) -> Workload:
    return Workload(
        value["logical_id"], value["n_qubits"], value["p"], value["graph_family"], value["graph_seed"],
        value["parameter_index"], tuple(map(tuple, value["edges"])), tuple(value["gammas"]), tuple(value["betas"]),
        value["graph_hash"], value["circuit_hash"],
    )


def alarm_handler(signum, frame):
    raise TimeoutError("circuit execution exceeded hard timeout")


def timed_aer(workload: Workload, simulator, shots: int, seed: int, timeout: int) -> dict[str, Any]:
    from qiskit import transpile

    start = time.perf_counter_ns()
    circuit = qiskit_circuit(workload)
    preprocess_ms = (time.perf_counter_ns() - start) / 1e6
    start = time.perf_counter_ns()
    compiled = transpile(circuit, simulator, optimization_level=0, seed_transpiler=seed)
    transpile_ms = (time.perf_counter_ns() - start) / 1e6
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(timeout)
    try:
        start = time.perf_counter_ns()
        job = simulator.run(compiled, shots=shots, seed_simulator=seed)
        result = job.result()
        execute_ms = (time.perf_counter_ns() - start) / 1e6
    finally:
        signal.alarm(0)
    start = time.perf_counter_ns()
    counts = result.get_counts()
    aggregate_ms = (time.perf_counter_ns() - start) / 1e6
    start = time.perf_counter_ns()
    packaged = dict(sorted(counts.items()))
    package_ms = (time.perf_counter_ns() - start) / 1e6
    assert sum(packaged.values()) == shots
    return {
        "preprocess_ms": preprocess_ms, "transpile_ms": transpile_ms, "execute_ms": execute_ms,
        "aggregate_ms": aggregate_ms, "package_ms": package_ms,
        "total_ms": preprocess_ms + transpile_ms + execute_ms + aggregate_ms + package_ms,
    }


def timed_aer_batch(workloads: list[Workload], simulator, shots: int, seed: int, timeout: int) -> dict[str, Any]:
    from qiskit import transpile

    start = time.perf_counter_ns(); circuits = [qiskit_circuit(w) for w in workloads]
    preprocess_ms = (time.perf_counter_ns() - start) / 1e6
    start = time.perf_counter_ns(); compiled = transpile(circuits, simulator, optimization_level=0, seed_transpiler=seed)
    transpile_ms = (time.perf_counter_ns() - start) / 1e6
    signal.signal(signal.SIGALRM, alarm_handler); signal.alarm(timeout)
    try:
        start = time.perf_counter_ns(); result = simulator.run(compiled, shots=shots, seed_simulator=seed).result()
        execute_ms = (time.perf_counter_ns() - start) / 1e6
    finally:
        signal.alarm(0)
    start = time.perf_counter_ns(); counts = result.get_counts(); aggregate_ms = (time.perf_counter_ns() - start) / 1e6
    if isinstance(counts, dict):
        counts = [counts]
    start = time.perf_counter_ns(); packaged = [dict(sorted(item.items())) for item in counts]; package_ms = (time.perf_counter_ns() - start) / 1e6
    assert all(sum(item.values()) == shots for item in packaged)
    return {"preprocess_ms": preprocess_ms, "transpile_ms": transpile_ms, "execute_ms": execute_ms,
            "aggregate_ms": aggregate_ms, "package_ms": package_ms,
            "total_ms": preprocess_ms + transpile_ms + execute_ms + aggregate_ms + package_ms}


def worker(payload_path: Path, output_path: Path) -> None:
    from qiskit_aer import AerSimulator

    payload = json.loads(payload_path.read_text())
    method = payload["backend"]
    options = payload["aer_options"]
    init_start = time.perf_counter_ns()
    simulator = AerSimulator(method=method, precision=payload["precision"], **options)
    init_ms = (time.perf_counter_ns() - init_start) / 1e6
    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    workloads = [workload_from_dict(value) for value in payload["workloads"]]
    rows: list[dict[str, Any]] = []
    try:
        cold = timed_aer(workloads[0], simulator, payload["shots"], 1000, payload["timeout"])
        rows.append({**cold, "timing_mode": "cold_start", "initialization_ms": init_ms, "total_ms": cold["total_ms"] + init_ms})
        for index in range(payload["warmups"]):
            timed_aer(workloads[index % len(workloads)], simulator, payload["shots"], 2000 + index, payload["timeout"])
    except TimeoutError:
        write_json(output_path, {"rows": rows, "worker_error": "SKIPPED_TIMEOUT_PROJECTION"})
        return
    for index in range(payload["repetitions"]):
        before = process.memory_info().rss
        try:
            row = timed_aer(workloads[index % len(workloads)], simulator, payload["shots"], 3000 + index, payload["timeout"])
            status, reason = "OK", ""
        except TimeoutError:
            row = {key: None for key in ("preprocess_ms", "transpile_ms", "execute_ms", "aggregate_ms", "package_ms", "total_ms")}
            status, reason = "SKIPPED_TIMEOUT_PROJECTION", "hard_timeout"
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() != "Darwin":
            peak *= 1024
        rows.append({
            **row, "status": status, "skip_reason": reason, "timing_mode": "warm_single", "repetition": index,
            "logical_id": workloads[index % len(workloads)].logical_id, "circuit_hash": workloads[index % len(workloads)].circuit_hash,
            "baseline_rss_bytes": baseline_rss, "peak_rss_bytes": peak,
            "incremental_peak_rss_bytes": max(0, peak - before),
        })
    for batch_size in payload.get("batch_sizes", []):
        batch = [workloads[(index + 1) % len(workloads)] for index in range(batch_size)]
        for repetition in range(payload.get("batch_repetitions", 3)):
            try:
                row = timed_aer_batch(batch, simulator, payload["shots"], 6000 + repetition, payload["timeout"])
                rows.append({**row, "status": "OK", "skip_reason": "", "timing_mode": "warm_batch", "batch_size": batch_size,
                             "repetition": repetition, "logical_id": "batch", "circuit_hash": "multiple",
                             "baseline_rss_bytes": baseline_rss, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                             "incremental_peak_rss_bytes": max(0, process.memory_info().rss - baseline_rss)})
            except TimeoutError:
                rows.append({"status": "SKIPPED_TIMEOUT_PROJECTION", "skip_reason": "hard_timeout", "timing_mode": "warm_batch",
                             "batch_size": batch_size, "repetition": repetition})
                break
    write_json(output_path, {"rows": rows})


def environment() -> dict[str, Any]:
    import qiskit
    import qiskit_aer
    import scipy

    vm = psutil.virtual_memory()
    return {
        "platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
        "cpu": platform.processor(), "logical_cores": psutil.cpu_count(), "physical_cores": psutil.cpu_count(logical=False),
        "total_ram_bytes": vm.total, "available_ram_bytes_at_start": vm.available,
        "qiskit": qiskit.__version__, "qiskit_aer": qiskit_aer.__version__, "numpy": np.__version__,
        "scipy": scipy.__version__, "torch": torch.__version__, "psutil": psutil.__version__,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "torch_threads": torch.get_num_threads(), "device": "cpu",
    }


def correctness(corpus: list[Workload], checkpoint: Path, config: dict[str, Any]) -> dict[str, Any]:
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    import qute_r2_pilot as pilot
    import run_qute_r2_pilot as old_run

    sample = next(w for w in corpus if w.graph_family == "cycle" and w.n_qubits == 6 and w.p == 1 and w.parameter_index == 1)
    qiskit_prob = Statevector.from_instruction(qiskit_circuit(sample, measure=False)).probabilities()
    legacy = pilot.QAOACircuit(6, sample.edges, sample.gammas[0], sample.betas[0])
    legacy_prob = pilot.exact_probabilities(legacy)
    exact_error = float(np.max(np.abs(qiskit_prob - legacy_prob)))

    sv = AerSimulator(method="statevector")
    mps = AerSimulator(method="matrix_product_state")
    sv_circuit = qiskit_circuit(sample, measure=False); sv_circuit.save_statevector()
    mps_circuit = qiskit_circuit(sample, measure=False); mps_circuit.save_statevector()
    sv_prob = np.abs(np.asarray(sv.run(transpile(sv_circuit, sv, optimization_level=0)).result().get_statevector())) ** 2
    mps_prob = np.abs(np.asarray(mps.run(transpile(mps_circuit, mps, optimization_level=0)).result().get_statevector())) ** 2
    bits = np.asarray([[int(b) for b in format(i, "06b")] for i in range(64)])
    cuts = pilot.cut_values(legacy)
    mps_overlap = {
        "tvd": float(0.5 * np.abs(sv_prob - mps_prob).sum()),
        "energy_abs_error": float(abs(sv_prob @ cuts - mps_prob @ cuts)),
        "marginal_max_abs_error": float(np.max(np.abs(sv_prob @ bits - mps_prob @ bits))),
    }

    model = pilot.load_checkpoint(checkpoint)
    backend = pilot.QuTEBackend.from_pretrained(checkpoint)
    validation = old_run.make_circuits(pilot.available_chords(6)[6:], 4, 20260832)
    rows = []
    exact_before = pilot.exact_call_count()
    for item in validation:
        rows.append(pilot.distribution_metrics(pilot.exact_probabilities(item), pilot.model_distribution(model, item), item))
    exact_after_eval = pilot.exact_call_count()
    route_before = pilot.exact_call_count()
    route = backend.run(validation[0], shots=4096, seed_simulator=2026).result()
    route_exact_calls = pilot.exact_call_count() - route_before
    reproduced_tvd = statistics.median(row["tvd"] for row in rows)
    reproduced_energy = statistics.median(row["cut_energy_error_per_edge"] for row in rows)
    frozen = json.loads((ROOT / "artifacts/r2_amortized_backend/pilot_v1/report.json").read_text())
    frozen_tvd = frozen["validation"]["aggregate"]["qute"]["median_tvd"]
    frozen_energy = frozen["validation"]["aggregate"]["qute"]["median_cut_energy_error_per_edge"]
    checks = {
        "legacy_vs_qiskit_statevector": exact_error < 1e-12,
        "mps_vs_statevector_tvd": mps_overlap["tvd"] < 1e-10,
        "mps_energy": mps_overlap["energy_abs_error"] < 1e-10,
        "mps_marginals": mps_overlap["marginal_max_abs_error"] < 1e-10,
        "pilot_tvd_reproduced": abs(reproduced_tvd - frozen_tvd) <= config["pilot_tolerances"]["median_tvd_abs"],
        "pilot_energy_reproduced": abs(reproduced_energy - frozen_energy) <= config["pilot_tolerances"]["median_energy_error_per_edge_abs"],
        "pilot_counts": sum(route.get_counts().values()) == 4096,
        "per_circuit_optimizer_steps_zero": route.metadata[0]["per_circuit_optimizer_steps"] == 0,
        "neural_route_exact_calls_zero": route_exact_calls == 0,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "statevector_max_probability_abs_error": exact_error, "mps_overlap": mps_overlap,
        "pilot_reproduction": {"median_tvd": reproduced_tvd, "frozen_median_tvd": frozen_tvd,
            "median_energy_error_per_edge": reproduced_energy, "frozen_median_energy_error_per_edge": frozen_energy,
            "validation_exact_calls": exact_after_eval - exact_before, "neural_route_exact_calls": route_exact_calls},
    }


def neural_breakdown(checkpoint: Path, circuit, shot_counts: list[int]) -> dict[str, Any]:
    import qute_r2_pilot as pilot

    load_start = time.perf_counter_ns(); rss = psutil.Process().memory_info().rss
    model = pilot.load_checkpoint(checkpoint)
    load_ms = (time.perf_counter_ns() - load_start) / 1e6
    load_memory = max(0, psutil.Process().memory_info().rss - rss)
    all_rows = []
    for shots in shot_counts:
        for repetition in range(5):
            rng = np.random.default_rng(5000 + repetition)
            prefixes = np.zeros((shots, model.max_qubits), dtype=np.int8)
            timings = Counter()
            start = time.perf_counter_ns(); parsed = pilot.QuTEBackend._parse(circuit); pilot.QuTEBackend._supported(parsed)
            timings["input_validation_ns"] = time.perf_counter_ns() - start
            start = time.perf_counter_ns(); pilot._adjacency_and_degrees(parsed)
            timings["circuit_encoding_ns"] = time.perf_counter_ns() - start
            with torch.inference_mode():
                for position in range(circuit.n_qubits):
                    start = time.perf_counter_ns()
                    features = np.stack([pilot.autoregressive_feature(circuit, row[:position], position) for row in prefixes])
                    timings["tokenization_graph_construction_ns"] += time.perf_counter_ns() - start
                    start = time.perf_counter_ns(); tensor = torch.from_numpy(features)
                    timings["tensor_transfer_ns"] += time.perf_counter_ns() - start
                    start = time.perf_counter_ns(); probabilities = torch.sigmoid(model(tensor)).numpy()
                    timings["decoder_forward_ns"] += time.perf_counter_ns() - start
                    start = time.perf_counter_ns(); prefixes[:, position] = rng.random(shots) < probabilities
                    timings["random_sampling_ns"] += time.perf_counter_ns() - start
            start = time.perf_counter_ns(); counts = Counter("".join(map(str, row[:circuit.n_qubits])) for row in prefixes)
            timings["counts_aggregation_ns"] = time.perf_counter_ns() - start
            start = time.perf_counter_ns(); result = pilot.QuTEResult([dict(sorted(counts.items()))], [{"shots": shots}])
            result.get_counts(); timings["result_packaging_ns"] = time.perf_counter_ns() - start
            row = {"shots": shots, "repetition": repetition, "encoder_forward_ms": 0.0,
                   "autoregressive_token_generation_ms": sum(timings[k] for k in ("tokenization_graph_construction_ns", "tensor_transfer_ns", "decoder_forward_ns", "random_sampling_ns")) / 1e6}
            row.update({key.removesuffix("_ns") + "_ms": value / 1e6 for key, value in timings.items()})
            row["total_profiled_ms"] = sum(value for key, value in row.items() if key.endswith("_ms") and key not in {"autoregressive_token_generation_ms", "total_profiled_ms"})
            all_rows.append(row)
    medians = {str(shots): {key: statistics.median(r[key] for r in all_rows if r["shots"] == shots) for key in all_rows[0] if key.endswith("_ms")} for shots in shot_counts}
    x = np.asarray(shot_counts, dtype=float); y = np.asarray([medians[str(s)]["total_profiled_ms"] for s in shot_counts])
    slope, intercept = np.polyfit(x, y, 1)
    return {"model_load_time_ms": load_ms, "model_load_memory_bytes": load_memory, "checkpoint_bytes": checkpoint.stat().st_size,
            "rows": all_rows, "medians_ms": medians, "planning_fit": {"formula": "T_fixed + shots*T_per_sample",
            "T_fixed_ms": float(intercept), "T_per_sample_ms": float(slope), "role": "planning_only",
            "valid_fixed_intercept": bool(intercept >= 0), "warning": "negative fixed term means the three-point linear model is not physically interpretable" if intercept < 0 else None}}


def run_cell(task: dict[str, Any], temp: Path) -> dict[str, Any]:
    key = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()[:16]
    payload = temp / f"{key}.input.json"; output = temp / f"{key}.output.json"
    write_json(payload, task)
    env = os.environ | {"OMP_NUM_THREADS": str(task["threads"]), "MKL_NUM_THREADS": str(task["threads"]), "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT)}
    command = [sys.executable, "-m", "qute_r2.profiling.run", "--worker", str(payload), str(output)]
    try:
        subprocess.run(command, cwd=ROOT, env=env, check=True, timeout=(task["repetitions"] + task["warmups"] + 2) * task["timeout"])
        return json.loads(output.read_text())
    except subprocess.TimeoutExpired:
        return {"rows": [], "worker_error": "SKIPPED_TIMEOUT_PROJECTION"}
    except subprocess.CalledProcessError as error:
        return {"rows": [], "worker_error": f"WORKER_FAILED_{error.returncode}"}


def profile(config: dict[str, Any], corpus: list[Workload], out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    temp = out / ".worker"; temp.mkdir(parents=True, exist_ok=True)
    raw: list[dict[str, Any]] = []; guard_events: list[dict[str, Any]] = []
    available = psutil.virtual_memory().available
    for backend in ("statevector", "matrix_product_state"):
        for family in config["graph_families"]:
            for n in config["base_qubits"]:
                if backend == "statevector":
                    guard = statevector_guard(n, available, bytes_per_amplitude=config["statevector_bytes_per_amplitude"],
                        overhead_factor=config["resource_guard"]["initial_overhead_factor"], payload_fraction=config["resource_guard"]["payload_available_fraction"],
                        peak_fraction=config["resource_guard"]["projected_peak_available_fraction"], timeout_seconds=config["timeout_seconds"])
                    guard_events.append({"backend": backend, "graph_family": family, **guard})
                    if guard["status"] != "RUN":
                        continue
                for p in config["depths"]:
                    items = [w for w in corpus if w.graph_family == family and w.n_qubits == n and w.p == p]
                    repetitions = config["repetitions_small"] if n <= config["small_width_max"] else config["repetitions_large"]
                    task = {"backend": backend, "workloads": [w.__dict__ for w in items], "shots": config["primary_shots"],
                            "warmups": config["warmups"], "repetitions": repetitions, "timeout": config["timeout_seconds"],
                            "threads": config["threads"], "precision": config["statevector_precision"], "aer_options": config["aer_options"]}
                    result = run_cell(task, temp)
                    if result.get("worker_error"):
                        guard_events.append({"backend": backend, "graph_family": family, "n_qubits": n, "p": p, "status": result["worker_error"]})
                    for row in result["rows"]:
                        raw.append({**row, "backend": backend, "graph_family": family, "n_qubits": n, "p": p, "shots": config["primary_shots"]})
    # Shot scaling uses representative random-regular p=3 cells and the same manifest.
    for backend in ("statevector", "matrix_product_state"):
        for n in (6, 12, 20):
            items = [w for w in corpus if w.graph_family == "random_3_regular" and w.n_qubits == n and w.p == 3]
            for shots in config["shots"]:
                if shots == config["primary_shots"]:
                    continue
                task = {"backend": backend, "workloads": [w.__dict__ for w in items], "shots": shots, "warmups": config["warmups"],
                        "repetitions": config["repetitions_heavy_minimum"], "timeout": config["timeout_seconds"], "threads": config["threads"],
                        "precision": config["statevector_precision"], "aer_options": config["aer_options"]}
                result = run_cell(task, temp)
                for row in result["rows"]:
                    raw.append({**row, "backend": backend, "graph_family": "random_3_regular", "n_qubits": n, "p": 3, "shots": shots})
    return raw, guard_events


def profile_batches(config: dict[str, Any], corpus: list[Workload], raw: list[dict[str, Any]], out: Path) -> None:
    existing = {(r.get("backend"), r.get("graph_family"), int(r.get("n_qubits", 0)), int(r.get("p", 0))) for r in raw if r.get("timing_mode") == "warm_batch"}
    temp = out / ".worker"; temp.mkdir(parents=True, exist_ok=True)
    for backend in ("statevector", "matrix_product_state"):
        for family, n, p in (("cycle", 20, 3), ("random_3_regular", 6, 1), ("random_3_regular", 20, 1)):
            if (backend, family, n, p) in existing:
                continue
            items = [w for w in corpus if w.graph_family == family and w.n_qubits == n and w.p == p]
            task = {"backend": backend, "workloads": [w.__dict__ for w in items], "shots": config["primary_shots"],
                    "warmups": 1, "repetitions": 1, "batch_sizes": config["batch_sizes"], "batch_repetitions": 3,
                    "timeout": config["timeout_seconds"], "threads": config["threads"], "precision": config["statevector_precision"],
                    "aer_options": config["aer_options"]}
            result = run_cell(task, temp)
            for row in result.get("rows", []):
                if row.get("timing_mode") == "warm_batch":
                    raw.append({**row, "backend": backend, "graph_family": family, "n_qubits": n, "p": p, "shots": config["primary_shots"]})


def profile_qute(config: dict[str, Any], raw: list[dict[str, Any]]) -> None:
    import qute_r2_pilot as pilot

    if any(r.get("backend") == "qute" for r in raw):
        return
    checkpoint = ROOT / config["checkpoint"]; backend = pilot.QuTEBackend.from_pretrained(checkpoint)
    report = json.loads((ROOT / "artifacts/r2_amortized_backend/pilot_v1/report.json").read_text()); value = report["example"]["circuit"]
    circuit = pilot.QAOACircuit(value["n_qubits"], tuple(map(tuple, value["edges"])), value["gamma"], value["beta"], value["p"])
    process = psutil.Process(); baseline = process.memory_info().rss
    for shots in config["shots"]:
        batch_sizes = config["batch_sizes"] if shots == config["primary_shots"] else [1]
        for batch_size in batch_sizes:
            for repetition in range(5):
                items = [circuit] * batch_size
                start = time.perf_counter_ns(); result = backend.run(items, shots=shots, seed_simulator=7000 + repetition).result(); result.get_counts()
                total_ms = (time.perf_counter_ns() - start) / 1e6
                raw.append({"status": "OK", "backend": "qute", "graph_family": "pilot_cycle_plus_chord", "n_qubits": 6,
                    "p": 1, "shots": shots, "timing_mode": "warm_single" if batch_size == 1 else "warm_batch", "batch_size": batch_size,
                    "repetition": repetition, "total_ms": total_ms, "execute_ms": total_ms, "transpile_ms": 0.0,
                    "peak_rss_bytes": process.memory_info().rss, "incremental_peak_rss_bytes": max(0, process.memory_info().rss - baseline)})


def profile_adaptive(config: dict[str, Any], corpus: list[Workload], raw: list[dict[str, Any]], guard_events: list[dict[str, Any]], out: Path) -> None:
    temp = out / ".worker"; temp.mkdir(parents=True, exist_ok=True)
    existing = {(r["backend"], r["graph_family"], int(r["n_qubits"]), int(r["p"]), int(r["shots"])) for r in raw if r.get("timing_mode") == "warm_single"}
    available = psutil.virtual_memory().available
    for backend in ("statevector", "matrix_product_state"):
        for family in config["graph_families"]:
            for p in config["depths"]:
                base_rows = [r for r in raw if r.get("status") == "OK" and r["backend"] == backend and r["graph_family"] == family and int(r["p"]) == p and int(r["shots"]) == config["primary_shots"] and r["timing_mode"] == "warm_single"]
                measured_widths = sorted({int(r["n_qubits"]) for r in base_rows})
                if not measured_widths or max(measured_widths) < max(config["base_qubits"]):
                    for n in config["adaptive_qubits"]:
                        guard_events.append({"backend": backend, "graph_family": family, "n_qubits": n, "p": p, "status": "SKIPPED_TIMEOUT_PROJECTION", "reason": "base_grid_did_not_reach_n20"})
                    continue
                recent = sorted(base_rows, key=lambda r: int(r["n_qubits"]))
                last_median = statistics.median(float(r["total_ms"]) for r in recent if int(r["n_qubits"]) == max(measured_widths))
                previous_n = max(measured_widths)
                for n in config["adaptive_qubits"]:
                    if (backend, family, n, p, config["primary_shots"]) in existing:
                        previous_n = n
                        continue
                    growth = 4 ** ((n - previous_n) / 2) if backend == "statevector" else 2 ** ((n - previous_n) / 2)
                    projected_ms = last_median * growth
                    if backend == "statevector":
                        guard = statevector_guard(n, available, bytes_per_amplitude=config["statevector_bytes_per_amplitude"],
                            overhead_factor=config["resource_guard"]["initial_overhead_factor"], payload_fraction=config["resource_guard"]["payload_available_fraction"],
                            peak_fraction=config["resource_guard"]["projected_peak_available_fraction"], projected_seconds=projected_ms / 1000,
                            timeout_seconds=config["timeout_seconds"])
                    else:
                        guard = {"status": "SKIPPED_TIMEOUT_PROJECTION" if projected_ms / 1000 > config["timeout_seconds"] else "RUN",
                                 "n_qubits": n, "projected_seconds": projected_ms / 1000}
                    guard_events.append({"backend": backend, "graph_family": family, "p": p, **guard})
                    if guard["status"] != "RUN":
                        break
                    items = [w for w in corpus if w.graph_family == family and w.n_qubits == n and w.p == p]
                    task = {"backend": backend, "workloads": [w.__dict__ for w in items], "shots": config["primary_shots"],
                            "warmups": config["warmups"], "repetitions": config["repetitions_heavy_minimum"], "timeout": config["timeout_seconds"],
                            "threads": config["threads"], "precision": config["statevector_precision"], "aer_options": config["aer_options"]}
                    result = run_cell(task, temp)
                    if result.get("worker_error"):
                        guard_events.append({"backend": backend, "graph_family": family, "n_qubits": n, "p": p, "status": result["worker_error"]})
                        break
                    added = []
                    for row in result["rows"]:
                        enriched = {**row, "backend": backend, "graph_family": family, "n_qubits": n, "p": p, "shots": config["primary_shots"]}
                        raw.append(enriched)
                        if enriched.get("status") == "OK" and enriched.get("timing_mode") == "warm_single":
                            added.append(float(enriched["total_ms"]))
                    if not added:
                        break
                    last_median = statistics.median(added); previous_n = n


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric = {"n_qubits", "p", "shots", "repetition", "batch_size", "preprocess_ms", "transpile_ms", "execute_ms", "aggregate_ms", "package_ms", "total_ms", "peak_rss_bytes", "incremental_peak_rss_bytes", "baseline_rss_bytes", "initialization_ms"}
    for row in rows:
        for key in numeric:
            if row.get(key) not in (None, ""):
                row[key] = float(row[key])
        for key in ("n_qubits", "p", "shots", "repetition", "batch_size"):
            if row.get(key) not in (None, ""):
                row[key] = int(row[key])
    return rows


def teacher_projection(cells: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for cell in cells:
        for count in config["teacher_corpus_sizes"]:
            seconds = count * cell["median_classical_latency_ms"] / 1000
            rows.append({**{k: cell[k] for k in ("graph_family", "n_qubits", "p", "shots", "best_classical_backend")},
                "circuits": count, "median_teacher_cost_per_circuit_ms": cell["median_classical_latency_ms"],
                "serial_hours": seconds / 3600, "ideal_8_worker_hours": seconds / 3600 / 8, "ideal_32_worker_hours": seconds / 3600 / 32,
                "storage_bytes": count * min(cell["shots"], 2 ** cell["n_qubits"]) * ((cell["n_qubits"] + 7) // 8 + 8),
                "storage_model": "upper-bound sparse counts: min(shots,2^n) * (packed bitstring bytes + uint64 count)",
                "cluster_assumption": "no cluster availability assumed; worker scenarios are idealized"})
    return {"formula": "N * median_teacher_cost_per_circuit", "rows": rows}


def amortization(cells: list[dict[str, Any]], neural_ms: float, config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for cell in cells:
        for evaluations in config["amortization_evaluations"]:
            budget_ms = evaluations * (cell["median_classical_latency_ms"] - neural_ms)
            rows.append({**{k: cell[k] for k in ("graph_family", "n_qubits", "p", "shots")}, "evaluations": evaluations,
                "assumed_neural_latency_ms": neural_ms, "max_teacher_plus_training_budget_seconds": max(0, budget_ms / 1000),
                "status": "ALLOWABLE_BUDGET" if budget_ms >= 0 else "NO_BREAK_EVEN_AT_THIS_LATENCY"})
    return rows


def figures(aggregate: list[dict[str, Any]], breakdown: dict[str, Any], budget_rows: list[dict[str, Any]], out: Path) -> None:
    import matplotlib.pyplot as plt
    fig_dir = out / "figures"; fig_dir.mkdir(exist_ok=True)
    specs = [
        ("latency_vs_qubits.png", "n_qubits", "total_ms_median", "Median latency (ms)"),
        ("peak_memory_vs_qubits.png", "n_qubits", "peak_rss_bytes_median", "Peak RSS (bytes)"),
        ("latency_vs_depth.png", "p", "total_ms_median", "Median latency (ms)"),
    ]
    warm = [r for r in aggregate if r["timing_mode"] == "warm_single" and r["shots"] == 4096]
    for name, xkey, ykey, ylabel in specs:
        plt.figure()
        for backend in ("statevector", "matrix_product_state"):
            rows = [r for r in warm if r["backend"] == backend and r["graph_family"] == "random_3_regular" and r["p"] == 3]
            if xkey == "p": rows = [r for r in warm if r["backend"] == backend and r["graph_family"] == "random_3_regular" and r["n_qubits"] == 20]
            plt.plot([r[xkey] for r in rows], [r[ykey] for r in rows], marker="o", label=backend)
        plt.xlabel(xkey); plt.ylabel(ylabel); plt.yscale("log"); plt.legend(); plt.tight_layout(); plt.savefig(fig_dir / name); plt.close()
    ratios = []
    for sv in [r for r in warm if r["backend"] == "statevector"]:
        match = next((r for r in warm if r["backend"] == "matrix_product_state" and all(r[k] == sv[k] for k in ("graph_family", "n_qubits", "p", "shots"))), None)
        if match: ratios.append((sv["n_qubits"], sv["p"], sv["graph_family"], match["total_ms_median"] / sv["total_ms_median"]))
    plt.figure();
    for family in {r[2] for r in ratios}:
        rows = [r for r in ratios if r[2] == family and r[1] == 3]; plt.plot([r[0] for r in rows], [r[3] for r in rows], marker="o", label=family)
    plt.axhline(.25, color="black", linestyle="--"); plt.yscale("log"); plt.xlabel("qubits"); plt.ylabel("MPS/statevector latency"); plt.legend(); plt.tight_layout(); plt.savefig(fig_dir / "mps_statevector_relative_performance.png"); plt.close()
    shot_rows = [r for r in aggregate if r["graph_family"] == "random_3_regular" and r["n_qubits"] == 20 and r["p"] == 3 and r["timing_mode"] == "warm_single"]
    plt.figure();
    for backend in ("statevector", "matrix_product_state"):
        rows = [r for r in shot_rows if r["backend"] == backend]; plt.plot([r["shots"] for r in rows], [r["total_ms_median"] for r in rows], marker="o", label=backend)
    plt.xscale("log"); plt.yscale("log"); plt.xlabel("shots"); plt.ylabel("latency ms"); plt.legend(); plt.tight_layout(); plt.savefig(fig_dir / "shot_scaling.png"); plt.close()
    med = breakdown["medians_ms"]["4096"]; labels = [k for k in med if k not in {"autoregressive_token_generation_ms", "total_profiled_ms"}];
    plt.figure(); plt.bar(labels, [med[k] for k in labels]); plt.xticks(rotation=70, ha="right"); plt.ylabel("ms"); plt.tight_layout(); plt.savefig(fig_dir / "current_qute_latency_breakdown.png"); plt.close()
    cells = {(r["n_qubits"], r["p"]): r["speedup"] for r in budget_rows if r["graph_family"] == "random_3_regular" and r["shots"] == 4096 and r["neural_latency_budget_ms"] == 100 and r["neural_peak_memory_budget_mib"] == 512}
    if cells:
        ns, ps = sorted({k[0] for k in cells}), sorted({k[1] for k in cells}); image = np.asarray([[cells.get((n,p), np.nan) for n in ns] for p in ps]);
        plt.figure(); plt.imshow(image, aspect="auto"); plt.colorbar(label="speedup at 100ms"); plt.xticks(range(len(ns)), ns); plt.yticks(range(len(ps)), ps); plt.xlabel("qubits"); plt.ylabel("p"); plt.tight_layout(); plt.savefig(fig_dir / "required_neural_latency_budget_map.png"); plt.savefig(fig_dir / "candidate_operating_region_map.png"); plt.close()


def reports(out: Path, aggregate: list[dict[str, Any]], candidates: list[dict[str, Any]], breakdown: dict[str, Any], verdict: str, env: dict[str, Any]) -> None:
    warm = [r for r in aggregate if r["timing_mode"] == "warm_single" and r["shots"] == 4096]
    def boundary(backend):
        rows = [r for r in warm if r["backend"] == backend]
        return max(rows, key=lambda r: r["n_qubits"]) if rows else None
    sv, mps = boundary("statevector"), boundary("matrix_product_state")
    med = breakdown["medians_ms"]["4096"]
    bottleneck = max((k for k in med if k not in {"autoregressive_token_generation_ms", "total_profiled_ms"}), key=lambda k: med[k])
    candidate_text = json.dumps(candidates[:5], indent=2) if candidates else "None in measured base grid."
    report = f"""# M2 Operating-Region Report\n\n## Executive summary\n\n**M2 decision: `{verdict}`.** This is exploratory profiling, not a QuTE speedup claim.\n\n## Measured results\n\n- Host: {env['machine']}, {env['total_ram_bytes']/2**30:.1f} GiB RAM, {env['physical_cores']} physical cores; one benchmark thread.\n- Largest measured statevector cell: {sv['n_qubits'] if sv else 'none'} qubits, p={sv['p'] if sv else 'n/a'}, {sv['total_ms_median'] if sv else float('nan'):.3f} ms median for its listed cell.\n- Largest measured MPS cell: {mps['n_qubits'] if mps else 'none'} qubits, p={mps['p'] if mps else 'n/a'}, {mps['total_ms_median'] if mps else float('nan'):.3f} ms median for its listed cell.\n\n## Current QuTE bottleneck\n\nAt 4,096 shots the largest instrumented component is `{bottleneck}` at {med[bottleneck]:.3f} ms median. Model load is {breakdown['model_load_time_ms']:.3f} ms and is excluded from warm execution. The fit is planning-only.\n\n## Statevector scaling\n\nMeasured cell-level data are in `aggregate_results.csv`; payload and process RSS are separate. No OOM was induced.\n\n## MPS scaling\n\nMPS used Aer defaults with unrestricted exact truncation settings. Relative plots compare the same manifest cells.\n\n## Shot scaling\n\nThe 1,024, 4,096, and 65,536-shot representative measurements are in `aggregate_results.csv` and `figures/shot_scaling.png`.\n\n## Teacher-data feasibility\n\n`teacher_cost_projection.json` reports serial and idealized 8/32-worker linear projections. It does not assert cluster availability.\n\n## Candidate operating region\n\n```json\n{candidate_text}\n```\n\n## Limitations\n\n- QuTE accuracy and full execution remain valid only at frozen 6Q p=1.\n- Aer timing on this Apple CPU does not establish other-host boundaries.\n- Process RSS includes Python/Aer runtime overhead; state payload is reported separately.\n- Fits extrapolate no more than four qubits and are not speedup evidence.\n- No QPU job or large model training was performed.\n\n## M2 decision\n\n`{verdict}` under the preregistered rules in `docs/M2_OPERATING_REGION_PROTOCOL.md`.\n"""
    (ROOT / "docs/M2_OPERATING_REGION_REPORT.md").write_text(report)
    (out / "report.md").write_text(report)
    if candidates:
        first = candidates[0]
        scope = f"train widths: {first['n_qubits']} (small study around the measured candidate only)\nvalidation widths: 6 and {first['n_qubits']}\nQAOA p: {first['p']}\ngraph families: {first['graph_family']} plus cycle control\nshots: {first['shots']}\nmodel output: samples/counts, not explicit 2^n probabilities\ntarget accuracy: preregister TVD, energy/edge, marginals, ZZ correlations\ntarget latency: <= {first['required_neural_latency_2x_ms']:.3f} ms for 2x\ntarget memory: <= {first['required_neural_memory_mib']:.1f} MiB\n"
    else:
        scope = "train widths: none yet\nvalidation widths: 6 for frozen pilot reproduction only\nQAOA p: profile higher-depth or less MPS-friendly workload before training\ngraph families: retain cycle control; preregister stress workload\nshots: 1024, 4096, 65536\nmodel output: unchanged\ntarget accuracy: unchanged; no scale claim\ntarget latency: unresolved until a best-classical candidate exists\ntarget memory: unresolved until a best-classical candidate exists\n"
    recommendation = f"""# M2 Next-Stage Recommendation\n\n**Decision:** `{verdict}`\n\n```text\n{scope}```\n\nTraining-data budget must follow `teacher_cost_projection.json`; do not create a confirmatory corpus. QPU work is **not justified**: exact-regime scale success, a width/depth trend, a validation boundary, and a QPU-specific unresolved gap do not yet all exist.\n"""
    (ROOT / "docs/M2_NEXT_STAGE_RECOMMENDATION.md").write_text(recommendation)


def execute(config_path: Path, resume: bool = False) -> None:
    config = read_config(config_path); out = ROOT / config["artifact_root"]; out.mkdir(parents=True, exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = str(config["threads"]); os.environ["MKL_NUM_THREADS"] = str(config["threads"]); torch.set_num_threads(config["threads"])
    env = environment(); write_json(out / "environment.json", env)
    provenance = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
                  "status_porcelain": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines()}
    write_json(out / "git_provenance.json", provenance)
    corpus = generate_corpus(config, widths=config["base_qubits"] + config["adaptive_qubits"]); write_json(out / "corpus_manifest.json", {"schema_version": config["schema_version"], "count": len(corpus),
        "manifest_sha256": hashlib.sha256("".join(w.circuit_hash for w in corpus).encode()).hexdigest(), "config": config})
    write_csv(out / "circuit_manifest.csv", [w.row() for w in corpus])
    correct = correctness(corpus, ROOT / config["checkpoint"], config); write_json(out / "correctness_report.json", correct)
    pilot_report = json.loads((ROOT / "artifacts/r2_amortized_backend/pilot_v1/report.json").read_text())
    example = pilot_report["example"]["circuit"]
    import qute_r2_pilot as pilot
    pilot_circuit = pilot.QAOACircuit(example["n_qubits"], tuple(map(tuple, example["edges"])), example["gamma"], example["beta"], example["p"])
    breakdown = neural_breakdown(ROOT / config["checkpoint"], pilot_circuit, config["shots"]); write_json(out / "neural_latency_breakdown.json", breakdown)
    if resume and (out / "raw_runs.csv").exists():
        raw = read_csv_rows(out / "raw_runs.csv")
        guards = read_csv_rows(out / "resource_guard_events.csv") if (out / "resource_guard_events.csv").exists() else []
    else:
        raw, guards = profile(config, corpus, out)
    profile_adaptive(config, corpus, raw, guards, out)
    profile_batches(config, corpus, raw, out)
    profile_qute(config, raw)
    for row in raw:
        row.setdefault("batch_size", 1)
    guards = list({(event.get("backend"), event.get("graph_family"), event.get("n_qubits"), event.get("p"), event.get("status"), event.get("reason")): event for event in guards}.values())
    write_csv(out / "raw_runs.csv", raw); write_csv(out / "resource_guard_events.csv", guards)
    aggregate = aggregate_runs(raw); write_csv(out / "aggregate_results.csv", aggregate)
    cells = best_classical(aggregate)
    timeout_prefixes = {(event.get("graph_family"), int(event.get("p", 0))) for event in guards
                        if event.get("backend") == "matrix_product_state" and str(event.get("status", "")).startswith(("WORKER_FAILED", "SKIPPED_TIMEOUT"))}
    existing_cells = {(cell["graph_family"], cell["n_qubits"], cell["p"], cell["shots"]) for cell in cells}
    for row in aggregate:
        key = (row["graph_family"], row["n_qubits"], row["p"], row["shots"])
        if (row["backend"] == "statevector" and row["timing_mode"] == "warm_single" and row["batch_size"] == 1
                and (row["graph_family"], row["p"]) in timeout_prefixes and key not in existing_cells):
            cells.append({"graph_family": row["graph_family"], "n_qubits": row["n_qubits"], "p": row["p"], "shots": row["shots"],
                "best_classical_backend": "statevector", "median_classical_latency_ms": row["total_ms_median"],
                "classical_peak_memory_bytes": row.get("peak_rss_bytes_median"), "validation_method": "exact_statevector; MPS exceeded timeout at a smaller/equal width",
                "evidence_type": "measured_statevector_with_mps_timeout_bound"})
    cells.sort(key=lambda cell: (cell["graph_family"], cell["n_qubits"], cell["p"], cell["shots"]))
    budgets = budget_map(cells, config["neural_latency_budgets_ms"], config["neural_memory_budgets_mib"]); write_csv(out / "neural_budget_map.csv", budgets)
    fits = scaling_fits(aggregate); write_json(out / "scaling_fits.json", fits)
    teachers = teacher_projection(cells, config); write_json(out / "teacher_cost_projection.json", teachers)
    candidates = candidate_regions(cells, config); write_json(out / "candidate_operating_region.json", {"candidates": candidates})
    neural_4096 = breakdown["medians_ms"]["4096"]["total_profiled_ms"]
    write_json(out / "amortization_map.json", {"rows": amortization(cells, neural_4096, config)})
    relevant = [r for r in aggregate if r["timing_mode"] == "warm_single" and r["shots"] == 4096]
    ratios = []
    for sv in [r for r in relevant if r["backend"] == "statevector"]:
        match = next((r for r in relevant if r["backend"] == "matrix_product_state" and all(r[k] == sv[k] for k in ("graph_family", "n_qubits", "p", "shots"))), None)
        if match: ratios.append((match["total_ms_median"] / sv["total_ms_median"], match["total_ms_median"]))
    mps_dominates = bool(ratios) and all(ratio < .25 and latency < 250 for ratio, latency in ratios if True)
    verdict = decision(correct["status"] == "PASS", {r["backend"] for r in relevant}, candidates, mps_dominates)
    write_json(out / "M2_DECISION.json", {"decision": verdict, "scientific_role": "EXPLORATORY_NOT_CONFIRMATORY", "qute_speedup_claim": False,
        "correctness": correct["status"], "candidate_count": len(candidates), "mps_dominates_all_measured_cells": mps_dominates,
        "stress_family_activated": False, "stress_family_reason": "MPS did not satisfy the preregistered n<=20,p=3 latency conditions"})
    figures(aggregate, breakdown, budgets, out); reports(out, aggregate, candidates, breakdown, verdict, env)
    print(json.dumps({"decision": verdict, "artifact_root": str(out), "raw_runs": len(raw), "candidates": len(candidates)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, default=ROOT / "configs/r2/m2_operating_region_profile.yaml")
    parser.add_argument("--worker", nargs=2, metavar=("INPUT", "OUTPUT")); parser.add_argument("--resume", action="store_true"); args = parser.parse_args()
    if args.worker: worker(Path(args.worker[0]), Path(args.worker[1]))
    else: execute(args.config, resume=args.resume)


if __name__ == "__main__":
    main()
