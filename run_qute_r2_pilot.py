"""Run the frozen six-qubit M1 feasibility pilot and write its evidence."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import resource
import statistics
import time
from pathlib import Path

import numpy as np
import psutil
import torch

import qute_r2_pilot as pilot

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts" / "r2_amortized_backend" / "pilot_v1"
CHECKPOINT = OUT / "qute_qaoa_r1_pilot.pt"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_circuits(chords: tuple[tuple[int, int], ...], per_graph: int, seed: int) -> list[pilot.QAOACircuit]:
    rng = np.random.default_rng(seed)
    circuits = []
    for chord in chords:
        for _ in range(per_graph):
            circuits.append(
                pilot.QAOACircuit(
                    n_qubits=6,
                    edges=pilot.cycle_plus_chord(6, chord),
                    gamma=float(rng.uniform(0.15, 1.35)),
                    beta=float(rng.uniform(0.10, 0.70)),
                )
            )
    return circuits


def bit_matrix(n_qubits: int) -> np.ndarray:
    return np.asarray([[int(bit) for bit in format(value, f"0{n_qubits}b")] for value in range(1 << n_qubits)])


def fit_linear_baselines(
    circuits: list[pilot.QAOACircuit], probabilities: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    features = np.stack([np.append(pilot._adjacency_and_degrees(circuit), 1.0) for circuit in circuits])
    bits = bit_matrix(6)
    energy = np.asarray([probability @ pilot.cut_values(circuit) for circuit, probability in zip(circuits, probabilities)])
    marginals = np.stack([probability @ bits for probability in probabilities])
    ridge = 1e-4 * np.eye(features.shape[1])
    inverse = np.linalg.solve(features.T @ features + ridge, features.T)
    return inverse @ energy, inverse @ marginals


def independent_distribution(marginals: np.ndarray) -> np.ndarray:
    bits = bit_matrix(len(marginals))
    probabilities = np.prod(np.where(bits, marginals, 1 - marginals), axis=1)
    return probabilities / probabilities.sum()


def qiskit_cross_check(circuit: pilot.QAOACircuit) -> dict[str, object]:
    try:
        from qiskit import QuantumCircuit, __version__ as qiskit_version
        from qiskit.quantum_info import Statevector
    except Exception as error:
        return {"status": "UNAVAILABLE", "reason": type(error).__name__}
    quantum_circuit = QuantumCircuit(circuit.n_qubits)
    quantum_circuit.h(range(circuit.n_qubits))
    for a, b in circuit.edges:
        quantum_circuit.rzz(-circuit.gamma, a, b)
    quantum_circuit.rx(2 * circuit.beta, range(circuit.n_qubits))
    expected = Statevector.from_instruction(quantum_circuit).probabilities()
    actual = pilot.exact_probabilities(circuit)
    return {
        "status": "PASS" if np.max(np.abs(expected - actual)) < 1e-12 else "FAIL",
        "qiskit_version": qiskit_version,
        "max_probability_abs_error": float(np.max(np.abs(expected - actual))),
    }


def qiskit_backend_adapter_check(backend: pilot.QuTEBackend, circuit: pilot.QAOACircuit) -> dict[str, object]:
    try:
        from qiskit import QuantumCircuit
    except Exception as error:
        return {"status": "UNAVAILABLE", "reason": type(error).__name__}
    quantum_circuit = QuantumCircuit(circuit.n_qubits, circuit.n_qubits)
    quantum_circuit.h(range(circuit.n_qubits))
    for a, b in circuit.edges:
        quantum_circuit.rzz(-circuit.gamma, a, b)
    quantum_circuit.rx(2 * circuit.beta, range(circuit.n_qubits))
    quantum_circuit.measure(range(circuit.n_qubits), range(circuit.n_qubits))
    quantum_circuit.metadata = {"qute_qaoa": circuit.canonical()}
    exact_before = pilot.exact_call_count()
    result = backend.run(quantum_circuit, shots=17, seed_simulator=7).result()
    return {
        "status": "PASS" if sum(result.get_counts().values()) == 17 else "FAIL",
        "counts_total": sum(result.get_counts().values()),
        "exact_calls_on_route": pilot.exact_call_count() - exact_before,
    }


def benchmark_latency(backend: pilot.QuTEBackend, circuit: pilot.QAOACircuit, shots: int) -> dict[str, object]:
    for seed in range(3):
        backend.run(circuit, shots=shots, seed_simulator=seed).result()
    neural_ms = []
    exact_ms = []
    for seed in range(7):
        start = time.perf_counter_ns()
        backend.run(circuit, shots=shots, seed_simulator=100 + seed).result().get_counts()
        neural_ms.append((time.perf_counter_ns() - start) / 1e6)
        start = time.perf_counter_ns()
        probabilities = pilot.exact_probabilities(circuit)
        pilot.sample_counts(probabilities, circuit.n_qubits, shots, 100 + seed)
        exact_ms.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "shots": shots,
        "repetitions": 7,
        "warmups": 3,
        "neural_end_to_end_median_ms": statistics.median(neural_ms),
        "exact_statevector_to_counts_median_ms": statistics.median(exact_ms),
        "neural_samples_per_second": shots / (statistics.median(neural_ms) / 1000),
        "exact_samples_per_second": shots / (statistics.median(exact_ms) / 1000),
        "neural_all_ms": neural_ms,
        "exact_all_ms": exact_ms,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    process = psutil.Process()
    start_rss = process.memory_info().rss

    generation_start = time.perf_counter()
    chords = pilot.available_chords(6)
    train_chords, validation_chords = chords[:6], chords[6:]
    train_circuits = make_circuits(train_chords, 12, 20260831)
    validation_circuits = make_circuits(validation_chords, 4, 20260832)
    circuit_generation_seconds = time.perf_counter() - generation_start
    assert {c.graph_hash for c in train_circuits}.isdisjoint(c.graph_hash for c in validation_circuits)

    exact_before_teacher = pilot.exact_call_count()
    teacher_start = time.perf_counter()
    teacher_counts = []
    train_probabilities = []
    for index, circuit in enumerate(train_circuits):
        probabilities = pilot.exact_probabilities(circuit)
        train_probabilities.append(probabilities)
        teacher_counts.append((circuit, pilot.sample_counts(probabilities, 6, 1024, 30_000 + index)))
    teacher_seconds = time.perf_counter() - teacher_start
    teacher_exact_calls = pilot.exact_call_count() - exact_before_teacher

    training_start = time.perf_counter()
    model, losses = pilot.train_sampler(teacher_counts, epochs=120)
    training_seconds = time.perf_counter() - training_start
    model_hash = pilot.save_checkpoint(model, CHECKPOINT)
    backend = pilot.QuTEBackend.from_pretrained(CHECKPOINT)

    energy_coefficients, marginal_coefficients = fit_linear_baselines(train_circuits, train_probabilities)
    uniform = np.full(64, 1 / 64)
    validation_rows = []
    for circuit in validation_circuits:
        exact = pilot.exact_probabilities(circuit)
        predicted = pilot.model_distribution(model, circuit)
        linear_features = np.append(pilot._adjacency_and_degrees(circuit), 1.0)
        predicted_marginals = np.clip(linear_features @ marginal_coefficients, 1e-4, 1 - 1e-4)
        independent = independent_distribution(predicted_marginals)
        exact_energy = float(exact @ pilot.cut_values(circuit))
        validation_rows.append(
            {
                "circuit_hash": circuit.circuit_hash,
                "graph_hash": circuit.graph_hash,
                "gamma": circuit.gamma,
                "beta": circuit.beta,
                "qute": pilot.distribution_metrics(exact, predicted, circuit),
                "independent_bits": pilot.distribution_metrics(exact, independent, circuit),
                "uniform": pilot.distribution_metrics(exact, uniform, circuit),
                "observable_regressor_abs_error_per_edge": abs(exact_energy - float(linear_features @ energy_coefficients))
                / len(circuit.edges),
            }
        )

    def aggregate(name: str) -> dict[str, float]:
        keys = validation_rows[0][name].keys()
        return {f"median_{key}": statistics.median(row[name][key] for row in validation_rows) for key in keys}

    aggregate_metrics = {
        "qute": aggregate("qute"),
        "independent_bits": aggregate("independent_bits"),
        "uniform": aggregate("uniform"),
        "observable_regressor_median_abs_error_per_edge": statistics.median(
            row["observable_regressor_abs_error_per_edge"] for row in validation_rows
        ),
    }

    example_circuit = validation_circuits[0]
    exact_before_inference = pilot.exact_call_count()
    example_result = backend.run(example_circuit, shots=4096, seed_simulator=2026).result()
    example_counts = example_result.get_counts()
    exact_calls_on_inference = pilot.exact_call_count() - exact_before_inference
    assert exact_calls_on_inference == 0
    assert sum(example_counts.values()) == 4096
    assert example_result.metadata[0]["per_circuit_optimizer_steps"] == 0

    latency = benchmark_latency(backend, example_circuit, 4096)
    offline_seconds = circuit_generation_seconds + teacher_seconds + training_seconds
    direct_ms = latency["exact_statevector_to_counts_median_ms"]
    neural_ms = latency["neural_end_to_end_median_ms"]
    denominator_seconds = (direct_ms - neural_ms) / 1000
    break_even = offline_seconds / denominator_seconds if denominator_seconds > 0 else None

    current_rss = process.memory_info().rss
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != "Darwin":
        peak_rss *= 1024

    qute_metrics = aggregate_metrics["qute"]
    gate_checks = {
        "counts_sum_to_shots": sum(example_counts.values()) == 4096,
        "per_circuit_optimizer_steps_zero": example_result.metadata[0]["per_circuit_optimizer_steps"] == 0,
        "exact_simulator_calls_on_inference_zero": exact_calls_on_inference == 0,
        "median_iid_tvd_le_0_35": qute_metrics["median_tvd"] <= 0.35,
        "median_energy_error_per_edge_le_0_10": qute_metrics["median_cut_energy_error_per_edge"] <= 0.10,
        "latency_reported": neural_ms > 0 and direct_ms > 0,
        "peak_memory_reported": peak_rss > 0,
    }
    pilot_pass = all(gate_checks.values())

    report = {
        "schema_version": "qute-amortized-pilot-v1",
        "status": "PILOT_PASS" if pilot_pass else "PILOT_FAIL",
        "research_success": False,
        "research_success_reason": "SYSTEMS_AND_ECONOMIC_CONDITIONS_FAIL_IN_SIX_QUBIT_PILOT",
        "scientific_role": "FEASIBILITY_ONLY_NOT_CONFIRMATORY",
        "source_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "protocol_hashes": {
            name: sha256_file(ROOT / name)
            for name in ["RESEARCH_PIVOT.md", "R1_SCOPE.yaml", "BENCHMARK_PROTOCOL.md", "BACKEND_CONTRACT.md"]
        },
        "source_hashes": {
            name: sha256_file(ROOT / name)
            for name in ["qute_r2_pilot.py", "run_qute_r2_pilot.py", "tests/test_qute_r2_pilot.py"]
        },
        "corpus": {
            "train_circuits": len(train_circuits),
            "validation_circuits": len(validation_circuits),
            "teacher_shots_per_circuit": 1024,
            "generation_seeds": {"train": 20260831, "validation": 20260832, "teacher_sampling_base": 30000},
            "train_circuit_hashes": [c.circuit_hash for c in train_circuits],
            "validation_circuit_hashes": [c.circuit_hash for c in validation_circuits],
            "train_graph_hashes": sorted({c.graph_hash for c in train_circuits}),
            "validation_graph_hashes": sorted({c.graph_hash for c in validation_circuits}),
            "graph_hash_intersection": [],
            "stored_probability_tables": 0,
        },
        "cost": {
            "circuit_generation_seconds": circuit_generation_seconds,
            "transpilation_seconds": 0.0,
            "teacher_exact_and_sampling_seconds": teacher_seconds,
            "teacher_exact_calls": teacher_exact_calls,
            "training_seconds": training_seconds,
            "offline_total_seconds": offline_seconds,
            "checkpoint_bytes": CHECKPOINT.stat().st_size,
            "training_final_loss": losses[-1],
            "training_initial_loss": losses[0],
        },
        "model": {
            "type": "conditional_autoregressive_mlp",
            "parameters": pilot.model_parameter_count(model),
            "checkpoint_sha256": model_hash,
            "per_circuit_training": False,
            "per_circuit_optimizer_steps": 0,
            "explicit_2n_inference_output": False,
        },
        "validation": {
            "rows": validation_rows,
            "aggregate": aggregate_metrics,
            "qiskit_statevector_cross_check": qiskit_cross_check(example_circuit),
            "qiskit_quantum_circuit_adapter": qiskit_backend_adapter_check(backend, example_circuit),
        },
        "example": {
            "circuit": example_circuit.canonical(),
            "counts_top_12": dict(sorted(example_counts.items(), key=lambda item: item[1], reverse=True)[:12]),
            "counts_total": sum(example_counts.values()),
            "metadata": example_result.metadata[0],
        },
        "systems": {
            "latency": latency,
            "process_start_rss_bytes": start_rss,
            "process_current_rss_bytes": current_rss,
            "process_peak_rss_bytes_combined": peak_rss,
            "exact_statevector_payload_bytes": (1 << 6) * np.dtype(np.complex128).itemsize,
            "memory_attribution": "combined pilot process; isolated backend peaks deferred",
            "neural_to_exact_latency_ratio": neural_ms / direct_ms,
            "break_even_evaluations": break_even,
            "break_even_status": "FINITE" if break_even is not None else "INFINITE_NEURAL_NOT_FASTER",
        },
        "proof": {
            "exact_calls_on_neural_example_route": exact_calls_on_inference,
            "backend_metadata_exact_calls": example_result.metadata[0]["exact_simulator_calls_on_neural_route"],
            "test_strategy": "tests monkeypatch exact_probabilities to raise while backend.run still succeeds",
        },
        "gate_checks": gate_checks,
        "known_failures": [
            "pilot is six qubits and p=1 only",
            "uncertainty is not calibrated; support score is a hard envelope check",
            "Aer and MPS are not installed in the pilot environment",
            "memory high-water mark combines teacher, training, exact evaluation, and inference",
            "Qiskit adapter trusts canonical qute_qaoa metadata rather than parsing arbitrary transpiled circuits",
            "pilot circuits and thresholds are not confirmatory evidence",
        ],
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "qiskit": "2.5.2",
            "psutil": psutil.__version__,
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    top_counts = "\n".join(f"- `{key}`: {value}" for key, value in report["example"]["counts_top_12"].items())
    markdown = f"""# QuTE M1 R1 Pilot Report

**Verdict:** `{report['status']}`
**Role:** feasibility only; not confirmatory

## Frozen gate

| Check | Result |
|---|---|
{chr(10).join(f'| `{key}` | {"PASS" if value else "FAIL"} |' for key, value in gate_checks.items())}

## Evidence

- One checkpoint was trained on **{len(train_circuits)}** circuits and evaluated on **{len(validation_circuits)}** unseen-graph circuits.
- Per-circuit optimizer steps: **0**.
- Exact simulator calls on the demonstrated neural route: **{exact_calls_on_inference}**.
- Counts total: **{sum(example_counts.values())} / 4096**.
- Median TVD: **{qute_metrics['median_tvd']:.6f}** (pilot threshold `<= 0.35`).
- Median cut-energy error per edge: **{qute_metrics['median_cut_energy_error_per_edge']:.6f}** (threshold `<= 0.10`).
- Independent-bit median TVD: **{aggregate_metrics['independent_bits']['median_tvd']:.6f}**.
- Uniform median TVD: **{aggregate_metrics['uniform']['median_tvd']:.6f}**.
- Observable-regressor median energy error per edge: **{aggregate_metrics['observable_regressor_median_abs_error_per_edge']:.6f}**.

## Counts example

{top_counts}

## Systems result

- Neural parse-to-counts median latency, 4096 shots: **{neural_ms:.3f} ms**.
- Exact statevector-to-counts median latency: **{direct_ms:.3f} ms**.
- Neural/exact latency ratio: **{neural_ms / direct_ms:.2f}x slower**.
- Combined process peak RSS: **{peak_rss / 2**20:.2f} MiB**.
- Exact six-qubit state payload: **{((1 << 6) * np.dtype(np.complex128).itemsize) / 1024:.2f} KiB**.
- Checkpoint: **{CHECKPOINT.stat().st_size / 1024:.2f} KiB**, {pilot.model_parameter_count(model)} parameters.
- Teacher generation and sampling: **{teacher_seconds:.3f} s**.
- Training: **{training_seconds:.3f} s**.
- Break-even: **{f'{break_even:.0f} evaluations' if break_even is not None else 'none; neural execution was not faster in this six-qubit pilot'}**.

## Interpretation

The feasibility gate passes, but the overall research success condition does **not**: the neural path is slower, uses a larger payload than the six-qubit exact state, and has no finite break-even point. A pass permits Phase 1 profiling only. It does not establish scale, OOD reliability, or replacement of Aer/MPS.

## Known failures

{chr(10).join(f'- {item}' for item in report['known_failures'])}

## Next gate

Proceed only to simulator/workload profiling. Do not generate confirmatory data. Phase 4 must freeze exact thresholds, corpus hashes, seeds, baseline versions, hardware, statistics, model selection, and calibration before confirmatory generation.
"""
    (OUT / "PILOT_REPORT.md").write_text(markdown)
    print(json.dumps({"status": report["status"], "report": str(OUT / "PILOT_REPORT.md"), "metrics": qute_metrics}, indent=2))


if __name__ == "__main__":
    main()
