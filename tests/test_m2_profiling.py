from __future__ import annotations

import json
import signal
from pathlib import Path

import pytest

import qute_r2_pilot as pilot
from qute_r2.profiling.analysis import aggregate_runs, candidate_regions, decision, statevector_guard
from qute_r2.profiling.corpus import canonical_hash, generate_corpus, qiskit_circuit
from qute_r2.profiling.run import alarm_handler, timed_aer, write_csv, write_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/r2/m2_operating_region_profile.yaml").read_text())


def test_corpus_is_deterministic_and_manifest_hash_changes_with_content() -> None:
    first = generate_corpus(CONFIG, widths=[6], families=["cycle"])
    second = generate_corpus(CONFIG, widths=[6], families=["cycle"])
    assert first == second
    assert len(first) == 3 * 3 * 4
    assert len({row.circuit_hash for row in first}) == len(first)
    assert canonical_hash({"a": 1}) == canonical_hash({"a": 1})
    assert canonical_hash({"a": 1}) != canonical_hash({"a": 2})


def test_statevector_resource_guard_and_timeout_projection() -> None:
    assert statevector_guard(30, 2**30)["status"] == "SKIPPED_RESOURCE_GUARD"
    assert statevector_guard(6, 2**40, projected_seconds=121)["status"] == "SKIPPED_TIMEOUT_PROJECTION"
    assert statevector_guard(6, 2**40, projected_seconds=1)["status"] == "RUN"


def test_timeout_handler_is_not_silenced() -> None:
    with pytest.raises(TimeoutError):
        alarm_handler(signal.SIGALRM, None)


def test_timing_aggregation_and_rss_fields() -> None:
    rows = []
    for value in (1.0, 2.0, 3.0):
        rows.append({"status": "OK", "backend": "statevector", "graph_family": "cycle", "n_qubits": 6,
            "p": 1, "shots": 1024, "timing_mode": "warm_single", "total_ms": value, "execute_ms": value,
            "transpile_ms": value, "peak_rss_bytes": 100 + value, "incremental_peak_rss_bytes": 10 + value})
    result = aggregate_runs(rows)[0]
    assert result["total_ms_median"] == 2
    assert result["total_ms_p10"] == pytest.approx(1.2)
    assert result["peak_rss_bytes_median"] == 102


def test_candidate_and_decision_rules() -> None:
    cell = {"graph_family": "random_3_regular", "n_qubits": 20, "p": 3, "shots": 4096,
        "best_classical_backend": "matrix_product_state", "median_classical_latency_ms": 1000,
        "classical_peak_memory_bytes": 2 * 2**30, "validation_method": "mps", "evidence_type": "measured"}
    candidates = candidate_regions([cell], CONFIG)
    assert len(candidates) == 1
    assert candidates[0]["required_neural_latency_2x_ms"] == 500
    assert decision(True, {"statevector", "matrix_product_state"}, candidates, False) == "PROCEED_TO_M3"
    assert decision(True, {"statevector", "matrix_product_state"}, [], True) == "PIVOT_WORKLOAD_BEFORE_M3"
    assert decision(False, {"statevector", "matrix_product_state"}, [], False) == "BLOCKED"
    assert decision(True, {"statevector", "matrix_product_state"}, [], False) == "NO_FEASIBLE_REGION_FOUND"


def test_small_aer_statevector_and_mps_integration() -> None:
    from qiskit_aer import AerSimulator

    workload = generate_corpus(CONFIG, widths=[6], families=["cycle"])[1]
    for method in ("statevector", "matrix_product_state"):
        result = timed_aer(workload, AerSimulator(method=method), 128, 7, 30)
        assert result["total_ms"] > 0
        assert result["execute_ms"] > 0


def test_current_qute_route_has_no_exact_call_and_no_optimizer_steps(monkeypatch) -> None:
    checkpoint = ROOT / CONFIG["checkpoint"]
    backend = pilot.QuTEBackend.from_pretrained(checkpoint)
    circuit = pilot.QAOACircuit(6, pilot.cycle_plus_chord(6, (0, 3)), .7, .3)

    def forbidden(*args, **kwargs):
        raise AssertionError("exact path reached")

    monkeypatch.setattr(pilot, "exact_probabilities", forbidden)
    result = backend.run(circuit, shots=32).result()
    assert sum(result.get_counts().values()) == 32
    assert result.metadata[0]["exact_simulator_calls_on_neural_route"] == 0
    assert result.metadata[0]["per_circuit_optimizer_steps"] == 0


def test_resource_guard_skip_and_artifact_generation_smoke(tmp_path: Path) -> None:
    skip = statevector_guard(60, 16 * 2**30)
    assert skip["status"] == "SKIPPED_RESOURCE_GUARD"
    write_json(tmp_path / "report.json", skip)
    write_csv(tmp_path / "rows.csv", [skip])
    assert json.loads((tmp_path / "report.json").read_text())["status"] == "SKIPPED_RESOURCE_GUARD"
    assert "predicted_state_payload_bytes" in (tmp_path / "rows.csv").read_text()
