import argparse
import json

import numpy as np
import pytest

from qute_qpu_validation import (BIT_ORDER, build_workload, distribution, load_assets, normalize_counts, run,
                                 select_samples, statevector_distribution, tvd, validate_bit_order)


def test_selection_is_deterministic_and_unique():
    rows, _, _ = load_assets()
    first = select_samples(rows, 5)
    second = select_samples(rows, 5)
    assert [(row["sample_id"], row["circuit_id"]) for row in first] == [(row["sample_id"], row["circuit_id"]) for row in second]
    assert len(first) == 20
    assert all(sum(row["validation_split"] == split for row in first) == 5 for split in ("IID", "Parameter-OOD", "Composition-OOD", "Depth-OOD"))
    assert len({(row["validation_split"], row["circuit_id"]) for row in first}) == 20
    assert {row["state_family"] for row in first} <= {"product", "random-local"}


def test_probability_counts_and_tvd():
    state = np.zeros(16, complex); state[3] = 1
    probs = distribution(state)
    assert probs["0011"] == 1
    counts = normalize_counts({"00 11": 3, "0000": 1}, 4)
    assert sum(counts.values()) == pytest.approx(1)
    assert counts["0011"] == pytest.approx(0.75)
    assert tvd(probs, probs) == 0
    assert tvd(probs, counts) == pytest.approx(0.25)
    with pytest.raises(ValueError): normalize_counts({"0": 3}, 4)


def test_circuit_reconstruction_and_bit_order():
    pytest.importorskip("qiskit")
    rows, inputs, targets = load_assets()
    row = select_samples(rows, 1)[0]
    circuit = build_workload(row, inputs[row["array_index"]])
    assert circuit.num_qubits == 4 and circuit.num_clbits == 4
    assert tvd(statevector_distribution(circuit), distribution(targets[row["array_index"]])) < 1e-12
    result = validate_bit_order()
    assert result["status"] == "PASS"
    assert result["convention"] == BIT_ORDER


def test_offline_dry_run_artifact_schema(tmp_path):
    pytest.importorskip("qiskit")
    from qiskit.providers.fake_provider import GenericBackendV2

    output = tmp_path / "artifacts"
    args = argparse.Namespace(backend="offline-test-backend", shots=128, num_per_split=1, optimization_level=0,
                              instance=None, output_dir=str(output), dry_run=True, preflight=None)
    run(args, backend=GenericBackendV2(5, seed=7))
    required = {"config.json", "selected_circuits.json", "backend_metadata.json", "circuit_metrics.json",
                "per_circuit_results.json", "summary.json", "report.md", "job_manifest.json", "checksums.sha256"}
    assert required <= {path.name for path in output.iterdir()}
    config = json.loads((output / "config.json").read_text())
    results = json.loads((output / "per_circuit_results.json").read_text())
    assert config["status"] == "QPU_SUBMISSION_READY"
    assert len(results) == 4
    assert all(row["qpu_distribution"] is None and row["tvd_model_sim"] is not None for row in results)
    assert json.loads((output / "job_manifest.json").read_text())["status"] == "NOT_SUBMITTED_DRY_RUN"
