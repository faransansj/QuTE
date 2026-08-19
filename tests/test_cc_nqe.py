import json
import math
import os
import subprocess
import sys

import numpy as np
import torch

from cc_nqe import (CircuitTransformer, FlatMLP, Gate, StateOnly, audit_dataset, benchmark, build_dataset,
                    deserialize_circuit, fidelity_np, generate_circuit, generate_state, has_composition_motif,
                    normalize_real, serialize_circuit, simulate)


def test_circuit_serialization_and_determinism():
    circuit = generate_circuit(123, 8)
    assert deserialize_circuit(serialize_circuit(circuit)) == circuit
    assert generate_circuit(123, 8) == circuit
    assert generate_circuit(124, 8) != circuit
    assert all(not (a.name == b.name and a.qubits == b.qubits) for a, b in zip(circuit, circuit[1:]))
    command = "from cc_nqe import *; print(serialize_circuit(generate_circuit(123,8)))"
    outputs = []
    for hash_seed in ("1", "2"):
        env = os.environ | {"PYTHONHASHSEED": hash_seed}
        outputs.append(subprocess.check_output([sys.executable, "-c", command], env=env, text=True))
    assert outputs[0] == outputs[1]


def test_state_generation_deterministic_and_normalized():
    for family in ("product", "random-local", "entangled", "Haar-random"):
        a, b = generate_state(99, family), generate_state(99, family)
        assert np.array_equal(a, b)
        assert a.dtype == np.complex128
        assert abs(np.linalg.norm(a) - 1) < 1e-12


def test_teacher_simple_circuits():
    zero = np.zeros(16, np.complex128); zero[0] = 1
    x = simulate([Gate("X", (3,))], zero)
    assert np.argmax(abs(x)) == 1
    bell = simulate([Gate("H", (0,)), Gate("CNOT", (0, 1))], zero)
    expected = np.zeros(16, np.complex128); expected[0] = expected[12] = 1 / math.sqrt(2)
    assert np.allclose(bell, expected)


def test_fidelity_is_phase_invariant():
    state = generate_state(2, "Haar-random")
    assert np.isclose(fidelity_np(state[None], (np.exp(1.234j) * state)[None])[0], 1.0)
    e0, e1 = np.eye(16, dtype=complex)[:2]
    assert np.isclose(fidelity_np(e0[None], e1[None])[0], 0.0)


def test_split_contracts_and_regeneration(tmp_path):
    root = tmp_path / "dataset"
    build_dataset(root, {"dataset_seed": 7})
    result = audit_dataset(root)
    assert result["status"] == "PASS", result
    assert result["duplicate_sample_count"] == 0
    assert all(result["checks"].values())
    rows = [json.loads(x) for x in (root / "samples.jsonl").read_text().splitlines()]
    train = [x for x in rows if x["split_name"] == "train"]
    comp = [x for x in rows if x["split_name"] == "composition_ood"]
    assert all(not has_composition_motif([Gate.from_dict(g) for g in x["gate_sequence_structured"]]) for x in train)
    assert all(has_composition_motif([Gate.from_dict(g) for g in x["gate_sequence_structured"]]) for x in comp)
    assert {x["depth"] for x in train} == {2, 4, 6}
    assert {x["depth"] for x in rows if x["split_name"] == "depth_ood"} == {8}
    train_states = {x["state_id"] for x in train}
    for split in ("state_ood", "parameter_interpolation", "parameter_extrapolation", "composition_ood", "depth_ood"):
        assert not train_states & {x["state_id"] for x in rows if x["split_name"] == split}


def _model_inputs(batch=3):
    state = torch.randn(batch, 32)
    gates = torch.tensor([[1, 6, 0, 0, 0, 0, 0, 0]] * batch)
    qubits = torch.full((batch, 8, 2), 4, dtype=torch.long); qubits[:, 0, 0] = 0; qubits[:, 1] = torch.tensor([0, 1])
    params = torch.zeros(batch, 8, 3)
    mask = gates != 0
    return state, gates, qubits, params, mask


def test_model_shapes_and_prediction_normalization():
    args = _model_inputs()
    for model in (StateOnly(), FlatMLP(), CircuitTransformer()):
        output = model(*args)
        assert output.shape == (3, 32)
        assert torch.allclose(normalize_real(output).norm(dim=1), torch.ones(3), atol=1e-6)


def test_cached_context_equivalence():
    torch.manual_seed(1)
    model = CircuitTransformer().eval()
    args = _model_inputs()
    with torch.inference_mode():
        direct = model(*args)
        cached = model.forward_cached(args[0], model.encode_context(*args[1:]))
    assert torch.equal(direct, cached)


def test_benchmark_harness_sanity():
    torch.manual_seed(1)
    model = CircuitTransformer().eval()
    circuit = generate_circuit(55, 2)
    row = {"gate_sequence_structured": [g.to_dict() for g in circuit]}
    result = benchmark(model, row, seed=4, repetitions=2)
    assert len(result["repeated_context"]) == 5
    assert result["repeated_context"][-1]["N"] == 10_000
    assert result["single_query"]["exact_uncached_ms"]["median"] > 0
    assert result["single_query"]["neural_cached_forward_ms"]["median"] > 0
    assert "direct gate simulation" in result["exact_context_policy"]
    assert all(x["measurement_repetitions"] == 5 for x in result["repeated_context"])
