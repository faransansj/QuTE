from pathlib import Path

import numpy as np
import pytest
import torch

import qute_r2_pilot as pilot


def circuit() -> pilot.QAOACircuit:
    return pilot.QAOACircuit(
        n_qubits=6,
        edges=pilot.cycle_plus_chord(6, (0, 3)),
        gamma=0.7,
        beta=0.3,
    )


def test_exact_zero_angles_are_uniform() -> None:
    zero = pilot.QAOACircuit(6, pilot.cycle_edges(6), gamma=0.0, beta=0.0)
    np.testing.assert_allclose(pilot.exact_probabilities(zero), np.full(64, 1 / 64), atol=1e-14)


def test_backend_generates_counts_without_exact_path_or_per_circuit_training(tmp_path: Path, monkeypatch) -> None:
    torch.manual_seed(1)
    checkpoint = tmp_path / "model.pt"
    pilot.save_checkpoint(pilot.ConditionalAutoregressiveSampler(), checkpoint)
    backend = pilot.QuTEBackend.from_pretrained(checkpoint)

    def forbidden(*args, **kwargs):
        raise AssertionError("exact simulator reached from inference")

    monkeypatch.setattr(pilot, "exact_probabilities", forbidden)
    result = backend.run(circuit(), shots=257, seed_simulator=9).result()
    counts = result.get_counts()
    assert sum(counts.values()) == 257
    assert all(len(key) == 6 and set(key) <= {"0", "1"} for key in counts)
    assert result.metadata[0]["per_circuit_optimizer_steps"] == 0
    assert result.metadata[0]["exact_simulator_calls_on_neural_route"] == 0


def test_backend_accepts_qiskit_style_metadata(tmp_path: Path) -> None:
    class FakeQuantumCircuit:
        metadata = {"qute_qaoa": circuit().canonical()}

    checkpoint = tmp_path / "model.pt"
    pilot.save_checkpoint(pilot.ConditionalAutoregressiveSampler(), checkpoint)
    counts = pilot.QuTEBackend.from_pretrained(checkpoint).run(FakeQuantumCircuit(), shots=11).result().get_counts()
    assert sum(counts.values()) == 11


def test_backend_rejects_outside_envelope(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    pilot.save_checkpoint(pilot.ConditionalAutoregressiveSampler(), checkpoint)
    unsupported = pilot.QAOACircuit(6, pilot.cycle_edges(6), gamma=0.4, beta=0.2, p=2)
    with pytest.raises(pilot.QuTEUnsupportedCircuitError):
        pilot.QuTEBackend.from_pretrained(checkpoint).run(unsupported)


def test_small_width_model_distribution_is_normalized() -> None:
    distribution = pilot.model_distribution(pilot.ConditionalAutoregressiveSampler(), circuit())
    assert distribution.shape == (64,)
    assert np.all(distribution >= 0)
    assert distribution.sum() == pytest.approx(1.0)
