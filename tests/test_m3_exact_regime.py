from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from qute_r2.m3.model import GraphAutoregressiveSampler, M3Backend, graph_tensors, save_checkpoint
from qute_r2.m3.run import bits_from_outcomes, fixed_corpus, gate_decision, outcome_metrics

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/r2/m3_exact_regime_scale_gate.yaml").read_text())


def test_m3_corpus_is_frozen_and_split_disjoint() -> None:
    train_a = fixed_corpus(CONFIG, "calibration")
    train_b = fixed_corpus(CONFIG, "calibration")
    validation = fixed_corpus(CONFIG, "validation")
    assert train_a == train_b
    assert len(train_a) == CONFIG["calibration_circuits"] == 216
    assert len(validation) == 144
    assert {row.circuit_hash for row in train_a}.isdisjoint(row.circuit_hash for row in validation)
    assert {row.graph_seed for row in train_a if row.graph_family == "random_3_regular"}.isdisjoint(
        row.graph_seed for row in validation if row.graph_family == "random_3_regular"
    )


def test_graph_model_is_variable_width_and_causal() -> None:
    rows = fixed_corpus(CONFIG, "validation")
    selected = [next(row for row in rows if row.n_qubits == n) for n in (18, 24)]
    for pair_mode in ("dot", "mlp", "context", "shell", "gru", "hybrid"):
        model = GraphAutoregressiveSampler(hidden=16, message_passing_layers=1, pair_mode=pair_mode)
        for row in selected:
            feature, adjacency, mask = graph_tensors([row])
            bits = torch.randint(0, 2, (1, 3, 24), dtype=torch.float32)
            changed = bits.clone(); changed[:, :, 10:] = 1 - changed[:, :, 10:]
            first = model(feature, adjacency, mask, bits)
            second = model(feature, adjacency, mask, changed)
            torch.testing.assert_close(first[:, :, :10], second[:, :, :10])
            assert first.shape == (1, 3, 24)


def test_backend_has_no_exact_route_or_optimizer(tmp_path: Path) -> None:
    row = fixed_corpus(CONFIG, "validation")[0]
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(GraphAutoregressiveSampler(hidden=16, message_passing_layers=1), checkpoint)
    counts, metadata = M3Backend.from_pretrained(checkpoint).sample(row, 64, seed=7)
    assert sum(counts.values()) == 64
    assert metadata["per_circuit_optimizer_steps"] == 0
    assert metadata["exact_simulator_calls_on_neural_route"] == 0
    assert metadata["explicit_2n_inference_output"] is False


def test_outcome_bits_and_metrics() -> None:
    outcomes = np.asarray([[0, 1, 3]], dtype=np.uint32)
    bits = bits_from_outcomes(outcomes)
    assert bits.shape == (1, 3, 24)
    assert bits[0, 2, :2].tolist() == [1, 1]
    row = fixed_corpus(CONFIG, "validation")[0]
    same = outcome_metrics(outcomes[0], outcomes[0], row)
    assert same == {"energy_error_per_edge": 0.0, "marginal_mae": 0.0, "zz_mae": 0.0}


def test_gate_requires_full_corpus_and_all_metrics() -> None:
    validation = {"overall": {"median_energy_error_per_edge": 0.01, "median_marginal_mae": 0.01,
        "median_zz_mae": 0.01, "median_exact_tvd_18q": 0.1},
        "by_width": {str(n): {"median_energy_error_per_edge": 0.01} for n in (18, 20, 22, 24)}}
    systems = [{"n_qubits": n, "median_latency_ms": 50.0, "incremental_peak_rss_bytes": 10 * 2**20,
        "per_circuit_optimizer_steps": 0, "exact_calls": 0} for n in (18, 20, 22, 24)]
    decision, checks = gate_decision("full", validation, systems, CONFIG)
    assert decision == "M3_PASS_EXACT_REGIME"
    assert all(checks.values())
    decision, checks = gate_decision("calibration", validation, systems, CONFIG)
    assert decision == "M3_NEEDS_ITERATION"
    assert checks["full_10000_circuit_corpus"] is False
