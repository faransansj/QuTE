import json
from pathlib import Path

import numpy as np

from r1_gate4 import AUTHORIZATION, FEATURE_DIM, GATE4, syntax_pair_vector


def test_gate4_authorization_closed_with_frozen_verdict():
    authorization = json.loads(AUTHORIZATION.read_text())
    assert authorization["authorization_scope"] == "PILOT_VIABILITY_EVALUATION_ONLY"
    assert authorization["status"] == "CONSUMED"
    assert authorization["consumed"] is True
    assert authorization["final_verdict"] == "LEARNABILITY-BLOCKED"
    assert authorization["runner_source_sha256"]
    assert authorization["dependency_lock_sha256"]


def test_frozen_syntax_vector_is_305_dimensional_and_deterministic():
    left = [{"name": "H", "qubits": [0], "theta_hex": None}]
    right = [{"name": "X", "qubits": [1], "theta_hex": None}]
    first = syntax_pair_vector(left, right)
    second = syntax_pair_vector(left, right)
    assert first.shape == (FEATURE_DIM,) == (305,)
    assert np.array_equal(first, second)


def test_gate4_contract_and_results_remain_separate():
    protocol = json.loads((GATE4 / "evaluation_protocol.json").read_text())
    result = json.loads((GATE4 / "results" / "semantic_scoring.json").read_text())
    assert protocol["status"] == "FROZEN_EXECUTABLE"
    assert protocol["model_results_observed"] is False
    assert result["verdict"] == "LEARNABILITY-BLOCKED"
    assert not (GATE4 / "results" / "uncertainty.json").exists()
    assert not (GATE4 / "results" / "sample_size_planning.json").exists()
