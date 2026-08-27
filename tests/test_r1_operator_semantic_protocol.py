import hashlib
import json
import math
from pathlib import Path

import numpy as np

from cc_nqe import Gate, circuit_unitary


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "r1_operator_semantic_benchmark"
PROTOCOL = json.loads((ARTIFACT / "protocol.json").read_text())


def phase_equivalent(left: list[Gate], right: list[Gate]) -> bool:
    a, b = circuit_unitary(left), circuit_unitary(right)
    return abs(np.trace(a.conj().T @ b)) / a.shape[0] >= 1 - 1e-12


def test_protocol_is_frozen_but_not_run():
    assert PROTOCOL["status"] == "FROZEN_NOT_RUN"
    assert PROTOCOL["frozen"] is True
    assert PROTOCOL["execution_authorized"] is False
    assert PROTOCOL["execution_record"] == {
        "data_generated": False,
        "models_trained": False,
        "qpu_jobs_submitted": False,
        "final_test_accessed": False,
    }
    assert PROTOCOL["test_access_policy"]["final_test_access_count"] == 0


def test_rewrite_families_and_splits_are_disjoint():
    catalog = PROTOCOL["rewrite_catalog"]
    assert set(catalog["seen_families"]) == {
        "inverse_cancellation",
        "commuting_reorder",
        "rotation_fusion_split",
    }
    assert set(catalog["held_out_families"]) == {
        "identity_insertion_removal",
        "cross_decomposition",
        "nonlocal_operator_semantics",
    }
    assert not set(catalog["seen_families"]) & set(catalog["held_out_families"])
    assert {"rewrite_family_ood", "cross_decomposition_ood", "nonlocal_semantics_ood"} <= set(PROTOCOL["split_axes"])


def test_corpus_allocation_is_balanced_and_consistent():
    allocation = PROTOCOL["corpus_allocation"]
    for partition in ("train", "development"):
        values = allocation[partition]
        expected = len(values["families"]) * len(values["depths"]) * values["pairs_per_family_depth"]
        assert values["positive_pairs"] == expected == values["negative_pairs"]
    final = allocation["final_test"]
    subtotal = sum(v["positive_pairs"] for v in final.values() if isinstance(v, dict))
    assert subtotal == final["positive_pairs_total"] == final["negative_pairs_total"] == 21_504
    assert allocation["base_circuit_identity_disjoint_between_all_partitions"] is True


def test_exact_held_out_rewrite_templates_are_valid():
    h0, h1 = Gate("H", (0,)), Gate("H", (1,))
    c01, c10 = Gate("CNOT", (0, 1)), Gate("CNOT", (1, 0))
    assert phase_equivalent([c01], [h0, h1, c10, h0, h1])
    assert phase_equivalent([c01, c10, c01], [c10, c01, c10])
    assert phase_equivalent([], [Gate("H", (2,)), Gate("X", (2,)), Gate("X", (2,)), Gate("H", (2,))])
    a, b = 0.37, -1.12
    assert phase_equivalent(
        [Gate("RY", (3,), a), Gate("RY", (3,), b)],
        [Gate("RY", (3,), (a + b) % (2 * math.pi))],
    )


def test_selection_statistics_and_support_thresholds_are_predeclared():
    assert PROTOCOL["candidate_policy"]["seeds"] == [2026, 2027, 2028]
    assert PROTOCOL["candidate_policy"]["no_post_test_selection"] is True
    assert PROTOCOL["statistics"] == {
        "confidence_level": 0.95,
        "method": "paired stratified bootstrap over base-circuit identities",
        "resamples": 10000,
        "seed": 47011,
        "formal_significance": False,
        "family_macro_required": True,
        "negative_results_preserved": True,
        "no_single_aggregate_only": True,
    }
    thresholds = PROTOCOL["thresholds"]
    assert thresholds["semantic_action_fidelity_macro_minimum"] == 0.99
    assert thresholds["non_equivalent_control_rejection_minimum"] == 0.95
    assert thresholds["accepted_set_semantic_failure_rate_maximum"] == 0.01


def test_claims_and_north_star_are_guarded():
    assert PROTOCOL["classification"] == "ALIGNED_BENCHMARK"
    prohibited = " ".join(PROTOCOL["claim_boundaries"]["prohibited"])
    assert "QPU replacement" in prohibited
    assert "arbitrary-qubit" in prohibited
    assert "changes to QuTE North Star v1.0" in PROTOCOL["out_of_scope"]
    assert PROTOCOL["source_provenance"]["north_star_version"] == "1.0.0"


def test_frozen_file_hashes_pass():
    manifest = json.loads((ARTIFACT / "artifact_hashes.json").read_text())
    for relative, expected in manifest["files"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
