import hashlib
import json
from pathlib import Path

import pytest

from cc_nqe import Gate
from r1_corpus import (
    ORACLE_PROBE_SET_ID,
    SCIENTIFIC_DEVELOPMENT_NAMESPACE,
    SCIENTIFIC_SEALED_NAMESPACE,
    SEED_PREFIX,
    SEED_SCHEMA,
    SMOKE_V2_NAMESPACE,
    canonical_circuit,
    canonical_json_bytes,
    circuit_content_sha256,
    content_hash_intersections,
    derive_seed,
    generate_scientific_development,
    generate_scientific_sealed,
    load_corpus_by_role,
    namespaced_id,
    pair_content_hashes,
    rewrite_coverage_complete,
    semantic_class_sha256,
    validate_lifecycle_transition,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "r1_operator_semantic_benchmark"
PROTOCOL = ARTIFACT / "protocol.json"
REGISTRY = ARTIFACT / "registry" / "namespace_registry.json"
COVERAGE = ARTIFACT / "protocol" / "rewrite_coverage_matrix.json"
OVERLAP_POLICY = ARTIFACT / "protocol" / "overlap_policy.json"
SMOKE_V1 = ARTIFACT / "smoke_v1"
SMOKE_V2 = ARTIFACT / "smoke_v2"
AUDITS = ARTIFACT / "audits"
PROTOCOL_HASH = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def seed(namespace: str):
    return derive_seed(
        protocol_sha256=PROTOCOL_HASH,
        generation_namespace=namespace,
        corpus_role="SMOKE_DEVELOPMENT",
        partition_role="SMOKE_COVERAGE",
        master_seed=2026,
        rewrite_family="inverse_cancellation",
        template_id="inverse_rotation:RX",
        local_index=3,
    )


def test_namespace_registry_and_roles_are_explicit():
    registry = read_json(REGISTRY)
    assert set(registry["roles"]) == {
        "SMOKE_DEVELOPMENT",
        "SCIENTIFIC_DEVELOPMENT",
        "SCIENTIFIC_SEALED_FINAL",
    }
    assert set(registry["namespaces"]) == {
        "qute:r1:smoke:v1",
        SMOKE_V2_NAMESPACE,
        SCIENTIFIC_DEVELOPMENT_NAMESPACE,
        SCIENTIFIC_SEALED_NAMESPACE,
    }
    assert registry["roles"]["SMOKE_DEVELOPMENT"] == {
        "scientific_use": "FORBIDDEN",
        "sealable": False,
        "promotion_allowed": False,
    }


def test_lifecycle_blocks_retired_reuse_and_role_promotion():
    registry = read_json(REGISTRY)
    retired = registry["namespaces"]["qute:r1:smoke:v1"]
    with pytest.raises(PermissionError, match="retired"):
        validate_lifecycle_transition(retired, "SMOKE_DEVELOPMENT")
    smoke_v2 = registry["namespaces"][SMOKE_V2_NAMESPACE]
    with pytest.raises(PermissionError, match="promotion"):
        validate_lifecycle_transition(smoke_v2, "SCIENTIFIC_DEVELOPMENT")
    validate_lifecycle_transition(smoke_v2, "SMOKE_DEVELOPMENT")


def test_seed_derivation_matches_frozen_sha256_contract():
    coordinates = {
        "seed_schema": SEED_SCHEMA,
        "protocol_sha256": PROTOCOL_HASH,
        "generation_namespace": SMOKE_V2_NAMESPACE,
        "corpus_role": "SMOKE_DEVELOPMENT",
        "partition_role": "SMOKE_COVERAGE",
        "master_seed": 2026,
        "rewrite_family": "inverse_cancellation",
        "template_id": "inverse_rotation:RX",
        "local_index": 3,
    }
    expected = int.from_bytes(
        hashlib.sha256(SEED_PREFIX + canonical_json_bytes(coordinates)).digest()[:8],
        "big",
    )
    assert seed(SMOKE_V2_NAMESPACE) == expected == seed(SMOKE_V2_NAMESPACE)


def test_namespace_change_changes_seed():
    assert len(
        {
            seed(SMOKE_V2_NAMESPACE),
            seed(SCIENTIFIC_DEVELOPMENT_NAMESPACE),
            seed(SCIENTIFIC_SEALED_NAMESPACE),
        }
    ) == 3


def test_namespaced_ids_change_but_content_hash_does_not():
    circuit = [Gate("H", (0,)), Gate("RZ", (1,), 0.5)]
    digest = circuit_content_sha256(circuit)
    assert namespaced_id("circuit", SMOKE_V2_NAMESPACE, digest) != namespaced_id(
        "circuit", SCIENTIFIC_DEVELOPMENT_NAMESPACE, digest
    )
    assert digest == circuit_content_sha256(circuit)


def test_canonical_circuit_serialization_excludes_metadata():
    circuit = [Gate("RX", (2,), 0.1), Gate("CNOT", (2, 3))]
    canonical = canonical_circuit(circuit)
    assert canonical == {
        "n_qubits": 4,
        "gates": [
            {
                "position": 0,
                "gate": "RX",
                "qubits": [2],
                "parameters": [float(0.1).hex()],
            },
            {
                "position": 1,
                "gate": "CNOT",
                "qubits": [2, 3],
                "parameters": [],
            },
        ],
    }
    assert not ({"namespace", "partition", "timestamp", "record_id"} & set(canonical))


def test_ordered_and_unordered_pair_hashes():
    ordered_ab, unordered_ab = pair_content_hashes("a", "b")
    ordered_ba, unordered_ba = pair_content_hashes("b", "a")
    assert ordered_ab != ordered_ba
    assert unordered_ab == unordered_ba


def test_equivalence_class_and_pair_ledgers_have_required_schema():
    classes = read_jsonl(SMOKE_V2 / "equivalence_classes.jsonl")
    pairs = read_jsonl(SMOKE_V2 / "pairs.jsonl")
    assert len(classes) == 15
    assert len(pairs) == 30
    required_class = {
        "equivalence_class_id",
        "semantic_class_sha256",
        "base_circuit_id",
        "base_circuit_content_sha256",
        "semantic_operator_sha256",
        "rewrite_family",
        "template_id",
        "coverage_cell_id",
        "rewrite_chain_length",
        "variant_circuit_ids",
        "pair_ids",
        "class_cardinality",
        "derived_seed",
        "protocol_sha256",
        "generator_commit",
    }
    required_pair = {
        "pair_id",
        "equivalence_class_id",
        "source_circuit_id",
        "target_circuit_id",
        "source_circuit_content_sha256",
        "target_circuit_content_sha256",
        "ordered_pair_content_sha256",
        "unordered_pair_content_sha256",
        "rewrite_instance_content_sha256",
        "process_fidelity",
        "process_infidelity",
        "phase_aligned_relative_frobenius",
        "difficulty_stratum",
        "oracle_probe_verification",
    }
    assert all(required_class <= set(record) for record in classes)
    assert all(required_pair <= set(record) for record in pairs)
    assert all(record["class_cardinality"] == 2 for record in classes)


def test_semantic_class_hash_is_namespace_independent():
    operator = "a" * 64
    assert semantic_class_sha256(operator) == semantic_class_sha256(operator)
    classes = read_jsonl(SMOKE_V2 / "equivalence_classes.jsonl")
    assert all(
        record["semantic_class_sha256"]
        == semantic_class_sha256(record["semantic_operator_sha256"])
        for record in classes
    )


def test_smoke_v1_is_retired_nonsealable_evidence():
    manifest = read_json(SMOKE_V1 / "corpus_manifest.json")
    retirement = read_json(SMOKE_V1 / "retirement_manifest.json")
    assert manifest["status"] == "NON_SEALABLE_RETIRED_EVIDENCE"
    assert manifest["scientific_use"] == "FORBIDDEN"
    assert manifest["sealable"] is False
    assert manifest["promotion_allowed"] is False
    assert manifest["reuse_allowed"] is False
    assert retirement["retirement_reason"] == "FUTURE_SCIENTIFIC_OVERLAP_RISK"
    assert retirement["future_plan_overlap_count"] == 21
    assert retirement["future_plan_compared_count"] == 21


def test_rewrite_coverage_matrix_is_complete_and_realized():
    matrix = read_json(COVERAGE)
    cells = matrix["cells"]
    assert len(cells) == 15
    assert rewrite_coverage_complete(matrix)
    required_ids = {
        "inverse-rx",
        "inverse-rz",
        "commute-rz-rz",
        "fusion-rx",
        "fusion-ry",
        "identity-cnot-hh-cnot",
    }
    assert required_ids <= {cell["coverage_cell_id"] for cell in cells}
    realized = {
        record["coverage_cell_id"]
        for record in read_jsonl(SMOKE_V2 / "equivalence_classes.jsonl")
    }
    assert realized == {cell["coverage_cell_id"] for cell in cells}


def test_coverage_gate_fails_if_any_required_cell_is_unimplemented():
    matrix = read_json(COVERAGE)
    matrix["cells"][0]["implemented"] = False
    assert rewrite_coverage_complete(matrix) is False


def test_negative_controls_are_named_without_hard_negative_claim():
    negatives = [
        record
        for record in read_jsonl(SMOKE_V2 / "pairs.jsonl")
        if record["label"] == "NEGATIVE_NON_EQUIVALENT"
    ]
    assert len(negatives) == 15
    assert all(record["pair_role"] == "MATCHED_NON_EQUIVALENT_CONTROL" for record in negatives)
    assert all(record["difficulty_stratum"] == "FAR_NEGATIVE_SANITY" for record in negatives)
    assert all(record["phase_aligned_relative_frobenius"] >= 0.1 for record in negatives)
    assert all(0.0 <= record["process_infidelity"] <= 1.0 for record in negatives)


def test_oracle_and_model_probe_metadata_are_distinct():
    pairs = read_jsonl(SMOKE_V2 / "pairs.jsonl")
    assert all(
        record["oracle_probe_verification"]["oracle_probe_set_id"]
        == ORACLE_PROBE_SET_ID
        for record in pairs
    )
    assert all(record["oracle_probe_verification"]["basis_probe_count"] == 16 for record in pairs)
    assert all("model_probe_id" in record and "model_probe_content_sha256" in record for record in pairs)
    policy = read_json(OVERLAP_POLICY)
    assert "oracle_probe_set_id" in policy["exempt_fields"]
    assert "model_probe_content_sha256" in policy["zero_overlap_fields"]


def test_overlap_policy_requires_content_hashes_not_only_ids():
    policy = read_json(OVERLAP_POLICY)
    assert len(policy["prohibited_role_pairs"]) == 3
    assert policy["namespaced_id_uniqueness_required"] is True
    assert policy["namespaced_id_uniqueness_is_sufficient"] is False
    assert set(policy["zero_overlap_fields"]) == {
        "circuit_content_sha256",
        "base_circuit_content_sha256",
        "ordered_pair_content_sha256",
        "unordered_pair_content_sha256",
        "rewrite_instance_content_sha256",
        "semantic_class_sha256",
        "model_probe_content_sha256",
    }


def test_overlap_detection_reports_examples():
    fields = ["circuit_content_sha256", "semantic_class_sha256"]
    result = content_hash_intersections(
        {"circuit_content_sha256": {"a", "b"}, "semantic_class_sha256": {"x"}},
        {"circuit_content_sha256": {"b", "c"}, "semantic_class_sha256": set()},
        fields,
    )
    assert result["circuit_content_sha256"] == {"count": 1, "examples": ["b"]}
    assert result["semantic_class_sha256"] == {"count": 0, "examples": []}


def test_namespace_isolation_and_intersection_audits_pass():
    namespace = read_json(AUDITS / "namespace_isolation_audit.json")
    intersection = read_json(AUDITS / "intersection_audit.json")
    assert namespace["status"] == "PASS"
    assert namespace["derived_seed_collision_count"] == 0
    assert namespace["scientific_payloads_generated"] is False
    assert intersection["status"] == "PASS"
    assert all(
        result["count"] == 0
        for result in intersection["smoke_v1_vs_smoke_v2"].values()
    )
    assert intersection["scientific_development_payload_generated"] is False
    assert intersection["scientific_sealed_payload_generated"] is False


def test_smoke_v2_determinism_and_checksums_pass():
    audit = read_json(AUDITS / "determinism_audit.json")
    manifest = read_json(SMOKE_V2 / "corpus_manifest.json")
    assert audit["status"] == "PASS"
    assert audit["byte_mismatch_count"] == 0
    assert manifest["generation_namespace"] == SMOKE_V2_NAMESPACE
    assert manifest["corpus_role"] == "SMOKE_DEVELOPMENT"
    assert manifest["scientific_use"] == "FORBIDDEN"
    for line in (SMOKE_V2 / "checksums.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        assert hashlib.sha256((SMOKE_V2 / relative).read_bytes()).hexdigest() == expected


def test_all_required_sorted_indices_exist():
    expected = {
        "record_ids.txt",
        "circuit_content_hashes.txt",
        "base_circuit_content_hashes.txt",
        "ordered_pair_content_hashes.txt",
        "unordered_pair_content_hashes.txt",
        "equivalence_class_ids.txt",
        "semantic_class_hashes.txt",
        "rewrite_instance_hashes.txt",
        "model_probe_content_hashes.txt",
    }
    actual = {path.name for path in (SMOKE_V2 / "indices").iterdir()}
    assert actual == expected
    for path in (SMOKE_V2 / "indices").iterdir():
        values = path.read_text().splitlines()
        assert values == sorted(set(values))


def test_scientific_loader_rejects_retired_smoke():
    with pytest.raises(PermissionError, match="retired"):
        load_corpus_by_role(SMOKE_V1, loader_role="scientific_development")
    assert load_corpus_by_role(SMOKE_V1, loader_role="audit")["status"] == (
        "NON_SEALABLE_RETIRED_EVIDENCE"
    )


def test_normal_loader_rejects_sealed_role(tmp_path):
    root = tmp_path / "sealed"
    root.mkdir()
    (root / "corpus_manifest.json").write_text(
        json.dumps(
            {
                "corpus_role": "SCIENTIFIC_SEALED_FINAL",
                "scientific_use": "FINAL_EVALUATION_ONLY",
                "status": "PLANNED",
            }
        )
    )
    with pytest.raises(PermissionError, match="sealed"):
        load_corpus_by_role(root, loader_role="development")


def test_scientific_generation_commands_remain_blocked():
    with pytest.raises(PermissionError, match="development"):
        generate_scientific_development()
    with pytest.raises(PermissionError, match="sealed"):
        generate_scientific_sealed()


def test_no_scientific_or_sealed_payload_was_generated():
    registry = read_json(REGISTRY)
    assert registry["namespaces"][SCIENTIFIC_DEVELOPMENT_NAMESPACE]["payload_generated"] is False
    sealed = registry["namespaces"][SCIENTIFIC_SEALED_NAMESPACE]
    assert sealed["payload_generated"] is False
    assert sealed["sealed_state"] == {"state": "PLANNED", "access_count": 0, "evaluated": False}
    assert not (ARTIFACT / "scientific_development_v1").exists()
    assert not (ARTIFACT / "scientific_sealed_v1").exists()


def test_full_generation_gate_stays_blocked():
    gate = read_json(AUDITS / "full_generation_gate.json")
    assert gate["status"] == "BLOCKED"
    assert gate["authorized"] is False
    assert gate["blockers"] == [
        "generator_commit_match",
        "scientific_namespace_uniquely_authorized",
    ]
    assert gate["development_command_separate"] is True
    assert gate["sealed_command_separate"] is True
