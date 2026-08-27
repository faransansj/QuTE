import hashlib
from pathlib import Path

import pytest

from r1_corpus import build_corpus, load_partition, write_corpus


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "artifacts" / "r1_operator_semantic_benchmark" / "protocol.json"


def test_smoke_corpus_is_deterministic_and_oracle_valid():
    first = build_corpus(PROTOCOL, mode="smoke")
    second = build_corpus(PROTOCOL, mode="smoke")

    assert first == second
    assert first["audit"]["status"] == "PASS"
    assert first["audit"]["positive_oracle_failures"] == 0
    assert first["audit"]["negative_oracle_failures"] == 0
    assert first["audit"]["partition_leakage"] == []
    assert first["audit"]["counts"] == {
        "train": {"equivalent": 3, "non_equivalent": 3},
        "development": {"equivalent": 3, "non_equivalent": 3},
        "final_test": {"equivalent": 21, "non_equivalent": 21},
    }


def test_write_corpus_verifies_regeneration_checksums_and_final_guard(tmp_path):
    output = tmp_path / "smoke"
    manifest = write_corpus(output, PROTOCOL)

    assert manifest["deterministic_regeneration"] is True
    for line in (output / "checksums.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    assert load_partition(output, "train")
    with pytest.raises(PermissionError, match="final_test"):
        load_partition(output, "final_test")
    with pytest.raises(FileExistsError, match="overwrite"):
        write_corpus(output, PROTOCOL)


def test_smoke_covers_frozen_rewrite_families_and_probe_boundary():
    corpus = build_corpus(PROTOCOL)
    positives = [
        record
        for records in corpus["records"].values()
        for record in records
        if record["label"] == "equivalent"
    ]

    assert {record["rewrite_family"] for record in positives} == {
        "inverse_cancellation",
        "commuting_reorder",
        "rotation_fusion_split",
        "identity_insertion_removal",
        "cross_decomposition",
        "nonlocal_operator_semantics",
    }
    assert all(record["oracle"]["pass"] for record in positives)
    assert all(record["probe"]["family"] == "product" for record in positives if record["partition"] != "final_test")
    assert {record["probe"]["family"] for record in positives if record["partition"] == "final_test"} == {"entangled", "haar"}


def test_full_generation_requires_explicit_authorization():
    with pytest.raises(PermissionError, match="authorize_full"):
        build_corpus(PROTOCOL, mode="full")


def test_explicit_full_test_access_requires_reason_and_is_logged(tmp_path):
    root = tmp_path / "full"
    root.mkdir()
    (root / "manifest.json").write_text('{"mode":"full"}\n')
    (root / "final_test.sealed.jsonl").write_text('{"pair_id":"sealed-example"}\n')
    (root / "FINAL_TEST_ACCESS_LOG.jsonl").write_text("")

    with pytest.raises(ValueError, match="access_reason"):
        load_partition(root, "final_test", allow_final=True)
    assert load_partition(root, "final_test", allow_final=True, access_reason="one untouched evaluation") == [
        {"pair_id": "sealed-example"}
    ]
    log = (root / "FINAL_TEST_ACCESS_LOG.jsonl").read_text()
    assert "one untouched evaluation" in log
    assert "sealed-example" not in log
