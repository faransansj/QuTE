import hashlib

import pytest

from pathlib import Path

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
