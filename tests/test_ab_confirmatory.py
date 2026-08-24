import json

import cc_nqe_ab_confirmatory as ab


def test_protocol_is_minimal_validation_only():
    row = ab.protocol()
    assert row["track_a"]["arms"] == ["A3", "A4"]
    assert row["track_b"]["variants"] == ["B0", "B1", "B3"]
    assert row["new_training_seeds"] == [2027, 2028]
    assert row["validation_only"] is True
    assert row["sealed_test_access"] == "PROHIBITED"


def test_predeclared_verdict_rules():
    assert ab.classify([1, 2, 3], [1, 2, 3], [1, 2, 3]) == {"A3": "SUPPORTED", "B3": "SUPPORTED"}
    assert ab.classify([1, -1, 1], [1, -1, 2], [1, 1, 1]) == {"A3": "QUALIFIED", "B3": "QUALIFIED"}
    assert ab.classify([-1, -1, 1], [1, 1, 1], [-2, 1, 0]) == {"A3": "NOT_SUPPORTED", "B3": "NOT_SUPPORTED"}


def test_prepare_freezes_hashes_without_opening_sealed(tmp_path, monkeypatch):
    monkeypatch.setattr(ab, "CONFIRM_ROOT", tmp_path)
    anchor = ab.prepare()
    assert anchor["sealed_test_access_count"] == 0
    assert json.loads((tmp_path / "protocol.json").read_text())["status"] == "FROZEN_NOT_RUN"
    assert ab.verify_anchor()["status"] == "PASS"
