import json
from pathlib import Path

import numpy as np
import pytest

import cc_nqe_p4_8 as p48


def test_p47_anchor_and_candidate_role_freeze():
    assert p48.verify_p47()["cells"] == "12/12"
    freeze = json.loads((p48.ROOT / "candidate_freeze.json").read_text())
    assert freeze["source_p4_7_commit"] == p48.SOURCE_COMMIT
    assert set(freeze["candidate_roles"]) == {"C0", "C1", "C2"}
    assert "C3" not in freeze["candidate_roles"] and "privileged" in freeze["excluded_C3"]


def test_all_nine_checkpoints_are_best_balanced_and_hash_verified():
    records = p48.candidate_records()
    assert [(r["variant"], r["seed"]) for r in records] == list(p48.ORDER)
    assert len({r["checkpoint_sha256"] for r in records}) == 9
    assert all(r["training_completion_status"] == "COMPLETED" for r in records)
    assert p48.verify_freeze() == {"status": "PASS", "candidate_count": 9, "candidate_freeze_sha256": p48.sha(p48.ROOT / "candidate_freeze.json")}


def test_sealed_metadata_validation_never_loads_arrays(monkeypatch):
    monkeypatch.setattr(np, "load", lambda *_a, **_k: pytest.fail("array load forbidden in preflight"))
    result = p48.verify_sealed_metadata()
    assert result["status"] == "PASS" and result["historical_access_count"] == 0
    assert all(not d["target_arrays_loaded"] for d in result["datasets"].values())


def test_frozen_endpoint_calculations_and_depth_diagnostics():
    m = p48.calculate_metrics(np.array([.6, .8]), {8: np.array([.9]), 9: np.array([.8]), 10: np.array([.7])})
    assert m["F_comp_test"] == pytest.approx(.7)
    assert m["F_depth_macro"] == pytest.approx(.8)
    assert m["S_sealed"] == pytest.approx(.75)
    assert m["depth_slope"] == pytest.approx(-.1)
    assert m["depth_8_to_10_degradation"] == pytest.approx(.2)


def test_paired_delta_calculation():
    rows = []
    for seed in p48.SEEDS:
        for i, variant in enumerate(p48.VARIANTS):
            metrics = {k: float(i) for k in ("S_sealed", "F_comp_test", "F_depth_macro", "F_depth_8", "F_depth_9", "F_depth_10")}
            rows.append({"variant": variant, "seed": seed, "metrics": metrics})
    deltas = p48.paired_deltas(rows)
    assert len(deltas) == 9
    assert all(x["S_sealed"] == 1 for x in deltas if x["comparison"] in ("C1_minus_C0", "C2_minus_C1"))


def test_deterministic_paired_bootstrap_is_example_level():
    left = {"comp_examples": [.8, .9], "depth_examples": {"8": [.8], "9": [.7], "10": [.6]}}
    right = {"comp_examples": [.7, .7], "depth_examples": {"8": [.7], "9": [.6], "10": [.5]}}
    a = p48.paired_bootstrap(left, right, resamples=100)
    b = p48.paired_bootstrap(left, right, resamples=100)
    assert a == b and a["label"] == "example-level bootstrap uncertainty"
    assert a["S_sealed_delta"]["estimate"] == pytest.approx(.125)


@pytest.mark.parametrize(("recurrent","composition","expected"), [
    ([1, 2, 3], [1, 2, 3], "P4.8-SEALED-HYPOTHESES-SUPPORTED"),
    ([1, -1, 1], [1, 2, 3], "P4.8-SEALED-PARTIALLY-SUPPORTED"),
    ([-1, -2, -3], [-1, -2, -3], "P4.8-SEALED-NOT-SUPPORTED"),
])
def test_predeclared_verdict_logic(recurrent, composition, expected):
    assert p48.verdicts(recurrent, composition)["overall"] == expected


def test_no_post_test_selection_is_frozen_in_protocol():
    protocol = json.loads((p48.ROOT / "protocol.json").read_text())
    assert protocol["no_post_test_selection"] is True and protocol["no_more_tuning"] is True
    assert protocol["candidates"] == p48.ROLES


def test_unlock_token_derivation_wrong_token_and_staleness():
    current = {"protocol_sha256": "protocol", "candidate_freeze_sha256": "freeze", "implementation_commit": "commit"}
    token = p48.unlock_token("protocol", "freeze", "commit")
    manifest = {**current, "unlock_token": token}
    p48.validate_unlock_token(token, manifest, current)
    for wrong in (None, "wrong"):
        with pytest.raises(RuntimeError, match="missing, wrong, or stale"):
            p48.validate_unlock_token(wrong, manifest, current)
    with pytest.raises(RuntimeError, match="missing, wrong, or stale"):
        p48.validate_unlock_token(token, manifest, {**current, "implementation_commit": "changed"})


def test_access_count_and_status_schema_are_not_run():
    access = json.loads((p48.ROOT / "sealed_access_log.json").read_text())
    status = p48.status()
    assert access["state"] == "PREPARED" and access["access_count"] == 0 and access["sealed_test_evaluated"] is False
    assert status["status"] == "NOT_RUN" and status["sealed_scientific_evaluation"] == "NONE"


def test_atomic_write_replaces_complete_file(tmp_path):
    path = tmp_path / "state.json"
    p48.dump(path, {"state": "STARTED"}); p48.dump(path, {"state": "COMPLETED"})
    assert json.loads(path.read_text()) == {"state": "COMPLETED"}
    assert not list(tmp_path.glob("*.tmp"))


def test_xpu_native_inference_has_no_cpu_fallback():
    result = p48.xpu_preflight()
    assert result == {"status": "PASS", "device": "xpu:0", "native_cayley": True, "cpu_fallback": False}


def test_retry_requires_matching_started_transaction(tmp_path, monkeypatch):
    root = tmp_path / "p48"; root.mkdir()
    (root / "sealed_access_log.json").write_text(json.dumps({"state": "PREPARED", "access_count": 0, "transaction": None}))
    (root / "unlock_manifest.json").write_text("{}")
    monkeypatch.setattr(p48, "ROOT", root)
    with pytest.raises(RuntimeError, match="SEALED-RETRY-BLOCKED"):
        p48.sealed_evaluate(None, "wrong")


def test_dry_run_artifact_contract_if_present():
    path = p48.ROOT / "tests/dry_run_report.json"
    if not path.exists(): pytest.skip("dry-run executed after unit suite")
    value = json.loads(path.read_text())
    assert value["scientific_run"] is False and value["sealed_data_used"] is False
    assert value["purpose"] == "implementation_validation" and value["access_count"] == 0
