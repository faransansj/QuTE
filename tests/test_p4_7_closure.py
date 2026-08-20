import csv, hashlib, json
from pathlib import Path

from close_p4_7 import DATASET_HASH, SEEDS, SUPERVISION, VARIANTS, audit, sha

ROOT = Path("artifacts/cc_nqe_p4_7")


def test_p4_7_closure_archive_integrity():
    rows = audit()
    assert len(rows) == 12
    assert {(row["variant"], row["seed"]) for row in rows} == {(v, s) for v in VARIANTS for s in SEEDS}
    assert all(row["dataset_hash"] == DATASET_HASH for row in rows)
    assert all(row["supervision_class"] == SUPERVISION[row["variant"]] for row in rows)
    assert all(row["unitarity_error"] < 1e-5 for row in rows)
    assert len({row["config_hash"] for row in rows if row["variant"] == "C1"}) == 1
    assert all(row["source_phase"] == "P4.6_B3_anchor" for row in rows if (row["variant"], row["seed"]) == ("C0", 2026))

    summary = json.loads((ROOT / "confirmatory/summary.json").read_text())
    deltas = json.loads((ROOT / "confirmatory/paired_deltas.json").read_text())
    archived = list(csv.DictReader((ROOT / "confirmatory/paired_deltas.csv").open()))
    assert len(deltas) == len(archived)
    for delta in deltas:
        values = [delta[f"delta_{seed}"] for seed in SEEDS]
        assert abs(delta["mean_delta"] - sum(values) / 3) < 1e-12
    assert summary["cell_integrity"] == "12/12" and summary["sealed_test_access_count"] == 0

    verdict = json.loads((ROOT / "scientific_verdict.json").read_text())
    report = (ROOT / "P4_7_FINAL_REPORT.md").read_text()
    assert verdict["overall_verdict"] in report
    assert verdict["sealed_test_evaluated"] is False and verdict["sealed_test_access_count"] == 0
    assert verdict["multi_seed_confirmed"] is True
    assert verdict["composition_specialist_candidate"] == "C2"

    manifest = json.loads((ROOT / "artifact_hashes.json").read_text())
    assert all(sha(Path(path)) == expected for path, expected in manifest["artifacts"].items())
    assert not any("checkpoint" in path or "progress" in path or "status.json" in path for path in manifest["artifacts"])
