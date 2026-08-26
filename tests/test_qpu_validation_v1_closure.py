import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path("artifacts/qute_qpu_validation/20260826T043718Z-ibm_pittsburgh")
JOB_ID = "da76r2e0ukec7382tgcg"
NORTH_STAR_SHA256 = "39e65c7eb5078ff9a9898ef5543f1726508384cdd3e43bf1f36f846fee1d79c6"


def load(name):
    return json.loads((ROOT / name).read_text())


def test_execution_provenance_and_raw_counts_are_complete():
    job, backend, config = load("job_manifest.json"), load("backend_metadata.json"), load("config.json")
    results = load("per_circuit_results.json")
    raw = sorted((ROOT / "raw_counts").glob("*.json"))
    assert job["job_id"] == JOB_ID and job["status"] == "COMPLETED"
    assert backend["backend_name"] == "ibm_pittsburgh" and backend["simulator"] is False
    assert config["num_circuits"] == len(job["circuits"]) == len(results) == 20
    assert config["shots"] == job["shots"] == 4096
    assert len(raw) == 20
    assert all(sum(json.loads(path.read_text()).values()) == 4096 for path in raw)
    assert {row["job_id"] for row in results} == {JOB_ID}
    assert {row["shots"] for row in results} == {4096}
    assert all((ROOT / row["raw_counts_file"]).is_file() for row in results)


def test_original_checksums_and_closure_hashes_pass():
    for line in (ROOT / "checksums.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    ledger = load("artifact_hashes.json")
    assert ledger["original_checksums_verification"] == "PASS"
    assert ledger["raw_count_file_count"] == 20
    for record in ledger["files"]:
        path = ROOT / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_split_tvd_schema_and_overall_calculation():
    summary, results = load("summary.json"), load("per_circuit_results.json")
    expected = {
        "IID": (0.5057265339098619, 0.019332995459117715, 0.5102821007582372),
        "Parameter-OOD": (0.7815784320530572, 0.013428274383300748, 0.7759412102810146),
        "Composition-OOD": (0.9342713976676629, 0.031479209609962845, 0.9080239240137165),
        "Depth-OOD": (0.6723216598451401, 0.02357647855435905, 0.6613880580944775),
    }
    assert set(summary["by_split"]) == set(expected)
    for split, values in expected.items():
        block = summary["by_split"][split]
        assert block["count"] == 5
        assert tuple(block[key]["mean"] for key in ("tvd_model_sim", "tvd_sim_qpu", "tvd_model_qpu")) == values
    for key in ("tvd_model_sim", "tvd_sim_qpu", "tvd_model_qpu"):
        assert summary["overall"][key]["mean"] == statistics.fmean(row[key] for row in results)
        assert summary["overall"][key]["median"] == statistics.median(row[key] for row in results)


def test_error_budget_verdict_correlations_and_claim_boundaries():
    verdict, budget = load("scientific_verdict.json"), load("error_budget.json")
    assert verdict["verdict"] == budget["verdict"] == "MODEL_APPROXIMATION_DOMINANT"
    assert verdict["hardware_validation_status"] == "QPU_SIMULATOR_CLOSE_IN_THIS_FROZEN_SAMPLE"
    assert verdict["formal_significance_claimed"] is False
    assert verdict["hardware_correlations"]["label"] == budget["hardware_correlations"]["label"] == "DESCRIPTIVE_ONLY"
    assert verdict["hardware_correlations"]["sample_count"] == 20
    assert budget["hardware_correlations"]["formal_significance_claimed"] is False
    assert budget["hardware_correlations"]["causality_claimed"] is False
    prohibited = " ".join(budget["prohibited_claims"])
    assert "negligible in general" in prohibited
    assert "replace physical QPUs" in prohibited
    assert set(budget["hardware_correlations"]["hypotheses"]) == {"H_HW1", "H_HW2"}


def test_research_priority_r1_and_v2_guards():
    priority = load("research_priority_update.json")
    assert priority["qpu_role"] == "SPARSE_EXTERNAL_VALIDATION"
    assert priority["model_improvement_priority"] == "HIGH"
    assert priority["hardware_adapter_priority"] == "DEFERRED"
    assert [item["rank"] for item in priority["priorities"]] == [1, 2, 3, 4, 5]
    assert priority["r1_relationship"]["protocol_changes_authorized"] is False
    assert priority["r1_relationship"]["r1_data_generated"] is False
    assert priority["qpu_validation_v2"]["scheduled"] is False
    assert priority["qpu_validation_v2"]["final_numerical_threshold_selected"] is False
    assert priority["closure_actions"]["new_qpu_jobs_submitted"] == 0
    assert priority["closure_actions"]["model_training_executed"] == 0


def test_evidence_ledger_references_progression_and_qpu_result():
    ledger = Path("governance/SCIENTIFIC_EVIDENCE_LEDGER.md").read_text()
    for phase in ("P4.5", "P4.6", "P4.7", "P4.8", "QPU Validation v1"):
        assert phase in ledger
    assert JOB_ID in ledger
    assert "MODEL_APPROXIMATION_DOMINANT" in ledger
    assert "R1 Operator-Semantic Benchmark" in ledger


def test_north_star_is_byte_unchanged():
    path = Path("docs/QU_TE_NORTH_STAR.md")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == NORTH_STAR_SHA256
    assert "Compile quantum workloads into verified neural execution backends." in path.read_text()
    assert load("research_priority_update.json")["north_star_changed"] is False
