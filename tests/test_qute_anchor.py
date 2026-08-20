import hashlib, json, re
from pathlib import Path
from scripts.verify_qute_anchor import ROOT, HASHED, KO, EN, SLOGAN, CLASSES

A = json.loads((ROOT / "governance/qute_project_anchor_v1.json").read_text())

def test_schema_and_canonical_text():
    required = {"anchor_id","version","status","canonical_statement_ko","canonical_statement_en","slogan","immutable_principles","mutable_implementation_choices","non_goals","scientific_evidence","short_term_goals","medium_term_goals","long_term_goals","immediate_programs","project_boundaries","proposal_gate","drift_warnings","north_star_change_conditions","source_commits","source_artifacts","effective_date"}
    assert required <= A.keys()
    assert (A["canonical_statement_ko"], A["canonical_statement_en"], A["slogan"]) == (KO, EN, SLOGAN)

def test_governance_content():
    assert set(A["immutable_principles"]) == {f"P{i}" for i in range(1,11)}
    assert "loss functions" in A["mutable_implementation_choices"]
    assert len(A["non_goals"]) == 10
    assert all(A[k] for k in ("short_term_goals","medium_term_goals","long_term_goals"))
    assert len(A["proposal_gate"]["questions"]) == 10
    assert set(A["proposal_gate"]["classifications"]) == CLASSES
    assert len(A["drift_warnings"]) == 9
    assert {"policy","evidence","requirements"} <= A["north_star_change_conditions"].keys()
    assert "semantic version bump" in A["north_star_change_conditions"]["requirements"]
    assert set(A["project_boundaries"]) == {"QuTE Core","QuTE Runtime","QuTE Hardware Twin","Feynman","QuDDPM/QCNN","QuTE Benchmark"}

def test_manifest_and_readme_links():
    m=json.loads((ROOT/"governance/qute_anchor_manifest.json").read_text())
    assert set(m["files"]) == set(HASHED)
    for p,want in m["files"].items(): assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==want
    text=(ROOT/"README.md").read_text()
    for p in re.findall(r"\[[^]]+\]\(([^)]+)\)",text):
        if "://" not in p: assert (ROOT/p.split("#")[0]).exists()

def test_sources_and_p48_wording():
    for paths in A["source_artifacts"].values():
        for p in paths: assert (ROOT/p).is_file()
    v=A["p4_8_verdict"]
    assert v["overall"]=="P4.8-SEALED-PARTIALLY-SUPPORTED"
    assert v["recurrent"]=="SEALED-RECURRENT-QUALIFIED"
    assert v["composition"]=="SEALED-COMPOSITION-NOT-SUPPORTED"
    assert v["candidate_roles_unchanged"] and not v["formal_significance_claimed"]

def test_no_universal_or_physical_computer_claim():
    assert "claiming efficient exact simulation of arbitrary quantum circuits" in A["non_goals"]
    assert "claiming a classical neural network is a physical quantum computer" in A["non_goals"]
    assert "restricted and verified" in A["canonical_statement_en"]
