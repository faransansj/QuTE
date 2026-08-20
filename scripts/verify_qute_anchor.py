#!/usr/bin/env python3
"""Verify the immutable QuTE v1.0 governance anchor without opening sealed data."""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KO = "QuTE는 양자컴퓨터를 물리적으로 복제하는 프로젝트가 아니다.\nQuTE는 제한되고 검증된 양자 workload를 물리 제약이 반영된\n신경 연산자로 컴파일하여, 고전 AI 하드웨어에서 양자 백엔드처럼\n실행하고, 불확실하거나 범위 밖인 요청은 시뮬레이터 또는 QPU로\n전달하는 AI-native quantum runtime을 만드는 프로젝트다."
EN = "QuTE compiles restricted and verified quantum workloads into\nphysics-constrained neural execution backends running on classical\nAI hardware, while routing uncertain or out-of-domain requests to\nclassical simulators or physical QPUs."
SLOGAN = "Compile quantum workloads into verified neural execution backends."
HASHED = ["PROJECT_ANCHOR.md", "docs/QU_TE_NORTH_STAR.md", "docs/QU_TE_ROADMAP.md", "docs/QU_TE_RESEARCH_GOVERNANCE.md", "docs/ADR/0001-qute-north-star.md", "docs/templates/RESEARCH_PROPOSAL_GATE.md", "governance/qute_project_anchor_v1.json", "governance/NORTH_STAR_CHANGELOG.md"]
CLASSES = {"ALIGNED_CORE", "ALIGNED_RUNTIME", "ALIGNED_HARDWARE_TWIN", "ALIGNED_BENCHMARK", "ADJACENT_SEPARATE_PROJECT", "REJECTED_DIRECTION_DRIFT"}


def fail(message):
    raise AssertionError(message)


def main():
    required = HASHED + ["governance/qute_anchor_manifest.json", "README.md"]
    for name in required:
        if not (ROOT / name).is_file(): fail(f"missing: {name}")
    anchor = json.loads((ROOT / "governance/qute_project_anchor_v1.json").read_text())
    manifest = json.loads((ROOT / "governance/qute_anchor_manifest.json").read_text())
    if anchor["version"] != "1.0.0" or manifest["anchor_version"] != "1.0.0": fail("anchor version mismatch")
    if anchor["canonical_statement_ko"] != KO: fail("Korean statement mismatch")
    if anchor["canonical_statement_en"] != EN: fail("English statement mismatch")
    if anchor["slogan"] != SLOGAN: fail("slogan mismatch")
    if set(anchor["immutable_principles"]) != {f"P{i}" for i in range(1, 11)}: fail("immutable principles incomplete")
    if len(anchor["non_goals"]) != 10: fail("non-goals incomplete")
    if not all(anchor.get(k) for k in ("short_term_goals", "medium_term_goals", "long_term_goals")): fail("roadmap horizons incomplete")
    if set(anchor["proposal_gate"]["classifications"]) != CLASSES: fail("proposal classifications incomplete")
    verdict = anchor["p4_8_verdict"]
    expected = ("P4.8-SEALED-PARTIALLY-SUPPORTED", "SEALED-RECURRENT-QUALIFIED", "SEALED-COMPOSITION-NOT-SUPPORTED")
    if (verdict["overall"], verdict["recurrent"], verdict["composition"]) != expected or verdict["formal_significance_claimed"]: fail("P4.8 conclusions overstated")
    for name, want in manifest["files"].items():
        got = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if got != want: fail(f"hash mismatch: {name}")
    for paths in anchor["source_artifacts"].values():
        for name in paths:
            if not (ROOT / name).is_file(): fail(f"missing source artifact: {name}")
    for commit in anchor["source_commits"].values():
        if subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=ROOT).returncode: fail(f"missing source commit: {commit}")
    readme = (ROOT / "README.md").read_text()
    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", readme):
        if "://" not in target and not (ROOT / target.split("#")[0]).exists(): fail(f"broken README link: {target}")
    print(f"QuTE anchor {anchor['version']}: PASS ({len(manifest['files'])} hashes)")

if __name__ == "__main__":
    try: main()
    except (AssertionError, KeyError, json.JSONDecodeError) as exc:
        print(f"QuTE anchor: FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
