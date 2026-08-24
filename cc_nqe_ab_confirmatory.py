"""Frozen minimal multi-seed confirmation of P4.6 A/B screening choices."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIRM_ROOT = PROJECT_ROOT / "artifacts/cc_nqe_ab_confirmatory"
SEEDS = (2026, 2027, 2028)
A_ARMS = ("A3", "A4")
B_VARIANTS = ("B0", "B1", "B3")
SCHEMA = "cc-nqe-ab-confirmatory-v1"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def protocol() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "status": "FROZEN_NOT_RUN",
        "purpose": "minimal multi-seed confirmation of the P4.6 A3 and B3 validation selections",
        "seeds": list(SEEDS),
        "reuse_seed_2026": True,
        "new_training_seeds": [2027, 2028],
        "track_a": {"arms": list(A_ARMS), "primary_comparison": "A3_minus_A4"},
        "track_b": {"variants": list(B_VARIANTS), "primary_comparisons": ["B3_minus_B0", "B3_minus_B1"]},
        "checkpoint_selection": "best balanced validation",
        "balanced_score": "mean(IID, Composition-OOD, Depth-OOD fidelity)",
        "validation_only": True,
        "sealed_test_access": "PROHIBITED",
        "no_post_result_tuning": True,
        "formal_significance": False,
        "verdict_rules": {
            "A3": "SUPPORTED if A3-A4 > 0 for every seed; QUALIFIED if mean > 0 with mixed signs; otherwise NOT_SUPPORTED",
            "B3": "SUPPORTED if B3-B0 and B3-B1 are > 0 for every seed; QUALIFIED if both means > 0 with mixed signs; otherwise NOT_SUPPORTED",
        },
    }


def anchor_sources() -> list[Path]:
    paths = [
        PROJECT_ROOT / "cc_nqe_p4_6.py",
        PROJECT_ROOT / "cc_nqe_p4_6_track_a.py",
        PROJECT_ROOT / "cc_nqe_p4_6_track_b.py",
        PROJECT_ROOT / "cc_nqe_ab_confirmatory.py",
        PROJECT_ROOT / "governance/qute_anchor_manifest.json",
    ]
    for arm in A_ARMS:
        paths += [
            PROJECT_ROOT / f"artifacts/cc_nqe_p4_6/factorial/configs/{arm}.json",
            PROJECT_ROOT / f"artifacts/cc_nqe_p4_6/factorial/metrics/{arm}.json",
        ]
    for variant in B_VARIANTS:
        paths += [
            PROJECT_ROOT / f"artifacts/cc_nqe_p4_6/operator/configs/{variant}.json",
            PROJECT_ROOT / f"artifacts/cc_nqe_p4_6/operator/metrics/{variant}.json",
        ]
    return paths


def prepare() -> dict[str, Any]:
    missing = [str(p) for p in anchor_sources() if not p.exists()]
    if missing:
        raise FileNotFoundError(missing)
    access = json.loads((PROJECT_ROOT / "artifacts/cc_nqe_p4_6/test_access_log.json").read_text())
    if access.get("access_count") != 0:
        raise RuntimeError("sealed-test access is nonzero")
    for path in anchor_sources():
        if "/metrics/" in str(path):
            row = json.loads(path.read_text())
            if row.get("state") != "COMPLETED":
                raise RuntimeError(f"incomplete source metric: {path}")
    CONFIRM_ROOT.mkdir(parents=True, exist_ok=True)
    frozen = protocol()
    atomic_json(CONFIRM_ROOT / "protocol.json", frozen)
    anchor = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "source_hashes": {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in anchor_sources()},
        "dataset_manifest_hashes": {
            "A3": json.loads((PROJECT_ROOT / "artifacts/cc_nqe_p4_6/factorial/configs/A3.json").read_text())["dataset_manifest_hash"],
            "A4": json.loads((PROJECT_ROOT / "artifacts/cc_nqe_p4_6/factorial/configs/A4.json").read_text())["dataset_manifest_hash"],
            "B_A4": json.loads((PROJECT_ROOT / "artifacts/cc_nqe_p4_6/operator/configs/B3.json").read_text())["dataset_manifest_hash"],
        },
        "sealed_test_access_count": 0,
    }
    atomic_json(CONFIRM_ROOT / "anchor.json", anchor)
    return anchor


def verify_anchor() -> dict[str, Any]:
    path = CONFIRM_ROOT / "anchor.preflight-seed-amendment-v1.json"
    if not path.exists():
        path = CONFIRM_ROOT / "anchor.json"
    anchor = json.loads(path.read_text())
    failures = [name for name, expected in anchor["source_hashes"].items() if not (PROJECT_ROOT / name).exists() or sha256(PROJECT_ROOT / name) != expected]
    if failures:
        raise RuntimeError(f"anchor differs: {failures}")
    return anchor


@contextlib.contextmanager
def _track_a_context(seed: int) -> Iterator[Any]:
    import cc_nqe_p4_6_track_a as track
    old = track.TRACK_ROOT, track.SEED, track.RECIPE, track._STOP
    root = CONFIRM_ROOT / f"runs/seed-{seed}/track_a"
    recipe = {**track.RECIPE, "seed": seed}
    track.TRACK_ROOT, track.SEED, track.RECIPE, track._STOP = root, seed, recipe, False
    try:
        yield track
    finally:
        track.TRACK_ROOT, track.SEED, track.RECIPE, track._STOP = old


@contextlib.contextmanager
def _track_b_context(seed: int) -> Iterator[Any]:
    import cc_nqe_p4_6_track_b as track
    old = track.OP_ROOT, track.SEED, track.RECIPE, track._STOP
    root = CONFIRM_ROOT / f"runs/seed-{seed}/track_b"
    recipe = {**track.RECIPE, "seed": seed}
    track.OP_ROOT, track.SEED, track.RECIPE, track._STOP = root, seed, recipe, False
    try:
        yield track
    finally:
        track.OP_ROOT, track.SEED, track.RECIPE, track._STOP = old


def run_cell(kind: str, name: str, seed: int) -> dict[str, Any]:
    verify_anchor()
    if seed not in (2027, 2028):
        raise ValueError("seed 2026 is reused; only seeds 2027 and 2028 may train")
    if kind == "A" and name in A_ARMS:
        with _track_a_context(seed) as track:
            return track.train_arm(name)
    if kind == "B" and name in B_VARIANTS:
        with _track_b_context(seed) as track:
            return track.operator_run(name)
    raise ValueError(f"unknown cell: {kind} {name}")


def _score(kind: str, name: str, seed: int) -> float:
    if seed == 2026:
        root = PROJECT_ROOT / ("artifacts/cc_nqe_p4_6/factorial/metrics" if kind == "A" else "artifacts/cc_nqe_p4_6/operator/metrics")
    else:
        root = CONFIRM_ROOT / f"runs/seed-{seed}" / ("track_a/metrics" if kind == "A" else "track_b/metrics")
    row = json.loads((root / f"{name}.json").read_text())
    return float(row["primary_checkpoint"]["metrics"]["balanced_validation"] if kind == "A" else row["best_balanced_validation"])


def classify(a_deltas: list[float], b0_deltas: list[float], b1_deltas: list[float]) -> dict[str, str]:
    def one(values: list[float]) -> str:
        return "SUPPORTED" if all(x > 0 for x in values) else "QUALIFIED" if sum(values) / len(values) > 0 else "NOT_SUPPORTED"
    a = one(a_deltas)
    if all(x > 0 for x in b0_deltas + b1_deltas):
        b = "SUPPORTED"
    elif sum(b0_deltas) / len(b0_deltas) > 0 and sum(b1_deltas) / len(b1_deltas) > 0:
        b = "QUALIFIED"
    else:
        b = "NOT_SUPPORTED"
    return {"A3": a, "B3": b}


def aggregate() -> dict[str, Any]:
    verify_anchor()
    scores = {
        str(seed): {
            "A": {name: _score("A", name, seed) for name in A_ARMS},
            "B": {name: _score("B", name, seed) for name in B_VARIANTS},
        }
        for seed in SEEDS
    }
    a = [scores[str(seed)]["A"]["A3"] - scores[str(seed)]["A"]["A4"] for seed in SEEDS]
    b0 = [scores[str(seed)]["B"]["B3"] - scores[str(seed)]["B"]["B0"] for seed in SEEDS]
    b1 = [scores[str(seed)]["B"]["B3"] - scores[str(seed)]["B"]["B1"] for seed in SEEDS]
    result = {
        "schema_version": SCHEMA,
        "scores": scores,
        "paired_deltas": {"A3_minus_A4": a, "B3_minus_B0": b0, "B3_minus_B1": b1},
        "verdict": classify(a, b0, b1),
        "formal_significance_claimed": False,
        "sealed_test_evaluated": False,
    }
    atomic_json(CONFIRM_ROOT / "result.json", result)
    return result
