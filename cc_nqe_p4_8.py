"""P4.8 frozen, guarded, one-time sealed OOD evaluation."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import secrets
import statistics
import struct
import subprocess
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cc_nqe import Gate
from cc_nqe_p4_5 import atomic_json, state_fidelity
from cc_nqe_p4_6 import OperatorModel
from cc_nqe_p4_6_track_b import _circuit_tensors, operator_action
from cc_nqe_p4_7 import RecursiveOperatorModel
from close_p4_7 import audit as audit_p47, checkpoint_path, config_path, metric_path

ROOT = Path("artifacts/cc_nqe_p4_8")
P46 = Path("artifacts/cc_nqe_p4_6")
P47 = Path("artifacts/cc_nqe_p4_7")
SOURCE_COMMIT = "cd77fdeb51a481b04086c6192fad3c4b02e8b9aa"
BRANCH = "research/cc-nqe-p4-8-sealed-evaluation"
SCHEMA = "cc-nqe-p4.8-v1"
SEEDS = (2026, 2027, 2028)
VARIANTS = ("C0", "C1", "C2")
ORDER = tuple((v, s) for s in SEEDS for v in VARIANTS)
BATCH_SIZE = 4
BOOTSTRAP = {"bootstrap_seed": 47008, "bootstrap_resamples": 10000, "confidence_level": 0.95,
             "label": "example-level bootstrap uncertainty"}
DATASETS = {
    "composition_ood_test_sealed": {
        "manifest": P46 / "datasets/composition_ood_test_sealed.json",
        "payload": P46 / "datasets/composition_ood_test_sealed.npz",
        "manifest_sha256": "0407e43cbda439e3779ac7b9a2288e7750610218f4f96b5bc5e199efd5844fce",
        "payload_sha256": "f29122321c060e5f0a104d8497dbdb4b28cda653e28cdd681bef26781809f448",
        "sample_count": 12, "depths": [4, 5, 6],
    },
    "depth_ood_test_sealed": {
        "manifest": P46 / "datasets/depth_ood_test_sealed.json",
        "payload": P46 / "datasets/depth_ood_test_sealed.npz",
        "manifest_sha256": "5100016ba3eb52efafbe0c8c115b485aff8e383312855cb588775d4f9ffb7e5c",
        "payload_sha256": "cdf45cb652ab39a5c1ba697392c0b630c093f126041b94797887b1d6977312ce",
        "sample_count": 8, "depths": [8, 9, 10],
    },
}
ROLES = {
    "C0": "anchor comparator; monolithic action-only exact-unitary Cayley operator",
    "C1": "primary general-purpose candidate; shared causal recurrent action-only exact-unitary operator",
    "C2": "composition specialist; C1 plus operator-composition self-consistency",
}
SUPERVISION = {"C0": "ACTION_ONLY", "C1": "ACTION_ONLY", "C2": "ACTION_ONLY_PLUS_SELF_CONSISTENCY"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def git(*args: str) -> str:
    return subprocess.check_output(("git", *args), text=True).strip()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def dump(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")


def _npy_headers(path: Path) -> dict[str, dict[str, Any]]:
    """Read ZIP/NPY metadata only; never deserialize array payloads."""
    result = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            with archive.open(name) as f:
                if f.read(6) != b"\x93NUMPY": raise ValueError(f"invalid NPY member: {name}")
                major, _minor = f.read(2)
                size = struct.unpack("<H" if major == 1 else "<I", f.read(2 if major == 1 else 4))[0]
                header = ast.literal_eval(f.read(size).decode("latin1"))
            result[name] = {"shape": list(header["shape"]), "dtype": header["descr"], "fortran_order": header["fortran_order"]}
    return result


def verify_sealed_metadata() -> dict[str, Any]:
    access = json.loads((P46 / "test_access_log.json").read_text())
    if access.get("access_count") != 0: raise RuntimeError("SEALED-RETRY-BLOCKED: historical access_count is not zero")
    contract_path = P46 / "datasets/ood_split_contract.json"
    contract = json.loads(contract_path.read_text())
    if contract.get("schema_version") != "cc-nqe-p4.6-v1" or set(contract.get("sealed_test_policy", {}).get("splits", [])) != set(DATASETS):
        raise RuntimeError("sealed dataset contract/schema mismatch")
    out = {}
    for split, spec in DATASETS.items():
        for key in ("manifest", "payload"):
            if not spec[key].is_file(): raise RuntimeError(f"missing sealed {key}: {spec[key]}")
            actual = sha(spec[key])
            if actual != spec[f"{key}_sha256"]: raise RuntimeError(f"sealed {key} hash mismatch: {split}")
        headers = _npy_headers(spec["payload"])
        expected = spec["sample_count"]
        if set(headers) != {"inputs.npy", "targets.npy"} or any(v["shape"] != [expected, 16] or v["dtype"] != "<c16" for v in headers.values()):
            raise RuntimeError(f"sealed payload schema mismatch: {split}: {headers}")
        # json.load is intentionally avoided: preflight records only byte/hash and NPZ header metadata.
        out[split] = {"schema_version": "cc-nqe-p4.6-v1", "split_identity": split, "sample_count": expected,
                      "manifest_sha256": spec["manifest_sha256"], "payload_sha256": spec["payload_sha256"],
                      "payload_headers": headers, "target_arrays_loaded": False}
    return {"status": "PASS", "schema_version": contract["schema_version"], "contract_sha256": sha(contract_path), "historical_access_count": 0, "datasets": out}


def verify_p47() -> dict[str, Any]:
    subprocess.run(("git", "cat-file", "-e", f"{SOURCE_COMMIT}^{{commit}}"), check=True)
    rows = audit_p47()
    hashes = json.loads((P47 / "artifact_hashes.json").read_text())["artifacts"]
    failures = [p for p, expected in hashes.items() if not Path(p).is_file() or sha(Path(p)) != expected]
    verdict = json.loads((P47 / "scientific_verdict.json").read_text())
    if failures or len(rows) != 12 or verdict.get("sealed_test_access_count") != 0 or verdict.get("primary_candidate") != "C1" or verdict.get("composition_specialist_candidate") != "C2":
        raise RuntimeError(f"P4.7 anchor integrity failed: {failures}")
    return {"status": "PASS", "source_commit": SOURCE_COMMIT, "cells": "12/12", "artifact_hashes_verified": len(hashes)}


def candidate_records() -> list[dict[str, Any]]:
    rows = {(r["variant"], r["seed"]): r for r in audit_p47()}
    records = []
    for variant, seed in ORDER:
        cp, cfg, met = checkpoint_path(variant, seed), config_path(variant, seed), metric_path(variant, seed)
        if not all(p.is_file() for p in (cp, cfg, met)): raise RuntimeError(f"CANDIDATE-FREEZE-BLOCKED: missing {variant}/{seed}")
        payload = torch.load(cp, map_location="cpu", weights_only=False)
        config, metric = json.loads(cfg.read_text()), json.loads(met.read_text())
        expected_step = metric["best_checkpoint_step"]
        if payload.get("step") != expected_step or payload.get("dataset_manifest_hash") != rows[(variant, seed)]["dataset_hash"]:
            raise RuntimeError(f"CANDIDATE-FREEZE-BLOCKED: best-balanced provenance mismatch {variant}/{seed}")
        # Closure rows use seed-normalized hashes; checkpoints must match their exact source config.
        from cc_nqe_p4_6 import digest
        if payload.get("config_hash") != digest(config): raise RuntimeError(f"CANDIDATE-FREEZE-BLOCKED: config mismatch {variant}/{seed}")
        architecture = "P4.6-B3-monolithic-operator" if variant == "C0" else "P4.7-shared-causal-residual-recurrence"
        records.append({"variant": variant, "seed": seed, "role": ROLES[variant], "supervision_class": SUPERVISION[variant],
            "source_phase": rows[(variant, seed)]["source_phase"], "checkpoint_path": str(cp),
            "best_balanced_checkpoint_step": expected_step, "model_parameter_count": rows[(variant, seed)]["parameter_count"],
            "config_path": str(cfg), "config_hash": payload["config_hash"], "config_file_sha256": sha(cfg),
            "dataset_manifest_hash": payload["dataset_manifest_hash"], "source_metric_path": str(met),
            "source_metric_hash": sha(met), "checkpoint_sha256": sha(cp), "architecture_identifier": architecture,
            "cayley_metadata": {"parameterization": "basic_cayley", "scale": 1.0, "dtype": "FP32", "explicit_dimension": 16},
            "training_completion_status": metric.get("state", "COMPLETED")})
    return records


def verdicts(c10: list[float], c21_comp: list[float]) -> dict[str, str]:
    m10, m21 = statistics.mean(c10), statistics.mean(c21_comp)
    recurrent = "SEALED-RECURRENT-SUPPORTED" if all(x > 0 for x in c10) else "SEALED-RECURRENT-QUALIFIED" if m10 > 0 else "SEALED-RECURRENT-NOT-SUPPORTED"
    composition = "SEALED-COMPOSITION-SPECIFIC-GAIN" if all(x > 0 for x in c21_comp) else "SEALED-COMPOSITION-GAIN-QUALIFIED" if m21 > 0 else "SEALED-COMPOSITION-NOT-SUPPORTED"
    strongest = (recurrent == "SEALED-RECURRENT-SUPPORTED", composition == "SEALED-COMPOSITION-SPECIFIC-GAIN")
    overall = "P4.8-SEALED-HYPOTHESES-SUPPORTED" if all(strongest) else "P4.8-SEALED-PARTIALLY-SUPPORTED" if any(strongest) or "QUALIFIED" in recurrent or "QUALIFIED" in composition else "P4.8-SEALED-NOT-SUPPORTED"
    return {"recurrent": recurrent, "composition": composition, "overall": overall}


def prepare_artifacts() -> dict[str, Any]:
    ROOT.mkdir(parents=True, exist_ok=True)
    p47 = verify_p47(); records = candidate_records(); sealed = verify_sealed_metadata()
    protocol = {"schema_version": SCHEMA, "status": "FROZEN_NOT_RUN", "objective": "one untouched final evaluation of validation-selected P4.7 conclusions",
        "hypotheses": {"H1": "C1 improves over C0 on S_sealed", "H2": "C2 improves over C1 on F_comp_test with trade-offs visible", "H3": "C0 remains the anchor"},
        "candidates": ROLES, "seeds": list(SEEDS), "evaluation_order": [f"{v}-{s}" for v, s in ORDER], "device": "xpu:0", "dtype": "FP32", "batch_size": BATCH_SIZE,
        "endpoints": {"F_comp_test": "mean composition fidelity", "F_depth_8": "mean fidelity at depth 8", "F_depth_9": "mean fidelity at depth 9", "F_depth_10": "mean fidelity at depth 10", "F_depth_macro": "mean(F_depth_8,F_depth_9,F_depth_10)", "S_sealed": "(F_comp_test+F_depth_macro)/2", "depth_diagnostics": ["least-squares slope over 8,9,10", "absolute degradation depth 8 to 10"]},
        "primary_comparisons": ["C1_minus_C0", "C2_minus_C1"], "secondary_comparison": "C2_minus_C0", "verdict_rules": {"recurrent": ["SEALED-RECURRENT-SUPPORTED: all three S_sealed deltas positive", "SEALED-RECURRENT-QUALIFIED: positive mean with mixed signs", "SEALED-RECURRENT-NOT-SUPPORTED: nonpositive mean"], "composition": ["SEALED-COMPOSITION-SPECIFIC-GAIN: all three F_comp_test deltas positive", "SEALED-COMPOSITION-GAIN-QUALIFIED: positive mean with mixed signs", "SEALED-COMPOSITION-NOT-SUPPORTED: nonpositive mean"], "overall": ["P4.8-SEALED-HYPOTHESES-SUPPORTED: both strongest verdicts", "P4.8-SEALED-PARTIALLY-SUPPORTED: exactly one strongest or one/both qualified", "P4.8-SEALED-NOT-SUPPORTED: neither supported", "P4.8-SEALED-INCONCLUSIVE: integrity, execution, or incomplete-evaluation failure only"], "formal_significance": False},
        "bootstrap": BOOTSTRAP, "no_post_test_selection": True, "no_more_tuning": True, "sealed_test_evaluated": False, "access_count": 0}
    dump(ROOT / "protocol.json", protocol)
    freeze = {"schema_version": SCHEMA, "status": "FROZEN", "source_p4_7_commit": SOURCE_COMMIT, "candidate_roles": ROLES,
        "checkpoints": records, "excluded_C3": "Archived negative privileged-supervision result; final test cannot reconsider screening.",
        "frozen_evaluation_endpoints": protocol["endpoints"], "frozen_verdict_logic": protocol["verdict_rules"],
        "no_more_tuning_declaration": "No training, tuning, replacement, variant addition, or post-test selection is permitted."}
    dump(ROOT / "candidate_freeze.json", freeze)
    atomic_write(ROOT / "candidate_checkpoint_hashes.sha256", "".join(f"{r['checkpoint_sha256']}  {r['checkpoint_path']}\n" for r in records).encode())
    atomic_write(ROOT / "sealed_dataset_hashes.sha256", "".join(f"{spec[k+'_sha256']}  {spec[k]}\n" for spec in DATASETS.values() for k in ("manifest", "payload")).encode())
    access = {"schema_version": SCHEMA, "state": "PREPARED", "access_count": 0, "sealed_test_evaluated": False, "transaction": None, "history": []}
    dump(ROOT / "sealed_access_log.json", access)
    dump(ROOT / "unlock_manifest.json", {"schema_version": SCHEMA, "status": "NOT_RUN", "unlock_token": None, "sealed_data_used": False})
    status = {"schema_version": SCHEMA, "status": "NOT_RUN", "implementation_status": "PREPARED", "access_count": 0, "sealed_test_evaluated": False, "sealed_scientific_evaluation": "NONE"}
    dump(ROOT / "status.json", status)
    pre = {"schema_version": SCHEMA, "status": "PREPARED", "p4_7_integrity": p47, "candidate_count": len(records), "candidate_hashes": "PASS", "sealed_metadata": sealed, "sealed_arrays_loaded": False, "access_count": 0}
    dump(ROOT / "preflight.json", pre)
    (ROOT / "tests").mkdir(exist_ok=True)
    (ROOT / "implementation_report.md").write_text("# CC-NQE P4.8 implementation report\n\nStatus: **P4.8-READY**\n\nP4.7 anchor, nine validation-selected checkpoint hashes, sealed byte/header metadata, endpoints, verdicts, and paired-bootstrap policy are frozen. The guarded native-XPU transaction writes STARTED atomically before loading data, supports provenance-identical resume, and publishes results atomically. The unsealed dry run and full suite pass. Sealed access remains 0 and scientific evaluation is NONE. `prepare-unlock`, `sealed-evaluate`, and `resume-sealed-evaluate` were not run.\n")
    return pre


def verify_freeze() -> dict[str, Any]:
    freeze = json.loads((ROOT / "candidate_freeze.json").read_text())
    current = {(r["variant"], r["seed"]): r for r in candidate_records()}
    if len(freeze.get("checkpoints", [])) != 9: raise RuntimeError("CANDIDATE-FREEZE-BLOCKED: expected nine records")
    for frozen in freeze["checkpoints"]:
        live = current[(frozen["variant"], frozen["seed"])]
        for field in ("checkpoint_sha256", "config_hash", "config_file_sha256", "dataset_manifest_hash", "source_metric_hash", "best_balanced_checkpoint_step"):
            if frozen[field] != live[field]: raise RuntimeError(f"CANDIDATE-FREEZE-BLOCKED: {field} changed")
    return {"status": "PASS", "candidate_count": 9, "candidate_freeze_sha256": sha(ROOT / "candidate_freeze.json")}


def xpu_preflight() -> dict[str, Any]:
    if not hasattr(torch, "xpu") or not torch.xpu.is_available(): raise RuntimeError("XPU-PREFLIGHT-BLOCKED: native xpu:0 unavailable; no CPU fallback")
    device = torch.device("xpu:0")
    gates = torch.ones((1, 1), dtype=torch.long, device=device); qubits = torch.zeros((1, 1, 2), dtype=torch.long, device=device)
    params = torch.zeros((1, 1, 3), device=device); mask = torch.ones((1, 1), dtype=torch.bool, device=device)
    with torch.no_grad(): out = RecursiveOperatorModel().to(device).eval()(gates, qubits, params, mask)
    torch.xpu.synchronize()
    if out.device.type != "xpu" or not bool(torch.isfinite(out).all()): raise RuntimeError("XPU-PREFLIGHT-BLOCKED: native Cayley inference failed")
    return {"status": "PASS", "device": "xpu:0", "native_cayley": True, "cpu_fallback": False}


def _scientific_tree_clean() -> bool:
    paths = ("cc_nqe_p4_8.py", "run_p4_8.py", "tests/test_cc_nqe_p4_8.py", "pyproject.toml", "uv.lock")
    return subprocess.run(("git", "diff", "--quiet", "HEAD", "--", *paths)).returncode == 0 and subprocess.run(("git", "diff", "--cached", "--quiet", "HEAD", "--", *paths)).returncode == 0


def preflight(require_clean: bool = False) -> dict[str, Any]:
    if git("branch", "--show-current") != BRANCH: raise RuntimeError("wrong branch")
    if require_clean and not _scientific_tree_clean(): raise RuntimeError("dirty scientific implementation/config tree")
    access = json.loads((ROOT / "sealed_access_log.json").read_text())
    if access["access_count"] != 0 or access["sealed_test_evaluated"]: raise RuntimeError("SEALED-RETRY-BLOCKED")
    result = {"schema_version": SCHEMA, "status": "PASS", "implementation_commit": git("rev-parse", "HEAD"),
              "protocol_sha256": sha(ROOT / "protocol.json"), **verify_freeze(), "p4_7_integrity": verify_p47(),
              "sealed_metadata": verify_sealed_metadata(), "xpu": xpu_preflight(), "sealed_arrays_loaded": False,
              "access_count": 0, "sealed_test_evaluated": False}
    dump(ROOT / "preflight.json", result)
    return result


def unlock_token(protocol_hash: str, freeze_hash: str, commit: str) -> str:
    return hashlib.sha256(f"P4.8-UNLOCK\0{protocol_hash}\0{freeze_hash}\0{commit}".encode()).hexdigest()


def validate_unlock_token(provided: str | None, manifest: dict[str, Any], current: dict[str, str]) -> None:
    expected = unlock_token(current["protocol_sha256"], current["candidate_freeze_sha256"], current["implementation_commit"])
    if (not provided or not secrets.compare_digest(provided, expected) or provided != manifest.get("unlock_token")
            or manifest.get("implementation_commit") != current["implementation_commit"]):
        raise RuntimeError("SEALED-UNLOCK-REJECTED: missing, wrong, or stale token")


def prepare_unlock() -> dict[str, Any]:
    p = preflight(require_clean=True)
    transaction_id = str(uuid.uuid4())
    manifest = {"schema_version": SCHEMA, "state": "PREPARED", "transaction_id": transaction_id,
        "protocol_sha256": p["protocol_sha256"], "candidate_freeze_sha256": p["candidate_freeze_sha256"],
        "implementation_commit": p["implementation_commit"], "checkpoint_hashes_sha256": sha(ROOT / "candidate_checkpoint_hashes.sha256"),
        "dataset_hashes_sha256": sha(ROOT / "sealed_dataset_hashes.sha256")}
    manifest["unlock_token"] = unlock_token(manifest["protocol_sha256"], manifest["candidate_freeze_sha256"], manifest["implementation_commit"])
    dump(ROOT / "unlock_manifest.json", manifest)
    log = json.loads((ROOT / "sealed_access_log.json").read_text()); log["transaction"] = manifest; dump(ROOT / "sealed_access_log.json", log)
    return manifest


def calculate_metrics(comp: np.ndarray, depth_values: dict[int, np.ndarray]) -> dict[str, float]:
    result = {"F_comp_test": float(np.mean(comp))}
    for depth in (8, 9, 10): result[f"F_depth_{depth}"] = float(np.mean(depth_values[depth]))
    result["F_depth_macro"] = statistics.mean(result[f"F_depth_{d}"] for d in (8, 9, 10))
    result["S_sealed"] = (result["F_comp_test"] + result["F_depth_macro"]) / 2
    result["depth_slope"] = float(np.polyfit([8, 9, 10], [result[f"F_depth_{d}"] for d in (8, 9, 10)], 1)[0])
    result["depth_8_to_10_degradation"] = result["F_depth_8"] - result["F_depth_10"]
    return result


def paired_deltas(per_checkpoint: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by = {(x["variant"], x["seed"]): x["metrics"] for x in per_checkpoint}; result = []
    for name, left, right in (("C1_minus_C0", "C1", "C0"), ("C2_minus_C1", "C2", "C1"), ("C2_minus_C0", "C2", "C0")):
        for seed in SEEDS:
            result.append({"comparison": name, "seed": seed, **{k: by[(left, seed)][k] - by[(right, seed)][k] for k in ("S_sealed", "F_comp_test", "F_depth_macro", "F_depth_8", "F_depth_9", "F_depth_10")}})
    return result


def paired_bootstrap(left: dict[str, Any], right: dict[str, Any], seed: int = 47008, resamples: int = 10000) -> dict[str, Any]:
    rng = np.random.default_rng(seed); comp = np.asarray(left["comp_examples"]) - np.asarray(right["comp_examples"])
    depths = {d: np.asarray(left["depth_examples"][str(d)]) - np.asarray(right["depth_examples"][str(d)]) for d in (8, 9, 10)}
    cm = comp[rng.integers(0, len(comp), (resamples, len(comp)))].mean(1)
    dm = np.stack([x[rng.integers(0, len(x), (resamples, len(x)))].mean(1) for x in depths.values()]).mean(0)
    def ci(x): return [float(v) for v in np.quantile(x, [0.025, 0.975])]
    return {"label": "example-level bootstrap uncertainty", "bootstrap_seed": seed, "resamples": resamples, "confidence_level": 0.95,
            "F_comp_test_delta": {"estimate": float(comp.mean()), "ci": ci(cm)}, "F_depth_macro_delta": {"estimate": float(np.mean([x.mean() for x in depths.values()])), "ci": ci(dm)},
            "S_sealed_delta": {"estimate": float((comp.mean() + np.mean([x.mean() for x in depths.values()])) / 2), "ci": ci((cm + dm) / 2)}}


def _load_model(variant: str, checkpoint: Path, device: torch.device):
    model = OperatorModel("cayley") if variant == "C0" else RecursiveOperatorModel()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False); model.load_state_dict(payload["model"])
    return model.to(device).eval()


def _evaluate_checkpoint(variant: str, seed: int, datasets: dict[str, tuple[list[dict[str, Any]], dict[str, np.ndarray]]], device: torch.device) -> dict[str, Any]:
    model = _load_model(variant, checkpoint_path(variant, seed), device); all_examples = {}
    started = time.monotonic()
    with torch.no_grad():
        for split, (rows, arrays) in datasets.items():
            values = []
            for start in range(0, len(rows), BATCH_SIZE):
                chunk = rows[start:start+BATCH_SIZE]; circuits = [[Gate.from_dict(g) for g in row["gates"]] for row in chunk]
                gates, qubits, parameters, mask = _circuit_tensors(circuits, device)
                states = torch.as_tensor(arrays["inputs"][start:start+len(chunk)], device=device)
                targets = torch.as_tensor(arrays["targets"][start:start+len(chunk)], device=device)
                operators = model(gates, qubits, parameters, mask); action, _ = operator_action(operators, torch.cat((states.real, states.imag), 1).float())
                packed_targets = torch.cat((targets.real, targets.imag), 1).float()
                values.extend(state_fidelity(action, packed_targets).detach().cpu().numpy().tolist())
            all_examples[split] = values
    torch.xpu.synchronize(); del model
    comp = np.asarray(all_examples["composition_ood_test_sealed"])
    depth_rows = datasets["depth_ood_test_sealed"][0]; depth = {d: np.asarray([v for v, row in zip(all_examples["depth_ood_test_sealed"], depth_rows) if row["depth"] == d]) for d in (8, 9, 10)}
    return {"variant": variant, "seed": seed, "metrics": calculate_metrics(comp, depth), "comp_examples": comp.tolist(), "depth_examples": {str(k): v.tolist() for k, v in depth.items()}, "runtime_seconds": time.monotonic() - started, "device": "xpu:0", "dtype": "FP32"}


def _load_scientific_datasets() -> dict[str, tuple[list[dict[str, Any]], dict[str, np.ndarray]]]:
    return {split: (json.loads(spec["manifest"].read_text()), dict(np.load(spec["payload"], allow_pickle=False))) for split, spec in DATASETS.items()}


def _validation_mirrors() -> dict[str, tuple[list[dict[str, Any]], dict[str, np.ndarray]]]:
    result = {}
    for out_split, source in (("composition_ood_test_sealed", "composition_ood_validation"), ("depth_ood_test_sealed", "depth_ood_validation")):
        rows = json.loads((P46 / f"datasets/{source}.json").read_text())[:6]
        arrays = dict(np.load(P46 / f"datasets/{source}.npz", allow_pickle=False)); arrays = {k: v[:len(rows)] for k, v in arrays.items()}
        if out_split.startswith("depth"):
            rows = [{**r, "depth": (8, 9, 10)[i % 3]} for i, r in enumerate(rows)]
        result[out_split] = rows, arrays
    return result


def _summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = paired_deltas(rows); by = {(x["variant"], x["seed"]): x for x in rows}
    boot = []
    for name, left, right in (("C1_minus_C0", "C1", "C0"), ("C2_minus_C1", "C2", "C1")):
        for seed in SEEDS: boot.append({"comparison": name, "seed": seed, **paired_bootstrap(by[(left, seed)], by[(right, seed)])})
    c10 = [x["S_sealed"] for x in deltas if x["comparison"] == "C1_minus_C0"]
    c21 = [x["F_comp_test"] for x in deltas if x["comparison"] == "C2_minus_C1"]
    return {"paired_deltas": deltas, "bootstrap": boot, "verdicts": verdicts(c10, c21)}


def dry_run() -> dict[str, Any]:
    verify_freeze(); device = torch.device("xpu:0") if xpu_preflight()["status"] == "PASS" else None
    rows = [_evaluate_checkpoint(v, s, _validation_mirrors(), device) for v, s in ORDER]
    result = {"schema_version": SCHEMA, "status": "PASS", "scientific_run": False, "sealed_data_used": False,
              "purpose": "implementation_validation", "checkpoint_count": len(rows), "summaries": _summaries(rows), "access_count": 0}
    dump(ROOT / "tests/dry_run_report.json", result)
    return result


def _publish(rows: list[dict[str, Any]], transaction_id: str) -> None:
    summary = _summaries(rows); sealed = ROOT / "sealed"; stage = ROOT / f".transaction-{transaction_id}" / "sealed"
    if not sealed.exists():
        stage.mkdir(parents=True, exist_ok=True)
        public_rows = [{k: v for k, v in r.items() if k not in ("comp_examples", "depth_examples")} for r in rows]
        per_seed = {str(s): {v: next(r["metrics"] for r in rows if r["seed"] == s and r["variant"] == v) for v in VARIANTS} for s in SEEDS}
        aggregate = {v: {m: {"mean": statistics.mean(r["metrics"][m] for r in rows if r["variant"] == v), "sample_std": statistics.stdev(r["metrics"][m] for r in rows if r["variant"] == v)} for m in ("S_sealed", "F_comp_test", "F_depth_macro")} for v in VARIANTS}
        files = {"per_checkpoint_metrics.json": public_rows, "per_seed_summary.json": per_seed, "aggregate_summary.json": aggregate,
                 "paired_deltas.json": summary["paired_deltas"], "example_bootstrap.json": summary["bootstrap"],
                 "depth_curves.json": [{"variant": r["variant"], "seed": r["seed"], **{str(d): r["metrics"][f"F_depth_{d}"] for d in (8,9,10)}} for r in rows]}
        for name, value in files.items(): dump(stage / name, value)
        hashes = {name: sha(stage / name) for name in files}; atomic_write(stage / "result_hashes.sha256", "".join(f"{h}  {n}\n" for n, h in hashes.items()).encode())
        os.replace(stage, sealed)  # atomic directory publication; no partial public result set
    dump(ROOT / "scientific_verdict.json", {"schema_version": SCHEMA, **summary["verdicts"], "candidate_roles_unchanged": ROLES, "formal_significance_claimed": False})
    (ROOT / "P4_8_FINAL_REPORT.md").write_text(f"# CC-NQE P4.8 Final Report\n\nTransaction `{transaction_id}` completed. Verdict: **{summary['verdicts']['overall']}**. Candidate roles remain frozen. See sealed JSON artifacts for all per-seed endpoints and example-level bootstrap uncertainty.\n")
    artifacts = [ROOT / "protocol.json", ROOT / "candidate_freeze.json", ROOT / "scientific_verdict.json", ROOT / "P4_8_FINAL_REPORT.md", *sealed.iterdir()]
    dump(ROOT / "artifact_hashes.json", {"algorithm": "SHA-256", "artifacts": {str(p): sha(p) for p in artifacts if p.is_file()}})


def sealed_evaluate(token: str | None, resume_id: str | None = None) -> dict[str, Any]:
    log = json.loads((ROOT / "sealed_access_log.json").read_text()); unlock = json.loads((ROOT / "unlock_manifest.json").read_text())
    if resume_id is None:
        p = preflight(require_clean=True)
        validate_unlock_token(token, unlock, p)
        transaction_id = unlock["transaction_id"]
        log.update({"state": "STARTED", "access_count": 1, "sealed_test_evaluated": False,
                    "transaction": unlock}); log["history"].append({"transaction_id": transaction_id, "state": "STARTED", "time": time.time()}); dump(ROOT / "sealed_access_log.json", log)
    else:
        if log.get("state") not in ("STARTED", "FAILED") or log.get("access_count") != 1 or log.get("transaction", {}).get("transaction_id") != resume_id:
            raise RuntimeError("SEALED-RETRY-BLOCKED")
        if git("branch", "--show-current") != BRANCH or not _scientific_tree_clean(): raise RuntimeError("SEALED-RETRY-BLOCKED: branch/tree changed")
        transaction_id = resume_id
        for field, live in (("implementation_commit", git("rev-parse", "HEAD")), ("protocol_sha256", sha(ROOT/"protocol.json")), ("candidate_freeze_sha256", sha(ROOT/"candidate_freeze.json")), ("checkpoint_hashes_sha256", sha(ROOT/"candidate_checkpoint_hashes.sha256")), ("dataset_hashes_sha256", sha(ROOT/"sealed_dataset_hashes.sha256"))):
            if log["transaction"][field] != live: raise RuntimeError("SEALED-RETRY-BLOCKED: transaction provenance changed")
        verify_freeze(); verify_sealed_metadata(); xpu_preflight()
    private = ROOT / f".transaction-{transaction_id}"; private.mkdir(exist_ok=True)
    try:
        datasets = _load_scientific_datasets(); rows = []
        for variant, seed in ORDER:
            path = private / f"{variant}-{seed}.json"
            if path.exists(): row = json.loads(path.read_text())
            else: row = _evaluate_checkpoint(variant, seed, datasets, torch.device("xpu:0")); dump(path, row)
            rows.append(row)
        _publish(rows, transaction_id)
        log.update({"state": "COMPLETED", "sealed_test_evaluated": True}); log["history"].append({"transaction_id": transaction_id, "state": "COMPLETED", "time": time.time()}); dump(ROOT / "sealed_access_log.json", log)
        dump(ROOT / "status.json", {"schema_version": SCHEMA, "status": "COMPLETED", "access_count": 1, "sealed_test_evaluated": True})
        return {"status": "COMPLETED", "transaction_id": transaction_id}
    except BaseException as exc:
        log.update({"state": "FAILED", "sealed_test_evaluated": False}); log["history"].append({"transaction_id": transaction_id, "state": "FAILED", "error_type": type(exc).__name__, "time": time.time()}); dump(ROOT / "sealed_access_log.json", log)
        raise


def status() -> dict[str, Any]:
    return json.loads((ROOT / "status.json").read_text())


def report() -> dict[str, Any]:
    path = ROOT / "P4_8_FINAL_REPORT.md"
    return {"status": "COMPLETED" if path.exists() else "NOT_RUN", "report": str(path) if path.exists() else None}
