"""Authorized R1 Gate 4 pilot-viability evaluation runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from cc_nqe import DIM, Gate, circuit_unitary, generate_state
from cc_nqe_p4_5 import state_fidelity
from cc_nqe_p4_6 import OperatorModel
from cc_nqe_p4_6_track_a import _tensorize_circuit

ROOT = Path("artifacts/r1_operator_semantic_benchmark")
GATE4 = ROOT / "gate4"
PILOT = ROOT / "scientific_pilot_v1"
AUTHORIZATION = ROOT / "authorization/pilot_viability_evaluation_authorization.json"
SEEDS = (2026, 2027, 2028)
BOOTSTRAP_SEED = 47011
BOOTSTRAP_REPLICATES = 10_000
FEATURE_BINS = 152
FEATURE_DIM = 305
BATCH_SIZE = 1024
EFFECTIVE_EXPOSURES = 10.24


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def runtime_environment(device: str) -> dict[str, Any]:
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "backend_device": device,
        "training_dtype": "float32",
        "operator_dtype": "complex64",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def verify_authorization(device: str = "cpu") -> dict[str, Any]:
    auth = read_json(AUTHORIZATION)
    plan = read_json(GATE4 / "evaluation_plan_manifest.json")
    manifest = read_json(PILOT / "corpus_manifest.json")
    if auth.get("authorization_scope") != "PILOT_VIABILITY_EVALUATION_ONLY" or auth.get("status") != "AUTHORIZED" or auth.get("consumed"):
        raise PermissionError("R1-GATE4-EVALUATION-AUTHORIZATION-BLOCKED")
    bindings = {
        "evaluation_plan_sha256": plan["evaluation_plan_sha256"],
        "evaluation_protocol_sha256": sha256_file(GATE4 / "evaluation_protocol.json"),
        "pilot_corpus_manifest_sha256": sha256_file(PILOT / "corpus_manifest.json"),
        "pilot_corpus_checksum_ledger_sha256": sha256_file(PILOT / "checksums.sha256"),
        "pilot_generated_corpus_sha256": manifest["generated_corpus_sha256"],
        "fold_manifest_sha256": sha256_file(GATE4 / "fold_manifest.json"),
        "metric_contract_sha256": sha256_file(GATE4 / "metric_contract.json"),
        "baseline_contract_sha256": sha256_file(GATE4 / "baseline_contract.json"),
        "runner_source_sha256": sha256_file(Path(__file__)),
        "dependency_lock_sha256": sha256_file(GATE4 / "dependency_lock.json"),
        "execution_environment_fingerprint": digest(runtime_environment(device)),
        "authorized_device": device,
    }
    if any(auth.get(key) != value for key, value in bindings.items()):
        raise PermissionError("R1-GATE4-EVALUATION-AUTHORIZATION-BLOCKED")
    return auth


def _gate_rows(circuit: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    return circuit["gates"] if isinstance(circuit, dict) else circuit


def _tokens(circuit: dict[str, Any] | list[dict[str, Any]]) -> list[str]:
    return [f"{gate.get('gate', gate.get('name'))}:{','.join(map(str, gate['qubits']))}" for gate in _gate_rows(circuit)]


def _hashed_syntax(circuit: dict[str, Any] | list[dict[str, Any]]) -> np.ndarray:
    out = np.zeros(FEATURE_BINS, np.float64)
    tokens = _tokens(circuit)
    features = [f"depth={len(tokens)}", f"twoq={sum(x.startswith('CNOT:') for x in tokens)}"]
    for width in (1, 2, 3):
        features.extend(f"{width}:" + "|".join(tokens[i:i + width]) for i in range(len(tokens) - width + 1))
    for feature in features:
        raw = hashlib.sha256(b"qute-r1-o1-syntax-v1\0" + feature.encode()).digest()
        out[int.from_bytes(raw[:8], "big") % FEATURE_BINS] += 1 if raw[8] & 1 else -1
    return out


def syntax_pair_vector(left: dict[str, Any] | list[dict[str, Any]], right: dict[str, Any] | list[dict[str, Any]]) -> np.ndarray:
    vector = np.r_[_hashed_syntax(left), _hashed_syntax(right), abs(len(left) - len(right))]
    assert vector.shape == (FEATURE_DIM,)
    return vector


def _pair_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    circuits = {row["circuit_id"]: row["canonical_circuit"] for row in read_jsonl(PILOT / "circuits.jsonl")}
    pairs = read_jsonl(PILOT / "pairs.jsonl")
    folds = {row["equivalence_class_id"]: row["fold"] for row in read_json(GATE4 / "fold_manifest.json")["assignments"]}
    x = np.stack([syntax_pair_vector(circuits[row["left_circuit_id"]], circuits[row["right_circuit_id"]]) for row in pairs])
    y = np.asarray([row["label"] == "POSITIVE_EQUIVALENT" for row in pairs], np.int8)
    group_fold = np.asarray([folds[row["equivalence_class_id"]] for row in pairs], np.int8)
    return x, y, group_fold, pairs


def _class_bootstrap_auc(y: np.ndarray, score: np.ndarray, pairs: list[dict[str, Any]]) -> dict[str, float]:
    by_class: dict[str, list[int]] = {}
    for index, row in enumerate(pairs):
        by_class.setdefault(row["equivalence_class_id"], []).append(index)
    classes = sorted(by_class)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.empty(BOOTSTRAP_REPLICATES)
    for i in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(classes, len(classes), replace=True)
        indices = np.asarray([index for group in sampled for index in by_class[group]])
        values[i] = roc_auc_score(y[indices], score[indices])
    return {"lower": float(np.quantile(values, .025)), "median": float(np.median(values)), "upper": float(np.quantile(values, .975))}


def run_o1(output: Path) -> dict[str, Any]:
    x, y, folds, pairs = _pair_dataset()
    results = {}
    for name in ("O1-LR", "O1-RF"):
        scores = np.empty(len(y)); predictions = np.empty(len(y), np.int8)
        for fold in range(5):
            train, test = folds != fold, folds == fold
            if name == "O1-LR":
                scaler = StandardScaler().fit(x[train]); model = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, random_state=47011)
                model.fit(scaler.transform(x[train]), y[train]); scores[test] = model.predict_proba(scaler.transform(x[test]))[:, 1]
            else:
                model = RandomForestClassifier(n_estimators=500, random_state=47011, n_jobs=1)
                model.fit(x[train], y[train]); scores[test] = model.predict_proba(x[test])[:, 1]
            predictions[test] = scores[test] >= .5
        ci = _class_bootstrap_auc(y, scores, pairs)
        results[name] = {"auroc": float(roc_auc_score(y, scores)), "balanced_accuracy": float(balanced_accuracy_score(y, predictions)), "class_bootstrap_95pct": ci}
    passed = all(row["class_bootstrap_95pct"]["upper"] < .70 and row["balanced_accuracy"] < .65 for row in results.values())
    report = {"schema_version": "qute-r1-gate4-o1-v1", "status": "PASS" if passed else "FAIL", "verdict": "O1-SYNTAX-SHORTCUT-PASS" if passed else "SYNTAX-SHORTCUT-BLOCKED", "feature_dimension": FEATURE_DIM, "models": results}
    output.mkdir(parents=True, exist_ok=True); (output / "o1_syntax_shortcut.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def _decode(row: dict[str, Any]) -> list[Gate]:
    return [
        Gate(
            gate.get("gate", gate.get("name")),
            tuple(gate["qubits"]),
            float.fromhex(gate["parameters"][0]) if gate.get("parameters") else None,
        )
        for gate in _gate_rows(row["canonical_circuit"])
    ]


def _training_probe(circuit_id: str, index: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(b"qute-r1-gate4-train-probe-v1\0" + circuit_id.encode() + index.to_bytes(2, "big")).digest()[:8], "big") % 2**32
    return generate_state(seed, "product")


def _basis_probes() -> np.ndarray:
    return np.eye(DIM, dtype=np.complex128)


def _tensors(circuits: list[list[Gate]], device: torch.device) -> list[torch.Tensor]:
    values = list(zip(*[_tensorize_circuit(circuit) for circuit in circuits]))
    return [torch.as_tensor(np.asarray(value)).to(device) for value in values]


def _train_fold(train_rows: list[dict[str, Any]], seed: int, device: torch.device) -> OperatorModel:
    torch.manual_seed(seed); np_rng = np.random.default_rng(seed)
    circuits = [_decode(row) for row in train_rows]
    tensors = _tensors(circuits, device)
    model = OperatorModel("cayley").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    sample_count = len(circuits) * 17
    updates = math.ceil(EFFECTIVE_EXPOSURES * sample_count / BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, updates)
    for _ in range(updates):
        indices = np_rng.integers(sample_count, size=BATCH_SIZE); ci, pi = np.divmod(indices, 17)
        states = np.stack([_training_probe(train_rows[int(c)]["circuit_id"], int(p)) for c, p in zip(ci, pi)])
        target = np.stack([circuit_unitary(circuits[int(c)]) @ state for c, state in zip(ci, states)])
        state_ri = np.c_[states.real, states.imag].astype(np.float32); target_ri = np.c_[target.real, target.imag].astype(np.float32)
        optimizer.zero_grad(set_to_none=True); predicted_u = model(*(value[ci] for value in tensors))
        state_complex = torch.as_tensor(states.astype(np.complex64)).to(device); predicted = torch.einsum("bij,bj->bi", predicted_u, state_complex)
        predicted_ri = torch.cat((predicted.real, predicted.imag), 1); target_tensor = torch.as_tensor(target_ri).to(device)
        loss = (1 - state_fidelity(predicted_ri, target_tensor)).mean(); loss.backward(); optimizer.step(); scheduler.step()
    return model.eval()


def run_semantic_baseline(output: Path, device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)
    rows = read_jsonl(PILOT / "circuits.jsonl"); folds = {row["equivalence_class_id"]: row["fold"] for row in read_json(GATE4 / "fold_manifest.json")["assignments"]}
    scored = []
    for seed in SEEDS:
        for fold in range(5):
            train = [row for row in rows if folds[row["equivalence_class_id"]] != fold]
            test = [row for row in rows if folds[row["equivalence_class_id"]] == fold]
            model = _train_fold(train, seed, device); circuits = [_decode(row) for row in test]
            with torch.inference_mode(): predicted = model(*_tensors(circuits, device)).cpu().numpy()
            for row, circuit, operator in zip(test, circuits, predicted):
                exact = circuit_unitary(circuit); fidelity = abs(np.vdot(exact, operator)) ** 2 / DIM**2
                scored.append({"seed": seed, "fold": fold, "circuit_id": row["circuit_id"], "equivalence_class_id": row["equivalence_class_id"], "circuit_role": row["circuit_role"], "absolute_action_fidelity": float(fidelity), "predicted_operator_real": operator.real.tolist(), "predicted_operator_imag": operator.imag.tolist()})
            del model
    output.mkdir(parents=True, exist_ok=True)
    with (output / "semantic_scores.jsonl").open("w") as handle:
        for row in scored: handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    report = {"schema_version": "qute-r1-gate4-semantic-baseline-v1", "status": "COMPLETED", "seeds": list(SEEDS), "folds": 5, "effective_exposures": EFFECTIVE_EXPOSURES, "checkpoint_policy": "FINAL_ONLY", "score_rows": len(scored)}
    (output / "semantic_baseline.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def _process_fidelity(left: np.ndarray, right: np.ndarray) -> float:
    return float(abs(np.vdot(left, right)) ** 2 / DIM**2)


def run_semantic_scoring(output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scores = read_jsonl(output / "semantic_scores.jsonl")
    by_key = {(row["seed"], row["equivalence_class_id"], row["circuit_role"]): row for row in scores}
    classes = read_jsonl(PILOT / "equivalence_classes.jsonl")
    ledger = read_json(ROOT / "preflight/pilot/pilot_coordinate_ledger.json")
    coverage_by_coordinate = {
        row["coordinate_id"]: row["coverage_cell_id"] for row in ledger["coordinates"]
    }
    rows = []
    for seed in SEEDS:
        for item in classes:
            eid = item["equivalence_class_id"]
            values = {role: by_key[(seed, eid, role)] for role in ("source", "target", "matched_control")}
            operators = {role: np.asarray(row["predicted_operator_real"]) + 1j * np.asarray(row["predicted_operator_imag"]) for role, row in values.items()}
            positive = _process_fidelity(operators["source"], operators["target"])
            negative = _process_fidelity(operators["source"], operators["matched_control"])
            rows.append({"seed": seed, "equivalence_class_id": eid, "coverage_cell_id": coverage_by_coordinate[item["coordinate_id"]], "fold": values["source"]["fold"], "absolute_action_fidelity": float(np.mean([row["absolute_action_fidelity"] for row in values.values()])), "positive_operator_consistency": positive, "negative_operator_consistency": negative, "margin": positive - negative, "ordering_correct": positive > negative})
    seed_reports = {}
    for seed in SEEDS:
        selected = [row for row in rows if row["seed"] == seed]
        positive = np.asarray([row["positive_operator_consistency"] for row in selected]); negative = np.asarray([row["negative_operator_consistency"] for row in selected])
        labels = np.r_[np.ones(len(selected)), np.zeros(len(selected))]; values = np.r_[positive, negative]
        predictions = values >= .5
        seed_reports[str(seed)] = {"absolute_action_fidelity": float(np.mean([row["absolute_action_fidelity"] for row in selected])), "pair_auroc": float(roc_auc_score(labels, values)), "fixed_0_5_balanced_accuracy": float(balanced_accuracy_score(labels, predictions)), "pair_ordering_accuracy": float(np.mean([row["ordering_correct"] for row in selected])), "margin_mean": float(np.mean([row["margin"] for row in selected])), "score_variance": float(np.var(values)), "prediction_class_count": int(len(np.unique(predictions)))}
    collapse_pass = all(row["prediction_class_count"] == 2 and row["score_variance"] > 0 for row in seed_reports.values())
    learnability_pass = collapse_pass and all(row["fixed_0_5_balanced_accuracy"] > .55 for row in seed_reports.values())
    report = {"schema_version": "qute-r1-gate4-semantic-scoring-v1", "status": "PASS" if learnability_pass else "FAIL", "verdict": "SEMANTIC-LEARNABILITY-PASS" if learnability_pass else ("REPRESENTATION-COLLAPSE" if not collapse_pass else "LEARNABILITY-BLOCKED"), "threshold_policy": "FIXED_0.5_REPORTING_ONLY_NO_TUNING", "seeds": seed_reports}
    (output / "semantic_scoring.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report, rows


def run_uncertainty_and_power(output: Path, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED); uncertainty = {}
    for seed in SEEDS:
        selected = [row for row in rows if row["seed"] == seed]
        strata: dict[str, list[dict[str, Any]]] = {}
        for row in selected: strata.setdefault(row["coverage_cell_id"], []).append(row)
        boot = np.empty(BOOTSTRAP_REPLICATES)
        for index in range(BOOTSTRAP_REPLICATES):
            sampled = [row for values in strata.values() for row in rng.choice(values, len(values), replace=True)]
            boot[index] = np.mean([row["margin"] for row in sampled])
        uncertainty[str(seed)] = {"margin_mean": float(np.mean([row["margin"] for row in selected])), "class_level_sd": float(np.std([row["margin"] for row in selected], ddof=1)), "stratified_cluster_bootstrap_95pct": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}
    means = np.asarray([uncertainty[str(seed)]["margin_mean"] for seed in SEEDS])
    uncertainty_report = {"schema_version": "qute-r1-gate4-uncertainty-v1", "status": "PASS" if np.isfinite(means).all() else "FAIL", "cluster": "equivalence_class_id", "stratum": "coverage_cell_id", "resamples": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "per_seed": uncertainty, "cross_seed_mean": float(means.mean()), "cross_seed_sd": float(means.std(ddof=1)), "fold_se_inference": False}
    (output / "uncertainty.json").write_text(json.dumps(uncertainty_report, sort_keys=True, indent=2) + "\n")
    sigma = max(row["class_level_sd"] for row in uncertainty.values()); z_alpha = 1.959963984540054
    precision_n = math.ceil((z_alpha * sigma / .015) ** 2)
    simulations = np.random.default_rng(47011); scenarios = []
    for effect in (.02, .03, .05):
        for rho in (0., .5, .9):
            se = sigma * math.sqrt(max(1e-12, 2 * (1 - rho)) / 2304)
            estimates = simulations.normal(effect, se, 10_000)
            power = float(np.mean(estimates / se > z_alpha))
            scenarios.append({"effect": effect, "rho": rho, "planned_development_n": 2304, "monte_carlo_replicates": 10_000, "power": power})
    anchor = next(row for row in scenarios if row["effect"] == .03 and row["rho"] == .5)
    feasible = precision_n <= 2304 and anchor["power"] >= .80
    power_report = {"schema_version": "qute-r1-gate4-power-v1", "status": "PASS" if feasible else "FAIL", "verdict": "SAMPLE-SIZE-FEASIBLE" if feasible else "VARIANCE-UNDERPOWERED", "mpme_anchor": .03, "sensitivity_effects": [.02, .05], "precision_half_width_target": .015, "precision_required_n": precision_n, "alpha": .05, "power_target": .80, "pilot_effect_used_for_future_n": False, "scenarios": scenarios}
    (output / "sample_size_planning.json").write_text(json.dumps(power_report, sort_keys=True, indent=2) + "\n")
    return uncertainty_report, power_report


def consume_authorization(auth: dict[str, Any], output: Path, verdict: str) -> None:
    files = sorted(path for path in output.iterdir() if path.is_file())
    result_hash = digest({path.name: sha256_file(path) for path in files})
    auth |= {"status": "CONSUMED", "consumed": True, "result_bundle_sha256": result_hash, "final_verdict": verdict}
    AUTHORIZATION.write_text(json.dumps(auth, sort_keys=True, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=GATE4 / "results"); parser.add_argument("--device", default="cpu"); parser.add_argument("--resume-stage-c", action="store_true"); args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    auth = verify_authorization(args.device)
    if args.resume_stage_c:
        frozen = {
            "o1_result_sha256": args.output / "o1_syntax_shortcut.json",
            "semantic_baseline_sha256": args.output / "semantic_baseline.json",
            "semantic_scores_sha256": args.output / "semantic_scores.jsonl",
        }
        if any(not path.is_file() or auth.get(key) != sha256_file(path) for key, path in frozen.items()):
            raise PermissionError("R1-GATE4-FROZEN-STAGE-B-EVIDENCE-MISMATCH")
    else:
        o1 = run_o1(args.output)
        if o1["status"] != "PASS": consume_authorization(auth, args.output, o1["verdict"]); print(o1["verdict"]); return
        run_semantic_baseline(args.output, args.device)
    semantic, rows = run_semantic_scoring(args.output)
    if semantic["status"] != "PASS": consume_authorization(auth, args.output, semantic["verdict"]); print(semantic["verdict"]); return
    uncertainty, power = run_uncertainty_and_power(args.output, rows)
    verdict = "R1-PILOT-VIABLE" if uncertainty["status"] == power["status"] == "PASS" else "VARIANCE-UNDERPOWERED"
    consume_authorization(auth, args.output, verdict); print(verdict)


if __name__ == "__main__":
    main()
