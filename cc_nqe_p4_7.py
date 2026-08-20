"""CC-NQE P4.7 compositional exact-unitary operator study (four qubits only)."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from cc_nqe import DIM, GATES, N_QUBITS, Gate, apply_gate, circuit_unitary
from cc_nqe_p4_5 import atomic_json, parameter_count, state_fidelity
from cc_nqe_p4_6 import SEED, digest, process_fidelity, raw_unitarity_error
from cc_nqe_p4_6_track_a import ArmData, DATA_ROOT, VALIDATION_SPLITS, _decode, load_validation
from cc_nqe_p4_6_track_b import (
    ALLOCATION, RECIPE, _circuit_tensors, _sha, action_metrics,
    operator_action, phase_aligned_normalized_frobenius_error,
)

ROOT = Path("artifacts/cc_nqe_p4_7")
SCHEMA = "cc-nqe-p4.7-v1"
ANCHOR_COMMIT = "f97990728194e5bedcebccc0294c89dd1bfb5b98"
ANCHOR_FILES = {
    "config": (Path("artifacts/cc_nqe_p4_6/operator/configs/B3.json"), "1880d510b97bbb027bae30418de7882b23b905fb6e2f196434b75d541d42f49e"),
    "dataset_manifest": (DATA_ROOT / "A4/manifest.json", "67a25b4384825dc477b22cbfc6f13bfc1424a95474cc8bcbc002c4ff010bf009"),
    "metric": (Path("artifacts/cc_nqe_p4_6/operator/metrics/B3.json"), "b77e5695205ec2fed3f7bfec69174cde8ba5a9b43c9a87efc82147c603d10352"),
}
WIDTH = 438
VARIANTS = {
    "C1": {"supervision_class": "ACTION_ONLY", "lambda_comp": 0.0, "lambda_prefix": 0.0},
    "C2": {"supervision_class": "ACTION_ONLY_PLUS_SELF_CONSISTENCY", "lambda_comp": 0.3, "lambda_prefix": 0.0},
    "C3": {"supervision_class": "PRIVILEGED_PREFIX_ACTION_SUPERVISION", "lambda_comp": 0.0, "lambda_prefix": 1.0},
}
_STOP = False


class GateTokenEncoder(nn.Module):
    """Structural gate tokens only: gate, operands, and continuous parameters."""
    def __init__(self, width: int):
        super().__init__()
        self.gate = nn.Embedding(len(GATES) + 1, width, padding_idx=0)
        self.qubit = nn.Embedding(N_QUBITS + 1, width, padding_idx=N_QUBITS)
        self.parameter = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, gate: torch.Tensor, qubits: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        return self.gate(gate) + self.qubit(qubits[..., 0]) + self.qubit(qubits[..., 1]) + self.parameter(parameters)


class RecursiveOperatorModel(nn.Module):
    """Shared causal residual recurrence followed by the frozen basic Cayley head."""
    def __init__(self, width: int = WIDTH):
        super().__init__()
        self.width = width
        self.token = GateTokenEncoder(width)
        self.initial_state = nn.Parameter(torch.zeros(width))
        self.transition = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, 160)
        # Identical head dimensions and Cayley map to the P4.6 B3 anchor.
        self.head = nn.Sequential(nn.Linear(160, 320), nn.GELU(), nn.Linear(320, 2 * DIM * DIM))

    def hidden_states(self, gates: torch.Tensor, qubits: torch.Tensor, parameters: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.initial_state.expand(gates.shape[0], -1)
        prefixes = []
        for position in range(gates.shape[1]):
            token = self.token(gates[:, position], qubits[:, position], parameters[:, position])
            candidate = self.norm(hidden + self.transition(torch.cat((hidden, token), dim=-1)))
            hidden = torch.where(mask[:, position, None], candidate, hidden)
            prefixes.append(hidden)
        if not prefixes:
            raise ValueError("circuits must have at least one tensor position")
        return torch.stack(prefixes, dim=1)

    def operators_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        raw = self.head(self.output_projection(hidden)).reshape(*hidden.shape[:-1], 2, DIM, DIM)
        b = torch.complex(raw[..., 0, :, :], raw[..., 1, :, :])
        a = 0.5 * (b - b.mH)
        eye = torch.eye(DIM, dtype=a.dtype, device=a.device).expand_as(a)
        return torch.linalg.solve(eye + a, eye - a)

    def forward(self, gates: torch.Tensor, qubits: torch.Tensor, parameters: torch.Tensor, mask: torch.Tensor, return_prefixes: bool = False):
        hidden = self.hidden_states(gates, qubits, parameters, mask)
        final_index = mask.sum(1).clamp_min(1) - 1
        rows = torch.arange(len(hidden), device=hidden.device)
        if not return_prefixes:
            return self.operators_from_hidden(hidden[rows, final_index])
        prefixes = self.operators_from_hidden(hidden)
        return prefixes[rows, final_index], prefixes


def composition_loss(direct: torch.Tensor, second: torch.Tensor, first: torch.Tensor) -> torch.Tensor:
    """U(C2 o C1) is compared with the independently evaluated U(C2) U(C1)."""
    return (1 - process_fidelity(direct, second @ first)).clamp_min(0).mean()


def deterministic_split(sample_id: str | int | bytes, depth: int, seed: int = SEED) -> int:
    if depth < 2:
        raise ValueError("composition split requires depth >= 2")
    raw = sample_id if isinstance(sample_id, bytes) else str(sample_id).encode()
    value = int.from_bytes(hashlib.sha256(raw + seed.to_bytes(8, "big")).digest()[:8], "big")
    return 1 + value % (depth - 1)


def split_circuits(circuits: list[list[Gate]], sample_ids: list[str | int | bytes]) -> tuple[list[list[Gate]], list[list[Gate]], list[list[Gate]]]:
    eligible = [(c, sid) for c, sid in zip(circuits, sample_ids) if len(c) >= 2]
    first, second, combined = [], [], []
    for circuit, sample_id in eligible:
        point = deterministic_split(sample_id, len(circuit))
        first.append(circuit[:point]); second.append(circuit[point:]); combined.append(circuit)
    return first, second, combined


def exact_prefix_actions(circuits: list[list[Gate]], states: np.ndarray, max_depth: int) -> np.ndarray:
    result = np.zeros((len(circuits), max_depth, 2 * DIM), np.float32)
    for row, (circuit, packed) in enumerate(zip(circuits, states)):
        current = packed[:DIM].astype(np.float64) + 1j * packed[DIM:].astype(np.float64)
        for position, gate in enumerate(circuit):
            current = apply_gate(current, gate)
            result[row, position] = np.r_[current.real, current.imag]
    return result


def prefix_action_loss(operators: torch.Tensor, state: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    action = operators @ torch.complex(state[:, :DIM], state[:, DIM:])[:, None, :, None]
    action = action.squeeze(-1)
    packed = torch.cat((action.real, action.imag), dim=-1)
    losses = (1 - state_fidelity(packed, targets)).clamp_min(0)
    return ((losses * mask).sum(1) / mask.sum(1).clamp_min(1)).mean()


def variant_config(variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(variant)
    count = parameter_count(RecursiveOperatorModel())
    return {
        "schema_version": SCHEMA, "variant": variant, "allocation": ALLOCATION, "recipe": RECIPE,
        "architecture": {"type": "shared_causal_residual_recurrence", "recurrent_width": WIDTH, "B3_head_input_width": 160, "B3_head_ff_width": 320, "initial_state": "learned", "shared_transition": True, "gate_embedding": True, "qubit_operand_embeddings": True, "continuous_parameter_encoding": "3-value existing tensor encoding", "analytical_gate_matrices": False, "operator_head": "P4.6_B3_dimensions_plus_basic_cayley", "cayley_scale": 1.0},
        "actual_parameters": count, "C0_parameters": 1_073_312,
        "parameter_fairness_fraction": abs(count - 1_073_312) / 1_073_312,
        "parameter_fairness_pass": abs(count - 1_073_312) / 1_073_312 <= 0.15,
        "composition_split_policy": "depth>=2: 1 + SHA256(decimal A4 pair sample ID || seed2026) mod (depth-1)" if variant == "C2" else None,
        "additional_compute": ({"operator_forward_paths": 4, "reason": "final action plus independently evaluated direct/full, first-part, and second-part composition paths"} if variant == "C2" else ({"operator_forward_paths": "one shared-head output per tensorized prefix (masked loss uses valid prefixes only)", "reason": "privileged prefix action loss"} if variant == "C3" else {"operator_forward_paths": 1})),
        "supervision": ["C", "psi_in", "psi_out"] + (["exact_prefix_actions_as_loss_targets_only"] if variant == "C3" else []),
        "privileged": variant == "C3", **VARIANTS[variant], "scientific_state": "NOT_RUN",
    }


def verify_anchor() -> dict[str, Any]:
    failures = {}
    hashes = {}
    for name, (path, expected) in ANCHOR_FILES.items():
        actual = _sha(path) if path.exists() else None
        hashes[name] = actual
        if actual != expected:
            failures[name] = {"expected": expected, "actual": actual, "path": str(path)}
    if failures:
        raise RuntimeError(f"C0-ANCHOR-MISMATCH: {failures}")
    config = json.loads(ANCHOR_FILES["config"][0].read_text())
    metric = json.loads(ANCHOR_FILES["metric"][0].read_text())
    if config.get("parameterization") != "basic_cayley" or config.get("cayley_scale") != 1.0 or config.get("supervision") != ["C", "psi_in", "psi_out"]:
        raise RuntimeError("C0-ANCHOR-MISMATCH: B3 semantic metadata differs")
    return {
        "schema_version": SCHEMA, "variant": "C0", "role": "frozen P4.6 B3 reference; no rerun",
        "commit": ANCHOR_COMMIT, "config_hash": hashes["config"], "canonical_config_hash": digest(config),
        "dataset_manifest_hash": hashes["dataset_manifest"], "metric_artifact_hash": hashes["metric"], "best_checkpoint_step": metric["best_checkpoint_step"],
        "best_balanced_score": metric["best_balanced_validation"], "actual_parameters": config["actual_parameters"],
        "supervision_class": "ACTION_ONLY", "scientific_state": "REFERENCE_ONLY", "anchor_integrity": "PASS",
    }


def prepare_artifacts() -> dict[str, Any]:
    for directory in ("anchor", "configs", "preflight", "smoke", "metrics", "checkpoints"):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    anchor = verify_anchor()
    atomic_json(ROOT / "anchor/C0.json", anchor)
    configs = {variant: variant_config(variant) for variant in VARIANTS}
    for variant, config in configs.items():
        atomic_json(ROOT / f"configs/{variant}.json", config)
    protocol = {
        "schema_version": SCHEMA, "status": "IMPLEMENTED_NOT_SCIENTIFICALLY_RUN", "seed": SEED,
        "scientific_question": "Can shared recursive encoding improve composition/depth OOD while preserving B3 exact-unitary representation?",
        "variants": {"C0": "frozen P4.6 B3 anchor", "C1": "recursive exact-unitary action-only", "C2": "C1 plus non-tautological operator composition self-consistency", "C3": "C1 plus privileged prefix action supervision"},
        "excluded": ["C4", "multi-seed confirmation", "sealed tests", "arbitrary-qubit scalability", "post-hoc variants"],
        "validation_splits": list(VALIDATION_SPLITS), "sealed_splits": ["composition_ood_test_sealed", "depth_ood_test_sealed"],
        "sealed_test_access_count": 0, "screen_order": list(VARIANTS), "C0_rerun": False,
    }
    summary = {
        **protocol, "anchor": anchor, "allocation": ALLOCATION, "recipe": RECIPE,
        "architecture": configs["C1"]["architecture"],
        "parameter_counts": {"C0": anchor["actual_parameters"], **{v: c["actual_parameters"] for v, c in configs.items()}},
        "supervision_classes": {"C0": "ACTION_ONLY", **{v: c["supervision_class"] for v, c in configs.items()}},
        "compute_note": "Equal optimizer updates and final state-action exposures; C2 triple operator evaluations and C3 prefix heads add FLOPs.",
    }
    atomic_json(ROOT / "protocol.json", protocol)
    atomic_json(ROOT / "protocol_summary.json", summary)
    atomic_json(ROOT / "status.json", {"schema_version": SCHEMA, "state": "IMPLEMENTED_NOT_SCIENTIFICALLY_RUN", "scientific_runs": {v: "NOT_RUN" for v in VARIANTS}, "sealed_test_access_count": 0})
    return summary


def assert_validation_split(split: str) -> None:
    if split not in VALIDATION_SPLITS:
        raise PermissionError(f"non-validation split refused: {split}")


def require_preconditions() -> None:
    verify_anchor()
    manifest = json.loads((DATA_ROOT / "A4/manifest.json").read_text())
    if (manifest["circuit_count"], manifest["probes_per_circuit"], manifest["pair_count"]) != (58_824, 17, 1_000_008):
        raise RuntimeError("P4.7-DATA-ALLOCATION-MISMATCH")
    access = json.loads(Path("artifacts/cc_nqe_p4_6/test_access_log.json").read_text())
    if access.get("access_count") != 0:
        raise RuntimeError("SEALED-TEST-VIOLATION")


def _sample_circuits(data: ArmData, indices: np.ndarray) -> list[list[Gate]]:
    ci = indices // data.probes
    return [_decode(data.gates[int(i)], data.qubits[int(i)], data.parameters[int(i)], data.masks[int(i)]) for i in ci]


def loss_for_batch(variant: str, model: RecursiveOperatorModel, batch: list[torch.Tensor], circuits: list[list[Gate]] | None = None, sample_ids: list[Any] | None = None):
    gates, qubits, parameters, mask, state, target = batch
    if variant == "C3":
        operator, prefixes = model(gates, qubits, parameters, mask, return_prefixes=True)
    else:
        operator, prefixes = model(gates, qubits, parameters, mask), None
    fidelity, norm = action_metrics(operator, state, target)
    final_loss = (1 - fidelity).clamp_min(0).mean()
    comp = prefix = None
    loss = final_loss
    if variant == "C2":
        if circuits is None or sample_ids is None:
            raise ValueError("C2 requires training circuits and sample IDs")
        first, second, combined = split_circuits(circuits, sample_ids)
        if first:
            direct = model(*_circuit_tensors(combined, operator.device))
            pred_first = model(*_circuit_tensors(first, operator.device))
            pred_second = model(*_circuit_tensors(second, operator.device))
            comp = composition_loss(direct, pred_second, pred_first)
        else:
            comp = operator.real.sum() * 0
        loss = loss + 0.3 * comp
    elif variant == "C3":
        if circuits is None:
            raise ValueError("C3 requires exact prefix action loss targets")
        targets = torch.as_tensor(exact_prefix_actions(circuits, state.detach().cpu().numpy(), mask.shape[1]), device=state.device)
        prefix = prefix_action_loss(prefixes, state, targets, mask)
        loss = loss + prefix
    return loss, fidelity, norm, operator, comp, prefix


def evaluate(model: RecursiveOperatorModel, device: torch.device | str) -> dict[str, Any]:
    result = {}
    model.eval()
    with torch.inference_mode():
        for split in VALIDATION_SPLITS:
            assert_validation_split(split)
            batch = [torch.as_tensor(value).to(device) for value in load_validation(split)]
            rows = json.loads((Path("artifacts/cc_nqe_p4_6/datasets") / f"{split}.json").read_text())
            circuits = [[Gate.from_dict(gate) for gate in row["gates"]] for row in rows]
            exact = torch.as_tensor(np.asarray([circuit_unitary(c) for c in circuits], np.complex64)).to(device)
            predicted = model(*batch[:4])
            action, norm = operator_action(predicted, batch[4])
            result[split] = {
                "normalized_action_fidelity": float(state_fidelity(action, batch[5]).mean().cpu()),
                "process_fidelity": float(process_fidelity(predicted, exact).mean().cpu()),
                "phase_aligned_frobenius_error": float(phase_aligned_normalized_frobenius_error(predicted, exact).mean().cpu()),
                "unitarity_error": float(raw_unitarity_error(predicted).mean().cpu()),
                "raw_action_norm": float(norm.mean().cpu()),
            }
    model.train()
    result["S_balanced"] = sum(result[s]["normalized_action_fidelity"] for s in ("iid_validation", "composition_ood_validation", "depth_ood_validation")) / 3
    return result


def validate_checkpoint(payload: dict[str, Any], config: dict[str, Any], dataset_hash: str) -> None:
    required = {"model", "optimizer", "scheduler", "step", "samples_seen", "numpy_rng", "torch_rng", "xpu_rng", "best_checkpoint_state"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"resume refused: incomplete checkpoint ({', '.join(sorted(missing))})")
    if payload.get("variant") != config["variant"] or payload.get("config_hash") != digest(config):
        raise ValueError("resume refused: variant/config hash differs")
    if payload.get("dataset_manifest_hash") != dataset_hash:
        raise ValueError("resume refused: dataset manifest hash differs")
    if not isinstance(payload["step"], int) or payload["step"] < 0 or not isinstance(payload["best_checkpoint_state"], dict) or not {"score", "step"} <= payload["best_checkpoint_state"].keys():
        raise ValueError("resume refused: invalid checkpoint state")


def _checkpoint_payload(model, optimizer, scheduler, config, dataset_hash, step, samples, rng, best, best_step):
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": step, "samples_seen": samples, "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(), "xpu_rng": torch.xpu.get_rng_state() if torch.xpu.is_available() else None, "variant": config["variant"], "config_hash": digest(config), "dataset_manifest_hash": dataset_hash, "best_checkpoint_state": {"score": best, "step": best_step}}


def save_checkpoint(path: Path, *args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(_checkpoint_payload(*args), temp)
    os.replace(temp, path)


def preflight() -> dict[str, Any]:
    prepare_artifacts()
    result = {"schema_version": SCHEMA, "scientific_run": False, "purpose": "implementation_validation", "maximum_updates": 1, "device": "xpu:0", "checks": {}, "status": "P4.7-XPU-BLOCKED"}
    path = ROOT / "preflight/preflight.json"
    try:
        require_preconditions()
        if not torch.xpu.is_available():
            raise RuntimeError("native XPU unavailable; no CPU fallback")
        torch.manual_seed(SEED)
        device = torch.device("xpu:0")
        model = RecursiveOperatorModel().to(device)
        circuits = [[Gate("H", (0,)), Gate("CNOT", (0, 1))], [Gate("RX", (2,), 0.4), Gate("X", (3,))]]
        args = _circuit_tensors(circuits, device)
        state = torch.randn(2, 2 * DIM, device=device)
        target = torch.randn_like(state)
        optimizer = torch.optim.AdamW(model.parameters(), 3e-4)
        before = next(model.parameters()).detach().clone()
        batch = [*args, state, target]
        c2_loss, _, _, operators, comp, _ = loss_for_batch("C2", model, batch, circuits, ["a", "b"])
        optimizer.zero_grad(); c2_loss.backward(retain_graph=False)
        c2_gradient = sum(float(p.grad.abs().sum().detach().cpu()) for p in model.parameters() if p.grad is not None)
        optimizer.step(); torch.xpu.synchronize()
        c3_loss, _, _, _, _, prefix = loss_for_batch("C3", model, batch, circuits)
        optimizer.zero_grad(); c3_loss.backward(); torch.xpu.synchronize()
        from torch.profiler import ProfilerActivity, profile
        optimizer.zero_grad(set_to_none=True)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as profiler:
            profiled = model(*args)
            (1 - process_fidelity(profiled, torch.eye(DIM, dtype=profiled.dtype, device=device)[None])).mean().backward()
            torch.xpu.synchronize()
        solve_device_time = sum(event.device_time_total for event in profiler.key_averages() if any(token in event.key.lower() for token in ("solve", "getrf", "trsm")))
        identity = torch.eye(DIM, dtype=operators.dtype, device=device)[None]
        synthetic_consistency = float(composition_loss(identity, identity, identity).detach().cpu())
        checks = {
            "anchor_integrity": True, "data_allocation": True, "sealed_access_zero": True,
            "xpu_residency": operators.device.type == "xpu", "no_cpu_fallback": operators.device.type == "xpu",
            "profiler_native_xpu_solve": solve_device_time > 0,
            "cayley_exact_unitarity": float(raw_unitarity_error(operators).max().detach().cpu()) < 1e-4,
            "finite_nonzero_c2_gradient": math.isfinite(c2_gradient) and c2_gradient > 0,
            "nonzero_composition_loss": float(comp.detach().cpu()) > 1e-7,
            "synthetic_consistency_zero": synthetic_consistency < 1e-6,
            "finite_prefix_loss": bool(torch.isfinite(prefix)), "C3_no_teacher_forcing": "state" not in RecursiveOperatorModel.forward.__code__.co_varnames[:6],
            "optimizer_update": not torch.equal(before, next(model.parameters()).detach()),
            "parameter_fairness": variant_config("C1")["parameter_fairness_pass"],
        }
        result.update(checks=checks, status="PASS" if all(checks.values()) else "P4.7-IMPLEMENTATION-BLOCKED", parameter_counts={v: variant_config(v)["actual_parameters"] for v in VARIANTS}, composition_loss=float(comp.detach().cpu()), synthetic_consistency_loss=synthetic_consistency, c2_gradient_l1=c2_gradient, prefix_loss=float(prefix.detach().cpu()), unitarity_error=float(raw_unitarity_error(operators).max().detach().cpu()), profiler_solve_device_time_total=solve_device_time, device_name=torch.xpu.get_device_name(0))
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    atomic_json(path, result)
    return result


def smoke() -> dict[str, Any]:
    require_preconditions()
    pre = preflight()
    result = {"schema_version": SCHEMA, "scientific_run": False, "purpose": "implementation_validation", "maximum_updates": 1, "scientific_runs": "NONE", "variants": {}, "status": "P4.7-IMPLEMENTATION-BLOCKED"}
    if pre["status"] != "PASS":
        result["reason"] = "preflight did not pass"
        atomic_json(ROOT / "smoke/C1-C3.json", result)
        return result
    device = torch.device("xpu:0")
    data = ArmData("A4")
    indices = np.arange(4) * data.probes
    circuits = _sample_circuits(data, indices)
    batch = [torch.as_tensor(value).to(device) for value in data.batch(indices)]
    dataset_hash = _sha(DATA_ROOT / "A4/manifest.json")
    for variant in VARIANTS:
        torch.manual_seed(SEED)
        model = RecursiveOperatorModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), 3e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, RECIPE["maximum_updates"])
        before = next(model.parameters()).detach().clone()
        loss, fidelity, norm, operator, comp, prefix = loss_for_batch(variant, model, batch, circuits, [int(i) for i in indices])
        optimizer.zero_grad(); loss.backward()
        finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        optimizer.step(); scheduler.step(); torch.xpu.synchronize()
        config = variant_config(variant)
        rng = np.random.default_rng(SEED)
        payload = _checkpoint_payload(model, optimizer, scheduler, config, dataset_hash, 1, len(indices), rng, -math.inf, None)
        validate_checkpoint(payload, config, dataset_hash)
        buffer = io.BytesIO(); torch.save(payload, buffer); buffer.seek(0)
        loaded = torch.load(buffer, map_location="cpu", weights_only=False); validate_checkpoint(loaded, config, dataset_hash)
        restored = RecursiveOperatorModel().to(device); restored.load_state_dict(loaded["model"])
        restored_optimizer = torch.optim.AdamW(restored.parameters(), 3e-4); restored_optimizer.load_state_dict(loaded["optimizer"])
        restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_optimizer, RECIPE["maximum_updates"]); restored_scheduler.load_state_dict(loaded["scheduler"])
        exact = torch.as_tensor(np.asarray([circuit_unitary(circuit) for circuit in circuits], np.complex64)).to(device)
        passed = finite and bool(torch.isfinite(loss)) and not torch.equal(before, next(model.parameters()).detach()) and operator.device.type == "xpu"
        result["variants"][variant] = {"status": "PASS" if passed else "FAIL", "scientific_run": False, "purpose": "implementation_validation", "optimizer_updates": 1, "loss": float(loss.detach().cpu()), "action_fidelity": float(fidelity.mean().detach().cpu()), "process_fidelity": float(process_fidelity(operator, exact).mean().detach().cpu()), "unitarity_error": float(raw_unitarity_error(operator).mean().detach().cpu()), "raw_action_norm": float(norm.mean().detach().cpu()), "composition_loss": float(comp.detach().cpu()) if comp is not None else None, "prefix_loss": float(prefix.detach().cpu()) if prefix is not None else None, "xpu_residency": operator.device.type == "xpu", "finite_gradients": finite, "checkpoint_resume": "PASS"}
    result["status"] = "PASS" if all(v["status"] == "PASS" for v in result["variants"].values()) else "P4.7-IMPLEMENTATION-BLOCKED"
    atomic_json(ROOT / "smoke/C1-C3.json", result)
    return result


def _status_row(**values) -> None:
    row = {"schema_version": SCHEMA, "timestamp": time.time(), **values}
    atomic_json(ROOT / "status.json", row)
    with (ROOT / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    print(" ".join(f"{key}={value}" for key, value in values.items()), flush=True)


def run(variant: str) -> dict[str, Any]:
    """Manual scientific entry point. Never called by preflight or smoke."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    require_preconditions()
    if preflight()["status"] != "PASS":
        raise RuntimeError("P4.7-XPU-BLOCKED")
    metric_path = ROOT / f"metrics/{variant}.json"
    if metric_path.exists():
        existing = json.loads(metric_path.read_text())
        if existing.get("state") == "COMPLETED":
            return existing
    config = variant_config(variant)
    dataset_hash = _sha(DATA_ROOT / "A4/manifest.json")
    atomic_json(ROOT / f"configs/{variant}.json", config)
    device = torch.device("xpu:0")
    torch.manual_seed(SEED)
    data = ArmData("A4")
    model = RecursiveOperatorModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), RECIPE["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, RECIPE["maximum_updates"])
    rng = np.random.default_rng(SEED)
    latest_path = ROOT / f"checkpoints/{variant}-latest.pt"
    best_path = ROOT / f"checkpoints/{variant}-best-balanced.pt"
    step = samples = 0; best = -math.inf; best_step = None
    if latest_path.exists():
        payload = torch.load(latest_path, map_location="cpu", weights_only=False)
        validate_checkpoint(payload, config, dataset_hash)
        model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"]); scheduler.load_state_dict(payload["scheduler"])
        step, samples = payload["step"], payload["samples_seen"]
        rng.bit_generator.state = payload["numpy_rng"]; torch.set_rng_state(payload["torch_rng"])
        if payload.get("xpu_rng") is not None: torch.xpu.set_rng_state(payload["xpu_rng"])
        best, best_step = payload["best_checkpoint_state"]["score"], payload["best_checkpoint_state"]["step"]
    started = time.monotonic(); start_samples = samples; validation = {}; loss = fidelity = norm = operator = comp = prefix = None
    while step < RECIPE["maximum_updates"] and not _STOP:
        indices = rng.integers(data.length, size=RECIPE["effective_batch_size"])
        circuits = _sample_circuits(data, indices)
        batch = [torch.as_tensor(value).to(device) for value in data.batch(indices)]
        loss, fidelity, norm, operator, comp, prefix = loss_for_batch(variant, model, batch, circuits, [int(i) for i in indices])
        optimizer.zero_grad(set_to_none=True); loss.backward()
        if not bool(torch.isfinite(loss)) or not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise FloatingPointError(f"{variant}: non-finite training")
        optimizer.step(); scheduler.step(); step += 1; samples += len(indices)
        checkpoint_name = str(latest_path)
        if step % RECIPE["validation_interval"] == 0 or step == RECIPE["maximum_updates"]:
            torch.xpu.synchronize(); validation = evaluate(model, device)
            if validation["S_balanced"] > best:
                best, best_step = validation["S_balanced"], step
                save_checkpoint(best_path, model, optimizer, scheduler, config, dataset_hash, step, samples, rng, best, best_step)
                checkpoint_name = str(best_path)
            save_checkpoint(latest_path, model, optimizer, scheduler, config, dataset_hash, step, samples, rng, best, best_step)
        if step % 50 == 0 or step % RECIPE["validation_interval"] == 0:
            elapsed = time.monotonic() - started
            metric = lambda split: validation.get(split, {}).get("normalized_action_fidelity")
            _status_row(variant=variant, step=f"{step}/{RECIPE['maximum_updates']}", loss=float(loss.detach().cpu()), action_fidelity=float(fidelity.mean().detach().cpu()), process_fidelity=validation.get("iid_validation", {}).get("process_fidelity"), unitarity_error=float(raw_unitarity_error(operator).mean().detach().cpu()), IID_val=metric("iid_validation"), State_OOD_val=metric("state_ood_validation"), Parameter_OOD_val=metric("parameter_ood_validation"), Composition_OOD_val=metric("composition_ood_validation"), Depth_OOD_val=metric("depth_ood_validation"), S_balanced=validation.get("S_balanced"), lr=optimizer.param_groups[0]["lr"], samples_per_second=(samples - start_samples) / max(elapsed, 1e-9), elapsed=elapsed, ETA=(RECIPE["maximum_updates"] - step) * elapsed / max(step, 1), device="xpu:0", checkpoint=checkpoint_name, composition_loss=float(comp.detach().cpu()) if comp is not None else None, prefix_loss=float(prefix.detach().cpu()) if prefix is not None else None, state="RUNNING", scientific_runs={v: ("RUNNING" if v == variant else status()["scientific_runs"].get(v, "NOT_RUN")) for v in VARIANTS}, sealed_test_access_count=0)
    save_checkpoint(latest_path, model, optimizer, scheduler, config, dataset_hash, step, samples, rng, best, best_step)
    state = "INTERRUPTED" if _STOP else "COMPLETED"
    if state == "COMPLETED":
        payload = torch.load(best_path, map_location="cpu", weights_only=False); validate_checkpoint(payload, config, dataset_hash); model.load_state_dict(payload["model"]); validation = evaluate(model, device)
    runtime = time.monotonic() - started
    result = {"schema_version": SCHEMA, "variant": variant, "state": state, "step": step, "samples_seen": samples, "best_balanced_validation": best, "best_checkpoint_step": best_step, "validation_at_best_balanced_checkpoint": validation, "runtime_seconds": runtime, "samples_per_second": (samples - start_samples) / max(runtime, 1e-9), "scientific": True, "sealed_test_access_count": 0}
    atomic_json(metric_path, result)
    return result


def screen() -> dict[str, Any]:
    require_preconditions()
    if preflight()["status"] != "PASS":
        raise RuntimeError("P4.7-XPU-BLOCKED: screen refused before any scientific variant")
    results = {}
    for variant in ("C1", "C2", "C3"):
        results[variant] = run(variant)
        if results[variant]["state"] != "COMPLETED":
            break
    return results


def status() -> dict[str, Any]:
    value = json.loads((ROOT / "status.json").read_text()) if (ROOT / "status.json").exists() else {"schema_version": SCHEMA, "state": "PENDING"}
    value["scientific_runs"] = {variant: (json.loads((ROOT / f"metrics/{variant}.json").read_text())["state"] if (ROOT / f"metrics/{variant}.json").exists() else "NOT_RUN") for variant in VARIANTS}
    value["sealed_test_access_count"] = 0
    return value


def install_signal_handlers() -> None:
    def stop(_signum, _frame):
        global _STOP
        _STOP = True
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
