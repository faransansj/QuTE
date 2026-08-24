"""Minimal, reproducible CC-NQE P1-P4 experiment components (4 qubits only)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import resource
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_info, threadpool_limits
from torch import nn

N_QUBITS = 4
DIM = 1 << N_QUBITS
SCHEMA_VERSION = "cc-nqe-p1-v1"
SIMULATOR = "cc_nqe.numpy_exact_statevector_v1"
NORM_TOL = 1e-12
GATES = ("H", "X", "RX", "RY", "RZ", "CNOT")
PARAM_GATES = ("RX", "RY", "RZ")
GATE_TO_ID = {g: i + 1 for i, g in enumerate(GATES)}  # zero is padding
PARAM_REGIONS = {
    "train": ((0.0, 2 * math.pi / 3), (math.pi, 5 * math.pi / 3)),
    "interpolation": ((2 * math.pi / 3, math.pi),),
    "extrapolation": ((5 * math.pi / 3, 2 * math.pi),),
}
COMPOSITION_MOTIF = ("CNOT", 0, 1)

# ponytail: accelerator selection, cuda > xpu > cpu; single source of truth for device strings
ACCEL = "cuda" if torch.cuda.is_available() else ("xpu" if getattr(torch, "xpu", None) and torch.xpu.is_available() else "cpu")
ACCEL_DEVICE = torch.device(f"{ACCEL}:0") if ACCEL != "cpu" else torch.device("cpu")


def accel_synchronize() -> None:
    if ACCEL == "cuda":
        torch.cuda.synchronize()
    elif ACCEL == "xpu":
        torch.xpu.synchronize()


def accel_device_name(index: int = 0) -> str:
    if ACCEL == "cuda":
        return torch.cuda.get_device_name(index)
    if ACCEL == "xpu":
        return torch.xpu.get_device_name(index)
    return platform.processor() or "CPU"


def accel_rng_state():
    if ACCEL == "cuda":
        return torch.cuda.get_rng_state()
    if ACCEL == "xpu":
        return torch.xpu.get_rng_state()
    return None


def accel_set_rng_state(state) -> None:
    if state is None:
        return
    if ACCEL == "cuda":
        torch.cuda.set_rng_state(state)
    elif ACCEL == "xpu":
        torch.xpu.set_rng_state(state)


def accel_profiler_activities() -> list:
    from torch.profiler import ProfilerActivity
    activities = [ProfilerActivity.CPU]
    if ACCEL != "cpu":
        activities.append(getattr(ProfilerActivity, ACCEL.upper()))
    return activities


@dataclass(frozen=True)
class Gate:
    name: str
    qubits: tuple[int, ...]
    theta: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "qubits": list(self.qubits), "theta": self.theta}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Gate":
        return cls(value["name"], tuple(value["qubits"]), value.get("theta"))


def serialize_circuit(circuit: list[Gate]) -> str:
    return json.dumps([g.to_dict() for g in circuit], sort_keys=True, separators=(",", ":"), allow_nan=False)


def deserialize_circuit(value: str) -> list[Gate]:
    return [Gate.from_dict(g) for g in json.loads(value)]


def circuit_id(circuit: list[Gate]) -> str:
    return "c_" + hashlib.sha256(serialize_circuit(circuit).encode()).hexdigest()[:16]


def _single_matrix(gate: Gate) -> np.ndarray:
    if gate.name == "H":
        return np.array([[1, 1], [1, -1]], np.complex128) / math.sqrt(2)
    if gate.name == "X":
        return np.array([[0, 1], [1, 0]], np.complex128)
    t = float(gate.theta)
    if gate.name == "RX":
        return np.array([[math.cos(t / 2), -1j * math.sin(t / 2)], [-1j * math.sin(t / 2), math.cos(t / 2)]], np.complex128)
    if gate.name == "RY":
        return np.array([[math.cos(t / 2), -math.sin(t / 2)], [math.sin(t / 2), math.cos(t / 2)]], np.complex128)
    if gate.name == "RZ":
        return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)]).astype(np.complex128)
    raise ValueError(gate.name)


def apply_gate(state: np.ndarray, gate: Gate) -> np.ndarray:
    """Apply one gate with q0 as the most significant computational-basis bit."""
    state = np.asarray(state, dtype=np.complex128)
    if gate.name != "CNOT":
        q = gate.qubits[0]
        tensor = np.moveaxis(state.reshape((2,) * N_QUBITS), q, 0)
        return np.moveaxis(np.tensordot(_single_matrix(gate), tensor, axes=(1, 0)), 0, q).reshape(DIM)
    control, target = gate.qubits
    out = np.empty_like(state)
    for i in range(DIM):
        j = i ^ (1 << (N_QUBITS - 1 - target)) if i & (1 << (N_QUBITS - 1 - control)) else i
        out[j] = state[i]
    return out


def simulate(circuit: list[Gate], state: np.ndarray) -> np.ndarray:
    out = np.asarray(state, dtype=np.complex128).copy()
    for gate in circuit:
        out = apply_gate(out, gate)
    return out


def circuit_unitary(circuit: list[Gate]) -> np.ndarray:
    return np.column_stack([simulate(circuit, np.eye(DIM, dtype=np.complex128)[:, i]) for i in range(DIM)])


def _sample_angle(rng: np.random.Generator, regime: str) -> float:
    regions = PARAM_REGIONS[regime]
    low, high = regions[int(rng.integers(len(regions)))]
    return float(rng.uniform(low, high))


def has_composition_motif(circuit: list[Gate]) -> bool:
    return any((g.name, *g.qubits) == COMPOSITION_MOTIF for g in circuit)


def structural_signature(circuit: list[Gate]) -> str:
    """Parameter-free ordered gate/qubit signature used for composition leakage checks."""
    return "|".join(f"{g.name}:{','.join(map(str, g.qubits))}" for g in circuit)


def generate_circuit(seed: int, depth: int, regime: str = "train", require_motif: bool = False) -> list[Gate]:
    rng = np.random.default_rng(seed)
    circuit: list[Gate] = []
    forced_pos = int(rng.integers(depth)) if require_motif else -1
    forced_param = regime != "train"
    param_pos = int(rng.integers(depth)) if forced_param else -1
    if param_pos == forced_pos:
        param_pos = (param_pos + 1) % depth
    for pos in range(depth):
        if pos == forced_pos:
            gate = Gate("CNOT", (0, 1))
        else:
            choices = list(GATES)
            if pos == param_pos:
                choices = list(PARAM_GATES)
            for _ in range(100):
                name = str(rng.choice(choices))
                if name == "CNOT":
                    control = int(rng.integers(N_QUBITS))
                    target = int(rng.integers(N_QUBITS - 1))
                    target += target >= control
                    gate = Gate(name, (control, target))
                    if not require_motif and (name, control, target) == COMPOSITION_MOTIF:
                        continue
                else:
                    q = int(rng.integers(N_QUBITS))
                    gate = Gate(name, (q,), _sample_angle(rng, regime) if name in PARAM_GATES else None)
                # Excludes exact adjacent repetition/cancellation (H-H, X-X, R(theta)-R(-theta), duplicate CNOT).
                if not circuit or not (gate.name == circuit[-1].name and gate.qubits == circuit[-1].qubits):
                    break
            else:
                raise RuntimeError("could not generate nontrivial gate")
        circuit.append(gate)
    return circuit


def generate_unique_circuits(counts: dict[int, int], seed: int, regime: str = "train", require_motif: bool = False) -> list[dict[str, Any]]:
    records, seen_serialized, seen_structures = [], set(), set()
    attempt = 0
    for depth, count in counts.items():
        made = 0
        while made < count:
            generator_seed = seed + attempt
            attempt += 1
            circuit = generate_circuit(generator_seed, depth, regime, require_motif)
            serial = serialize_circuit(circuit)
            structure = structural_signature(circuit)
            if serial in seen_serialized or structure in seen_structures:
                continue
            # Reject global-phase identity circuits numerically.
            unitary = circuit_unitary(circuit)
            phase = unitary.flat[np.argmax(np.abs(unitary))]
            if np.linalg.norm(unitary - np.eye(DIM) * phase / abs(phase)) < 1e-10:
                continue
            seen_serialized.add(serial)
            seen_structures.add(structure)
            records.append({"circuit_id": circuit_id(circuit), "depth": depth, "gates": circuit, "generator_seed": generator_seed})
            made += 1
    return records


def _kron(states: Iterable[np.ndarray]) -> np.ndarray:
    out = np.array([1.0 + 0j])
    for state in states:
        out = np.kron(out, state)
    return out.astype(np.complex128)


def generate_state(seed: int, family: str) -> np.ndarray:
    """Generate labeled states: basis/Pauli product, local-rotation product, entangled, or Haar."""
    rng = np.random.default_rng(seed)
    zero, one = np.array([1, 0], complex), np.array([0, 1], complex)
    plus, plus_i = np.array([1, 1], complex) / math.sqrt(2), np.array([1, 1j], complex) / math.sqrt(2)
    if family == "product":
        state = _kron([rng.choice([zero, one, plus, plus_i]) for _ in range(N_QUBITS)])
    elif family == "random-local":
        local = []
        for _ in range(N_QUBITS):
            theta, phi = rng.uniform(0, math.pi), rng.uniform(0, 2 * math.pi)
            local.append(np.array([math.cos(theta / 2), np.exp(1j * phi) * math.sin(theta / 2)]))
        state = _kron(local)
    elif family == "entangled":
        local = generate_state(seed + 10_000_000, "random-local")
        state = simulate([Gate("CNOT", (0, 1)), Gate("CNOT", (1, 2)), Gate("CNOT", (2, 3)), Gate("RY", (0,), float(rng.uniform(0.2, 2.9)))], local)
    elif family == "Haar-random":
        state = rng.normal(size=DIM) + 1j * rng.normal(size=DIM)
    else:
        raise ValueError(f"unknown state family: {family}")
    return (state / np.linalg.norm(state)).astype(np.complex128)


def generate_states(per_family: int, seed: int, prefix: str) -> list[dict[str, Any]]:
    out = []
    for family_index, family in enumerate(("product", "random-local", "entangled", "Haar-random")):
        for i in range(per_family):
            generator_seed = seed + family_index * 10_000 + i
            out.append({"state_id": f"s_{prefix}_{family_index}_{i}", "family": family, "generator_seed": generator_seed,
                        "state": generate_state(generator_seed, family)})
    return out


def complex_to_real(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x.real, x.imag], axis=-1)


def real_to_complex(x: np.ndarray) -> np.ndarray:
    return x[..., :DIM] + 1j * x[..., DIM:]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def build_dataset(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Build compact NPZ arrays plus reconstructable JSONL metadata and manifests."""
    root.mkdir(parents=True, exist_ok=True)
    base = int(config["dataset_seed"])
    circuit_specs = {
        "train": ({2: 16, 4: 16, 6: 16}, "train", False),
        "validation": ({2: 2, 4: 2, 6: 2}, "train", False),
        "iid": ({2: 3, 4: 3, 6: 2}, "train", False),
        "parameter_interpolation": ({2: 3, 4: 3, 6: 2}, "interpolation", False),
        "parameter_extrapolation": ({2: 3, 4: 3, 6: 2}, "extrapolation", False),
        "composition_ood": ({2: 3, 4: 3, 6: 2}, "train", True),
        "depth_ood": ({8: 8}, "train", False),
    }
    circuits: dict[str, list[dict[str, Any]]] = {}
    used_structures: set[str] = set()
    for split_index, (split, (counts, regime, motif)) in enumerate(circuit_specs.items()):
        # Different seed ranges; reject structure collisions across independently held-out circuit splits.
        candidate_seed = base + (split_index + 1) * 100_000
        while True:
            generated = generate_unique_circuits(counts, candidate_seed, regime, motif)
            structures = {structural_signature(x["gates"]) for x in generated}
            if not structures & used_structures:
                break
            candidate_seed += 10_000
        circuits[split] = generated
        used_structures |= structures
    # State-OOD intentionally reuses train circuit contexts and exclusively new states.
    circuits["state_ood"] = circuits["train"][:8]
    states = {
        "train": generate_states(4, base + 1_000, "train"),
        "validation": generate_states(2, base + 2_000, "validation"),
        "iid": generate_states(2, base + 1_000, "train"),  # known states, unseen circuits
        "state_ood": generate_states(2, base + 3_000, "stateood"),
        "parameter_interpolation": generate_states(2, base + 4_000, "pint"),
        "parameter_extrapolation": generate_states(2, base + 5_000, "pext"),
        "composition_ood": generate_states(2, base + 6_000, "comp"),
        "depth_ood": generate_states(2, base + 7_000, "depth"),
    }
    rows, inputs, targets = [], [], []
    for split in circuit_specs | {"state_ood": None}:
        split_type = "train" if split == "train" else ("validation" if split == "validation" else "test")
        for circuit in circuits[split]:
            gates = circuit["gates"]
            for state_record in states[split]:
                inp = state_record["state"]
                target = simulate(gates, inp)
                if abs(np.linalg.norm(inp) - 1) > NORM_TOL or abs(np.linalg.norm(target) - 1) > NORM_TOL:
                    raise ValueError("teacher normalization failure")
                index = len(rows)
                sample_id = f"sample_{index:06d}"
                inputs.append(inp)
                targets.append(target)
                rows.append({
                    "sample_id": sample_id, "array_index": index, "circuit_id": circuit["circuit_id"],
                    "state_id": state_record["state_id"], "n_qubits": N_QUBITS, "depth": circuit["depth"],
                    "gate_sequence": [g.name for g in gates], "gate_parameters": [g.theta for g in gates],
                    "gate_sequence_structured": [g.to_dict() for g in gates], "input_state": {"artifact": "samples.npz", "array": "input_state", "index": index},
                    "target_state": {"artifact": "samples.npz", "array": "target_state", "index": index},
                    "state_family": state_record["family"], "split_type": split_type, "split_name": split,
                    "generator_seed": {"circuit": circuit["generator_seed"], "state": state_record["generator_seed"]},
                    "simulator_backend": SIMULATOR, "schema_version": SCHEMA_VERSION,
                    "structural_signature": structural_signature(gates),
                })
    npz_path = root / "samples.npz"
    np.savez_compressed(npz_path, input_state=np.asarray(inputs, np.complex128), target_state=np.asarray(targets, np.complex128))
    jsonl_path = root / "samples.jsonl"
    jsonl_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))
    split_manifest = {
        "state_ood": "test state_id values are disjoint from train; same train circuit contexts are reused",
        "parameter_ood_interpolation": {"test": PARAM_REGIONS["interpolation"], "train_excludes": PARAM_REGIONS["interpolation"]},
        "parameter_ood_extrapolation": {"test": PARAM_REGIONS["extrapolation"], "train_excludes": PARAM_REGIONS["extrapolation"]},
        "composition_ood": {"withheld_motif": list(COMPOSITION_MOTIF), "policy": "all test circuits contain directed CNOT 0->1; train circuits exclude it"},
        "depth_ood": {"train_depths": [2, 4, 6], "test_depths": [8]},
        "iid": "unseen circuits from training distribution, state IDs reused from train",
    }
    write_json(root / "split_manifest.json", split_manifest)
    manifest = {
        "schema_version": SCHEMA_VERSION, "generation_config": config, "seeds": {"dataset": base},
        "sample_counts": dict(Counter(row["split_name"] for row in rows)),
        "state_family_counts": dict(Counter(row["state_family"] for row in rows)),
        "depth_counts": {str(k): v for k, v in Counter(row["depth"] for row in rows).items()},
        "gate_counts": dict(Counter(g for row in rows for g in row["gate_sequence"])),
        "unique_circuit_counts": {s: len({r["circuit_id"] for r in rows if r["split_name"] == s}) for s in states},
        "unique_state_counts": {s: len({r["state_id"] for r in rows if r["split_name"] == s}) for s in states},
        "split_definitions": split_manifest, "git_commit": git_sha(), "simulator": SIMULATOR,
        "storage_dtype": "complex128", "training_dtype": "float32 (explicit conversion)",
        "numerical_tolerances": {"normalization": NORM_TOL},
        "file_hashes": {"samples.npz": sha256(npz_path), "samples.jsonl": sha256(jsonl_path), "split_manifest.json": sha256(root / "split_manifest.json")},
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def load_dataset(root: Path) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    rows = [json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines()]
    arrays = np.load(root / "samples.npz")
    return rows, arrays["input_state"], arrays["target_state"]


def _angle_in(theta: float, ranges: tuple[tuple[float, float], ...]) -> bool:
    return any(low <= theta < high for low, high in ranges)


def audit_dataset(root: Path) -> dict[str, Any]:
    rows, inputs, targets = load_dataset(root)
    by_split = {s: [r for r in rows if r["split_name"] == s] for s in {r["split_name"] for r in rows}}
    train = by_split["train"]
    train_states = {r["state_id"] for r in train}
    train_circuits = {r["circuit_id"] for r in train}
    train_structures = {r["structural_signature"] for r in train}
    checks: dict[str, bool] = {}
    checks["state_ood_no_state_leakage"] = not train_states & {r["state_id"] for r in by_split["state_ood"]}
    checks["all_ood_no_state_leakage"] = all(not train_states & {r["state_id"] for r in by_split[s]} for s in ("state_ood", "parameter_interpolation", "parameter_extrapolation", "composition_ood", "depth_ood"))
    checks["heldout_circuit_ids_disjoint"] = all(not train_circuits & {r["circuit_id"] for r in by_split[s]} for s in ("validation", "iid", "parameter_interpolation", "parameter_extrapolation", "composition_ood", "depth_ood"))
    checks["heldout_structures_disjoint"] = all(not train_structures & {r["structural_signature"] for r in by_split[s]} for s in ("validation", "iid", "composition_ood", "depth_ood"))
    checks["parameter_interpolation_contract"] = all(_angle_in(t, PARAM_REGIONS["interpolation"]) for r in by_split["parameter_interpolation"] for t in r["gate_parameters"] if t is not None)
    checks["parameter_extrapolation_contract"] = all(_angle_in(t, PARAM_REGIONS["extrapolation"]) for r in by_split["parameter_extrapolation"] for t in r["gate_parameters"] if t is not None)
    heldout = PARAM_REGIONS["interpolation"] + PARAM_REGIONS["extrapolation"]
    checks["training_excludes_parameter_holdouts"] = all(not _angle_in(t, heldout) for r in train for t in r["gate_parameters"] if t is not None)
    checks["composition_contract"] = all(has_composition_motif([Gate.from_dict(g) for g in r["gate_sequence_structured"]]) for r in by_split["composition_ood"]) and all(not has_composition_motif([Gate.from_dict(g) for g in r["gate_sequence_structured"]]) for r in train)
    checks["depth_contract"] = all(r["depth"] <= 6 for r in train) and all(r["depth"] == 8 for r in by_split["depth_ood"])
    serial_by_id = {r["circuit_id"]: json.dumps(r["gate_sequence_structured"], sort_keys=True) for r in rows}
    checks["no_duplicate_circuit_serializations"] = len(set(serial_by_id.values())) == len(serial_by_id)
    checks["state_normalization"] = bool(np.max(np.abs(np.linalg.norm(inputs, axis=1) - 1)) <= NORM_TOL)
    checks["target_normalization"] = bool(np.max(np.abs(np.linalg.norm(targets, axis=1) - 1)) <= NORM_TOL)
    # Deterministically regenerate every state and target from recorded seeds/context.
    regenerated = True
    for row, inp, target in zip(rows, inputs, targets):
        expected_in = generate_state(row["generator_seed"]["state"], row["state_family"])
        circuit = [Gate.from_dict(g) for g in row["gate_sequence_structured"]]
        regime = "interpolation" if row["split_name"] == "parameter_interpolation" else ("extrapolation" if row["split_name"] == "parameter_extrapolation" else "train")
        expected_circuit = generate_circuit(row["generator_seed"]["circuit"], row["depth"], regime, row["split_name"] == "composition_ood")
        if circuit != expected_circuit or not np.array_equal(inp, expected_in) or not np.allclose(target, simulate(circuit, expected_in), atol=1e-14, rtol=1e-14):
            regenerated = False
            break
    checks["deterministic_regeneration"] = regenerated
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "duplicate_sample_count": len(rows) - len({(r["circuit_id"], r["state_id"], r["split_name"]) for r in rows}),
        "max_input_norm_error": float(np.max(np.abs(np.linalg.norm(inputs, axis=1) - 1))),
        "max_target_norm_error": float(np.max(np.abs(np.linalg.norm(targets, axis=1) - 1))),
    }
    write_json(root / "audit.json", result)
    return result


def encode_circuits(rows: list[dict[str, Any]], max_depth: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gate_ids = np.zeros((len(rows), max_depth), np.int64)
    qubits = np.full((len(rows), max_depth, 2), N_QUBITS, np.int64)  # N_QUBITS is padding id
    params = np.zeros((len(rows), max_depth, 3), np.float32)  # sin, cos, has-parameter
    mask = np.zeros((len(rows), max_depth), bool)
    for i, row in enumerate(rows):
        for j, gate_value in enumerate(row["gate_sequence_structured"]):
            gate = Gate.from_dict(gate_value)
            gate_ids[i, j] = GATE_TO_ID[gate.name]
            qubits[i, j, :len(gate.qubits)] = gate.qubits
            if gate.theta is not None:
                params[i, j] = (math.sin(gate.theta), math.cos(gate.theta), 1.0)
            mask[i, j] = True
    return gate_ids, qubits, params, mask


def flat_circuits(gate_ids: torch.Tensor, qubits: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    gate_onehot = torch.nn.functional.one_hot(gate_ids, len(GATES) + 1)[..., 1:].float()
    q0 = torch.nn.functional.one_hot(qubits[..., 0], N_QUBITS + 1)[..., :N_QUBITS].float()
    q1 = torch.nn.functional.one_hot(qubits[..., 1], N_QUBITS + 1)[..., :N_QUBITS].float()
    return torch.cat((gate_onehot, q0, q1, params), -1).flatten(1)


class StateOnly(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(32, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 32))
    def forward(self, state, gate_ids=None, qubits=None, params=None, mask=None):
        return self.net(state)


class FlatMLP(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(32 + 8 * 17, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 32))
    def forward(self, state, gate_ids, qubits, params, mask=None):
        return self.net(torch.cat((state, flat_circuits(gate_ids, qubits, params)), 1))


class CircuitTransformer(nn.Module):
    def __init__(self, d_model: int = 48, heads: int = 4, layers: int = 2, state_hidden: int = 64):
        super().__init__()
        self.gate_embedding = nn.Embedding(len(GATES) + 1, d_model, padding_idx=0)
        self.qubit_embedding = nn.Embedding(N_QUBITS + 1, d_model, padding_idx=N_QUBITS)
        self.param_embedding = nn.Linear(3, d_model, bias=False)
        self.position_embedding = nn.Embedding(8, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, d_model * 2, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.state_encoder = nn.Sequential(nn.Linear(32, state_hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(d_model + state_hidden, 128), nn.ReLU(), nn.Linear(128, 32))

    def encode_context(self, gate_ids, qubits, params, mask):
        pos = torch.arange(gate_ids.shape[1], device=gate_ids.device)
        tokens = self.gate_embedding(gate_ids) + self.qubit_embedding(qubits[..., 0]) + self.qubit_embedding(qubits[..., 1]) + self.param_embedding(params) + self.position_embedding(pos)[None]
        encoded = self.encoder(tokens, src_key_padding_mask=~mask)
        return (encoded * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)

    def forward_cached(self, state, context):
        if context.shape[0] == 1 and state.shape[0] != 1:
            context = context.expand(state.shape[0], -1)
        return self.head(torch.cat((self.state_encoder(state), context), 1))

    def forward(self, state, gate_ids, qubits, params, mask):
        return self.forward_cached(state, self.encode_context(gate_ids, qubits, params, mask))


def normalize_real(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def fidelity_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = normalize_real(pred)
    target = normalize_real(target)
    pr, pi, tr, ti = pred[:, :DIM], pred[:, DIM:], target[:, :DIM], target[:, DIM:]
    real = (pr * tr + pi * ti).sum(1)
    imag = (pr * ti - pi * tr).sum(1)
    return real.square() + imag.square()


def fidelity_np(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, np.complex128)
    target = np.asarray(target, np.complex128)
    pred = pred / np.linalg.norm(pred, axis=-1, keepdims=True)
    target = target / np.linalg.norm(target, axis=-1, keepdims=True)
    return np.abs(np.sum(np.conj(target) * pred, axis=-1)) ** 2


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def make_model(name: str) -> nn.Module:
    return {"state_only": StateOnly, "flat_mlp": FlatMLP, "transformer": CircuitTransformer}[name]()


def tensorize(rows: list[dict[str, Any]], inputs: np.ndarray, targets: np.ndarray, indices: list[int]) -> tuple[torch.Tensor, ...]:
    selected = [rows[i] for i in indices]
    gate_ids, qubits, params, mask = encode_circuits(selected)
    return (torch.from_numpy(complex_to_real(inputs[indices]).astype(np.float32)), torch.from_numpy(gate_ids), torch.from_numpy(qubits),
            torch.from_numpy(params), torch.from_numpy(mask), torch.from_numpy(complex_to_real(targets[indices]).astype(np.float32)))


def train_model(name: str, seed: int, data: tuple[torch.Tensor, ...], validation: tuple[torch.Tensor, ...], config: dict[str, Any]) -> tuple[nn.Module, dict[str, Any]]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = make_model(name)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    batch_size, epochs = config["batch_size"], config["epochs"]
    history = []
    for epoch in range(epochs):
        model.train()
        generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        losses = []
        for idx in torch.randperm(len(data[0]), generator=generator).split(batch_size):
            batch = tuple(x[idx] for x in data)
            optimizer.zero_grad()
            loss = (1 - fidelity_torch(model(*batch[:-1]), batch[-1])).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            val_loss = float((1 - fidelity_torch(model(*validation[:-1]), validation[-1])).mean())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_loss": val_loss})
    return model, {"history": history, "stopping_rule": f"fixed {epochs} epochs; no test-based selection"}


def predict(model: nn.Module, data: tuple[torch.Tensor, ...], batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    model.eval(); outputs = []
    with torch.inference_mode():
        for start in range(0, len(data[0]), batch_size):
            outputs.append(model(*(x[start:start + batch_size] for x in data[:-1])).numpy())
    raw_real = np.concatenate(outputs)
    raw = real_to_complex(raw_real.astype(np.float64))
    norms = np.linalg.norm(raw, axis=1)
    return raw, raw / np.maximum(norms[:, None], 1e-12)


def observable_errors(pred: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    def expected(state: np.ndarray, gates: list[Gate]) -> np.ndarray:
        return np.real(np.sum(np.conj(state) * np.asarray([simulate(gates, x) for x in state]), axis=1))
    result = {}
    for q in range(N_QUBITS):
        gates = [Gate("X", (q,))]
        result[f"X_{q}"] = np.abs(expected(target, gates) - expected(pred, gates))
        # Z expectation from basis probabilities avoids adding Z to training alphabet.
        signs = np.array([1 if not (i & (1 << (N_QUBITS - 1 - q))) else -1 for i in range(DIM)])
        result[f"Z_{q}"] = np.abs(np.sum(np.abs(target) ** 2 * signs, 1) - np.sum(np.abs(pred) ** 2 * signs, 1))
    for q in range(N_QUBITS):
        for r in range(q + 1, N_QUBITS):
            signs = np.array([(1 if not (i & (1 << (N_QUBITS - 1 - q))) else -1) * (1 if not (i & (1 << (N_QUBITS - 1 - r))) else -1) for i in range(DIM)])
            result[f"Z_{q}Z_{r}"] = np.abs(np.sum(np.abs(target) ** 2 * signs, 1) - np.sum(np.abs(pred) ** 2 * signs, 1))
    return result


def stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, float)
    return {"mean": float(values.mean()), "median": float(np.median(values)), "std": float(values.std()),
            "P05": float(np.quantile(values, .05)), "P01": float(np.quantile(values, .01)), "minimum": float(values.min()),
            "P95": float(np.quantile(values, .95)), "P99": float(np.quantile(values, .99)), "maximum": float(values.max())}


def evaluate_model(model: nn.Module, rows: list[dict[str, Any]], inputs: np.ndarray, targets: np.ndarray, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    indices = [i for i, r in enumerate(rows) if r["split_name"] == split]
    data = tensorize(rows, inputs, targets, indices)
    raw, normalized = predict(model, data)
    exact = targets[indices]
    fidelities = fidelity_np(normalized, exact)
    obs = observable_errors(normalized, exact)
    examples = []
    for j, i in enumerate(indices):
        examples.append({"sample_id": rows[i]["sample_id"], "split": split, "state_family": rows[i]["state_family"], "depth": rows[i]["depth"],
                         "fidelity": float(fidelities[j]), "raw_norm": float(np.linalg.norm(raw[j])), "raw_norm_error": float(abs(np.linalg.norm(raw[j]) - 1)),
                         "observable_errors": {key: float(value[j]) for key, value in obs.items()}})
    summary = {"fidelity": stats(fidelities), "raw_norm_error": stats(np.abs(np.linalg.norm(raw, axis=1) - 1)),
               "observables": {key: stats(value) for key, value in obs.items()},
               "by_state_family": {family: stats(fidelities[[rows[i]["state_family"] == family for i in indices]]) for family in sorted({rows[i]["state_family"] for i in indices})},
               "by_depth": {str(depth): stats(fidelities[[rows[i]["depth"] == depth for i in indices]]) for depth in sorted({rows[i]["depth"] for i in indices})}}
    return examples, summary


def linearity_diagnostic(model: nn.Module, rows: list[dict[str, Any]], seed: int, count: int = 32) -> dict[str, Any]:
    """Fidelity between f(normalized(aψ1+bψ2)) and normalized(a f(ψ1)+b f(ψ2)); phase-invariant."""
    rng = np.random.default_rng(seed)
    candidates: list[tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]] = []
    heldout = [r for r in rows if r["split_name"] == "state_ood"]
    by_circuit: dict[str, list[dict[str, Any]]] = {}
    for row in heldout: by_circuit.setdefault(row["circuit_id"], []).append(row)
    for _ in range(count):
        group = list(by_circuit.values())[int(rng.integers(len(by_circuit)))]
        r1, r2 = rng.choice(group, 2, replace=False)
        psi1 = generate_state(r1["generator_seed"]["state"], r1["state_family"])
        psi2 = generate_state(r2["generator_seed"]["state"], r2["state_family"])
        a, b = rng.normal(size=2) + 1j * rng.normal(size=2)
        combined = a * psi1 + b * psi2
        if np.linalg.norm(combined) < 1e-10: continue
        candidates.append((r1, psi1, psi2, combined / np.linalg.norm(combined)))
    def infer(sample_rows, states):
        dummy_targets = np.asarray(states)
        data = tensorize(sample_rows, np.asarray(states), dummy_targets, list(range(len(states))))
        return predict(model, data)[1]
    scores = []
    for row, psi1, psi2, combined in candidates:
        a, b = rng.normal(size=2) + 1j * rng.normal(size=2)
        # Reconstruct a valid superposition with coefficients retained for RHS.
        combined = a * psi1 + b * psi2
        if np.linalg.norm(combined) < 1e-10: continue
        combined /= np.linalg.norm(combined)
        p1, p2, pc = infer([row, row, row], [psi1, psi2, combined])
        circuit = [Gate.from_dict(g) for g in row["gate_sequence_structured"]]
        # Fix each branch's otherwise arbitrary fidelity-loss gauge against its exact target before addition.
        exact1, exact2 = simulate(circuit, psi1), simulate(circuit, psi2)
        overlap1, overlap2 = np.vdot(exact1, p1), np.vdot(exact2, p2)
        if abs(overlap1) > 1e-12: p1 *= np.exp(-1j * np.angle(overlap1))
        if abs(overlap2) > 1e-12: p2 *= np.exp(-1j * np.angle(overlap2))
        rhs = a * p1 + b * p2
        if np.linalg.norm(rhs) < 1e-10: continue
        rhs /= np.linalg.norm(rhs)
        scores.append(float(fidelity_np(pc[None], rhs[None])[0]))
    return {"metric": "phase-invariant fidelity of normalized direct prediction vs normalized, branch-phase-aligned linear combination", "phase_alignment": "each branch prediction is aligned to its exact branch target before addition; final comparison remains phase invariant", "count": len(scores), "fidelity": stats(np.array(scores))}


def environment_info() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    cpu = platform.processor()
    if not cpu:
        try:
            cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
        except (OSError, StopIteration):
            cpu = "unknown"
    return {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "psutil": psutil.__version__, "cpu": cpu, "logical_cpus": os.cpu_count(), "ram_bytes": vm.total,
            "gpu": "none/driver unavailable", "device": "CPU", "torch_threads": torch.get_num_threads(),
            "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "native_threadpools": threadpool_info(), "dtype": "float32 neural; complex128 exact"}


def benchmark(model: CircuitTransformer, row: dict[str, Any], seed: int, repetitions: int = 50) -> dict[str, Any]:
    """Same-CPU benchmark; both methods may cache their fixed-circuit representation."""
    torch.set_num_threads(1)
    thread_controller = threadpool_limits(limits=1)  # retained until timings finish
    rng = np.random.default_rng(seed)
    states = np.asarray([(lambda z: z / np.linalg.norm(z))(rng.normal(size=DIM) + 1j * rng.normal(size=DIM)) for _ in range(10_000)])
    circuit = [Gate.from_dict(g) for g in row["gate_sequence_structured"]]
    model.eval()

    def neural_context():
        encoded = encode_circuits([row])
        tensors = tuple(torch.from_numpy(x) for x in encoded)
        return tensors, model.encode_context(*tensors)

    def neural_states(value):
        return torch.from_numpy(complex_to_real(value).astype(np.float32))

    def normalized_forward(value, context):
        return normalize_real(model.forward_cached(value, context))

    def neural_full_one():
        tensors, context = neural_context()
        value = neural_states(states[:1])
        return normalized_forward(value, context)

    # Exact fixed-context optimization is a precomputed unitary followed by BLAS matrix multiplication.
    def exact_context():
        return circuit_unitary(circuit)

    def measure(fn, reps):
        values = []
        for _ in range(reps):
            start = time.perf_counter_ns(); fn(); values.append((time.perf_counter_ns() - start) / 1e6)
        return values

    # Warm up each exact/neural path.
    for _ in range(10):
        simulate(circuit, states[0]); exact_context()
    with torch.inference_mode():
        for _ in range(10): neural_full_one()
        tensors, context = neural_context()
        state_t = neural_states(states)
        for _ in range(10): normalized_forward(state_t[:1], context)

    exact_single = measure(lambda: simulate(circuit, states[0]), repetitions)
    exact_context_ms = measure(exact_context, repetitions)
    unitary = exact_context()
    exact_cached_ms = measure(lambda: states[:1] @ unitary.T, repetitions)
    with torch.inference_mode():
        neural_full = measure(neural_full_one, repetitions)
        preprocess_ms = measure(lambda: encode_circuits([row]), repetitions)
        tensors = tuple(torch.from_numpy(x) for x in encode_circuits([row]))
        encode_ms = measure(lambda: model.encode_context(*tensors), repetitions)
        context = model.encode_context(*tensors)
        state_preprocess_ms = measure(lambda: neural_states(states[:1]), repetitions)
        cached_forward_ms = measure(lambda: model.forward_cached(state_t[:1], context), repetitions)
        normalization_ms = measure(lambda: normalize_real(model.forward_cached(state_t[:1], context)), repetitions)
        # Remove forward median to estimate the normalization-only component without asynchronous-device concerns (CPU only).
        normalization_only = max(0.0, float(np.median(normalization_ms)) - float(np.median(cached_forward_ms)))

    repeated, repeated_measurements = [], 5

    def exact_direct(n):
        for state in states[:n]:
            simulate(circuit, state)

    def repeated_neural(n):
        values_ms = []
        start = time.perf_counter(); encoded = encode_circuits([row]); values_ms.append((time.perf_counter() - start) * 1000)
        tensors_n = tuple(torch.from_numpy(x) for x in encoded)
        start = time.perf_counter(); context_n = model.encode_context(*tensors_n); values_ms.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter(); values = neural_states(states[:n]); values_ms.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        outputs = [model.forward_cached(values[offset:min(offset + 1024, n)], context_n) for offset in range(0, n, 1024)]
        values_ms.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter(); normalize_real(torch.cat(outputs)); values_ms.append((time.perf_counter() - start) * 1000)
        return values_ms

    for n in (1, 10, 100, 1000, 10000):
        # Warm the size-specific BLAS and neural kernels before robust repeated timing.
        states[:n] @ unitary.T
        with torch.inference_mode(): repeated_neural(n)
        direct_ms = float(np.median(measure(lambda n=n: exact_direct(n), repeated_measurements)))
        preparation_ms = float(np.median(measure(exact_context, repeated_measurements)))
        apply_ms = float(np.median(measure(lambda n=n: states[:n] @ unitary.T, repeated_measurements)))
        with torch.inference_mode():
            neural_components = np.median(np.asarray([repeated_neural(n) for _ in range(repeated_measurements)]), axis=0)
        circuit_preprocess, encoding, state_preprocess, infer, postprocess = map(float, neural_components)
        cached_exact_total = preparation_ms + apply_ms
        if direct_ms <= cached_exact_total:
            exact_total, exact_mode = direct_ms, "direct_gate_simulation"
        else:
            exact_total, exact_mode = cached_exact_total, "cached_unitary"
        neural_total = float(neural_components.sum())
        repeated.append({"N": n, "measurement_repetitions": repeated_measurements, "exact_direct_gate_ms": direct_ms,
                         "exact_context_preparation_ms": preparation_ms, "exact_cached_apply_ms": apply_ms,
                         "exact_cached_total_ms": cached_exact_total, "exact_best_mode": exact_mode,
                         "exact_total_ms": exact_total, "exact_ms_per_query": exact_total / n, "exact_states_per_sec": 1000 * n / exact_total,
                         "circuit_preprocessing_ms": circuit_preprocess, "context_encoding_ms": encoding,
                         "state_preprocessing_ms": state_preprocess, "cached_inference_ms": infer, "normalization_postprocessing_ms": postprocess,
                         "neural_total_ms": neural_total, "neural_ms_per_query": neural_total / n, "neural_states_per_sec": 1000 * n / neural_total})
    batch = []
    with torch.inference_mode():
        for n in (1, 10, 100, 1000):
            timings = measure(lambda n=n: normalized_forward(state_t[:n], context), 20)
            median = float(np.median(timings))
            batch.append({"batch_size": n, "median_ms": median, "states_per_sec": 1000 * n / median})
    advantageous = [x["N"] for x in repeated if x["neural_total_ms"] < x["exact_total_ms"]]
    break_even = advantageous[0] if advantageous else None
    sustained_break_even = next((x["N"] for i, x in enumerate(repeated) if all(y["neural_total_ms"] < y["exact_total_ms"] for y in repeated[i:])), None)
    return {"environment": environment_info(), "warmup_repetitions": 10, "timing_repetitions": repetitions,
            "single_query": {"exact_uncached_ms": stats(np.array(exact_single)), "exact_context_preparation_ms": stats(np.array(exact_context_ms)),
                             "exact_cached_ms": stats(np.array(exact_cached_ms)), "neural_full_query_ms": stats(np.array(neural_full)),
                             "circuit_preprocessing_ms": stats(np.array(preprocess_ms)), "context_encoding_ms": stats(np.array(encode_ms)),
                             "state_preprocessing_ms": stats(np.array(state_preprocess_ms)), "neural_cached_forward_ms": stats(np.array(cached_forward_ms)),
                             "normalization_postprocessing_median_ms": normalization_only},
            "repeated_context": repeated, "batch_throughput": batch, "neural_advantage_tested_N": advantageous,
            "break_even_N_observed": break_even, "sustained_break_even_N_observed": sustained_break_even,
            "timing_boundaries": "full neural query includes circuit preprocessing, context encoding, state conversion, forward, and normalization; cached forward excludes all but forward; repeated exact total is the faster measured direct-gate or unitary-cache path; neural total includes preprocessing, encoding, conversion, forward, and normalization",
            "exact_context_policy": "report both direct gate simulation and a precomputed 16x16 unitary with NumPy BLAS, selecting the faster measured exact path at each N; all native and Torch pools limited to one thread",
            "current_process_rss_bytes": psutil.Process().memory_info().rss, "peak_rusage_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "memory_note": "process-level high-water mark for the combined run; separate exact/neural CPU allocator peaks are not reliably attributable at this scale"}
