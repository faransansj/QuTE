"""CC-NQE P4.5 scaling, operator diagnostics, provenance, and XPU gates."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import resource
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import psutil
import torch
from torch import nn

from cc_nqe import (ACCEL, ACCEL_DEVICE, DIM, GATE_TO_ID, N_QUBITS, PARAM_REGIONS,
                    Gate, accel_device_name, accel_synchronize, circuit_id,
                    circuit_unitary, generate_circuit, generate_state, git_sha,
                    has_composition_motif, serialize_circuit, sha256, simulate,
                    structural_signature)

ROOT = Path("artifacts/cc_nqe_p4_5")
BASELINE = Path("artifacts/cc_nqe_p1_p4")
SCHEMA = "cc-nqe-p4.5-v1"
# Frozen before P4.5 implementation; prevents rewriting an artifact together with its manifest.
BASELINE_MANIFEST_SHA256 = "a55e2e7460a1b918c7b3dbb755ffa5dc5ac21430164ad4f3e37f3a4195a0e4cf"
SPLITS = ("validation", "iid", "state_ood", "parameter_interpolation",
          "parameter_extrapolation", "composition_ood", "depth_ood")
MODEL_SCALES = {
    "60k": {"width": 48, "ff": 96, "layers": 2, "heads": 4},
    "250k": {"width": 88, "ff": 176, "layers": 3, "heads": 4},
    "1m": {"width": 160, "ff": 320, "layers": 4, "heads": 8},
    "5m": {"width": 336, "ff": 672, "layers": 5, "heads": 8},
}
GRID = (("10k", "60k"), ("10k", "250k"), ("10k", "1m"),
        ("100k", "250k"), ("100k", "1m"), ("1m", "1m"), ("1m", "5m"))
CONFIRM = (("10k", "250k"), ("100k", "250k"), ("100k", "1m"), ("1m", "5m"))
TRAIN_COUNTS = {"10k": 10_000, "100k": 100_000, "1m": 1_000_000}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(canonical(value) + "\n")
        handle.flush()


def baseline_integrity() -> dict[str, Any]:
    manifest = BASELINE / "artifact_hashes.json"
    failures, entries = [], {}
    if manifest.exists():
        entries = json.loads(manifest.read_text())
        for name, expected in entries.items():
            path = Path(name)
            actual = sha256(path) if path.exists() else None
            if actual != expected:
                failures.append({"path": name, "expected": expected, "actual": actual})
    report = BASELINE / "REPORT.md"
    headings = [line[3:] for line in report.read_text().splitlines() if line.startswith("## ")] if report.exists() else []
    manifest_hash = sha256(manifest) if manifest.exists() else None
    if manifest_hash != BASELINE_MANIFEST_SHA256:
        failures.append({"path": str(manifest), "expected": BASELINE_MANIFEST_SHA256, "actual": manifest_hash})
    result = {
        "schema_version": SCHEMA, "gate": "G0", "status": "PASS" if manifest.exists() and report.exists() and not failures else "BASELINE-INTEGRITY-BLOCKED",
        "baseline_root": str(BASELINE), "hash_manifest": str(manifest), "hash_manifest_sha256": manifest_hash,
        "entries": len(entries), "verified": len(entries) - len(failures), "failures": failures,
        "report": str(report), "report_parsed": bool(headings), "report_headings": headings,
    }
    atomic_json(ROOT / "baseline_integrity.json", result)
    return result


def environment() -> dict[str, Any]:
    xpu = getattr(torch, "xpu", None)
    available = bool(xpu and xpu.is_available())
    supported = ["float32"]
    if available:
        supported += ["bfloat16", "float16"]
    info = {
        "schema_version": SCHEMA, "os": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
        "xpu_available": available, "xpu_device_count": xpu.device_count() if xpu else 0,
        "xpu_device_names": [xpu.get_device_name(i) for i in range(xpu.device_count())] if available else [],
        "pci_graphics_devices": _pci_graphics(), "xpu_build": getattr(torch.version, "xpu", None), "cpu": platform.processor() or _cpu_name(),
        "accel_kind": ACCEL, "accel_device": str(ACCEL_DEVICE), "accel_name": accel_device_name(),
        "ram_bytes": psutil.virtual_memory().total, "torch_threads": torch.get_num_threads(),
        "thread_environment": {k: os.getenv(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "supported_training_dtypes": supported, "teacher_dtype": "complex128", "training_reference_dtype": "float32",
    }
    atomic_json(ROOT / "environment.json", info)
    return info


def _pci_graphics() -> list[str]:
    try:
        return [line for line in subprocess.check_output(["lspci"], text=True).splitlines() if any(x in line for x in ("VGA", "Display", "3D"))]
    except (OSError, subprocess.SubprocessError):
        return []


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def state_fidelity(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = torch.complex(pred[..., :DIM], pred[..., DIM:])
    t = torch.complex(target[..., :DIM], target[..., DIM:])
    return (torch.abs((t.conj() * p).sum(-1)) ** 2 / (t.abs().square().sum(-1) * p.abs().square().sum(-1)).clamp_min(eps)).real


def operator_fidelity(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray) -> Any:
    if isinstance(pred, np.ndarray):
        overlap = np.sum(target.conj() * pred, axis=(-2, -1))
        denom = np.sum(abs(target) ** 2, axis=(-2, -1)) * np.sum(abs(pred) ** 2, axis=(-2, -1))
        return abs(overlap) ** 2 / np.maximum(denom, 1e-24)
    overlap = (target.conj() * pred).sum(dim=(-2, -1))
    denom = target.abs().square().sum(dim=(-2, -1)) * pred.abs().square().sum(dim=(-2, -1))
    return overlap.abs().square() / denom.clamp_min(1e-12)


def metric_summary(values: Iterable[float], bootstrap_seed: int = 0, resamples: int = 1000) -> dict[str, Any]:
    x=np.asarray(list(values),float)
    if not len(x): raise ValueError("metric summary requires values")
    rng=np.random.default_rng(bootstrap_seed); means=np.empty(resamples)
    for i in range(resamples): means[i]=rng.choice(x,len(x),replace=True).mean()
    return {"mean":float(x.mean()),"median":float(np.median(x)),"std":float(x.std()),"P05":float(np.percentile(x,5)),"P01":float(np.percentile(x,1)),"minimum":float(x.min()),"bootstrap_95_ci":[float(np.percentile(means,2.5)),float(np.percentile(means,97.5))],"count":len(x)}


def paired_bootstrap_difference(left: Iterable[float], right: Iterable[float], seed: int = 0, resamples: int = 1000) -> dict[str, Any]:
    a,b=np.asarray(list(left),float),np.asarray(list(right),float)
    if len(a)!=len(b) or not len(a): raise ValueError("paired fixed evaluation values required")
    d=a-b; rng=np.random.default_rng(seed); means=np.empty(resamples)
    for i in range(resamples): means[i]=rng.choice(d,len(d),replace=True).mean()
    return {"mean_difference":float(d.mean()),"bootstrap_95_ci":[float(np.percentile(means,2.5)),float(np.percentile(means,97.5))],"count":len(d)}


def phase_aligned_matrix_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    overlap = np.sum(target.conj() * pred, axis=(-2, -1))
    phase = np.where(abs(overlap) > 0, overlap.conj() / abs(overlap), 1.0)
    return np.linalg.norm(target - pred * phase[..., None, None], axis=(-2, -1)) / np.linalg.norm(target, axis=(-2, -1))


def unitarity_error(pred: torch.Tensor | np.ndarray) -> Any:
    if isinstance(pred, np.ndarray):
        ident = np.eye(DIM, dtype=pred.dtype)
        return np.linalg.norm(pred.conj().swapaxes(-1, -2) @ pred - ident, axis=(-2, -1)) / math.sqrt(DIM)
    ident = torch.eye(DIM, dtype=pred.dtype, device=pred.device)
    return torch.linalg.matrix_norm(pred.mH @ pred - ident) / math.sqrt(DIM)


def apply_operator(operator: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    psi = torch.complex(state[..., :DIM], state[..., DIM:])
    out = operator @ psi.unsqueeze(-1)
    out = out.squeeze(-1)
    return torch.cat((out.real, out.imag), -1)


def composition_fidelity(combined: torch.Tensor | np.ndarray, second: torch.Tensor | np.ndarray, first: torch.Tensor | np.ndarray) -> Any:
    return operator_fidelity(combined, second @ first)


class CircuitEncoder(nn.Module):
    def __init__(self, width: int, ff: int, layers: int, heads: int, max_depth: int = 16):
        super().__init__()
        self.gate = nn.Embedding(len(GATE_TO_ID) + 1, width, padding_idx=0)
        self.qubit = nn.Embedding(N_QUBITS + 1, width, padding_idx=N_QUBITS)
        self.parameter = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        self.position = nn.Embedding(max_depth, width)
        layer = nn.TransformerEncoderLayer(width, heads, ff, batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, layers, nn.LayerNorm(width))

    def forward(self, gates: torch.Tensor, qubits: torch.Tensor, parameters: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(gates.shape[1], device=gates.device)
        x = self.gate(gates) + self.qubit(qubits[..., 0]) + self.qubit(qubits[..., 1]) + self.parameter(parameters) + self.position(positions)
        x = self.encoder(x, src_key_padding_mask=~mask)
        return (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


class ScaledCCNQE(nn.Module):
    """One circuit encoder family; task only changes the diagnostic head."""
    def __init__(self, scale: str = "60k", task: str = "state"):
        super().__init__()
        if scale not in MODEL_SCALES or task not in ("state", "operator"):
            raise ValueError((scale, task))
        cfg = MODEL_SCALES[scale]
        self.scale, self.task = scale, task
        self.circuit = CircuitEncoder(**cfg)
        width = cfg["width"]
        if task == "state":
            self.state = nn.Sequential(nn.Linear(2 * DIM, width), nn.GELU(), nn.Linear(width, width))
            self.head = nn.Sequential(nn.Linear(2 * width, cfg["ff"]), nn.GELU(), nn.Linear(cfg["ff"], 2 * DIM))
        else:
            self.head = nn.Sequential(nn.Linear(width, cfg["ff"]), nn.GELU(), nn.Linear(cfg["ff"], 2 * DIM * DIM))

    def encode_context(self, gates: torch.Tensor, qubits: torch.Tensor, parameters: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.circuit(gates, qubits, parameters, mask)

    def forward_cached(self, context: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        if self.task == "operator":
            raw = self.head(context).reshape(-1, 2, DIM, DIM)
            return torch.complex(raw[:, 0], raw[:, 1])
        if state is None:
            raise ValueError("state task requires state")
        return self.head(torch.cat((context, self.state(state)), -1))

    def forward(self, gates: torch.Tensor, qubits: torch.Tensor, parameters: torch.Tensor, mask: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward_cached(self.encode_context(gates, qubits, parameters, mask), state)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def xpu_preflight() -> dict[str, Any]:
    env = environment()
    result: dict[str, Any] = {"schema_version": SCHEMA, "gate": "G1", "torch": torch.__version__, "xpu_available": env["xpu_available"],
                              "xpu_device_count": env["xpu_device_count"], "device": None, "checks": {}, "cpu_xpu_max_difference": None,
                              "tolerance": {"atol": 2e-4, "rtol": 2e-4}, "nan_inf": None}
    if ACCEL == "cpu":
        result.update(status="XPU-BLOCKED", reason="No native CUDA/XPU accelerator available; full scaling is forbidden. CPU fallback was not used.")
        atomic_json(ROOT / "xpu_preflight.json", result)
        return result
    try:
        device = ACCEL_DEVICE
        torch.manual_seed(73)
        cpu = ScaledCCNQE("60k", "state").eval()
        gates = torch.tensor([[1, 6, 0, 0]], dtype=torch.long)
        qubits = torch.full((1, 4, 2), N_QUBITS, dtype=torch.long); qubits[0, 0, 0] = 0; qubits[0, 1] = torch.tensor([0, 1])
        params = torch.zeros(1, 4, 3); mask = gates != 0; state = torch.randn(1, 2 * DIM); target = torch.randn(1, 2 * DIM)
        with torch.no_grad(): cpu_out = cpu(gates, qubits, params, mask, state)
        model = ScaledCCNQE("60k", "state"); model.load_state_dict(cpu.state_dict()); model.to(device)
        batch = tuple(x.to(device) for x in (gates, qubits, params, mask, state, target))
        opt = torch.optim.Adam(model.parameters(), 1e-4)
        result["checks"]["fp32_matmul"] = bool(torch.isfinite(torch.randn(8, 8, device=device) @ torch.randn(8, 8, device=device)).all().item())
        before = next(model.parameters()).detach().clone()
        out = model(*batch[:4], batch[4]); loss = (1 - state_fidelity(out, batch[5])).mean()
        opt.zero_grad(); loss.backward(); opt.step(); accel_synchronize()
        result["checks"].update(forward=tuple(out.shape) == (1, 2 * DIM), backward=all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),
                                optimizer_step=not torch.equal(before, next(model.parameters())), device_residency=all(x.device.type == ACCEL for x in (*batch, out, loss)))
        model.eval(); model.load_state_dict(cpu.state_dict())
        with torch.no_grad(): xpu_out = model(*batch[:4], batch[4]).cpu()
        difference = float((cpu_out - xpu_out).abs().max())
        result.update(device=accel_device_name(), model_device=str(next(model.parameters()).device), batch_device=str(batch[4].device),
                      output_device=str(out.device), loss_device=str(loss.device), cpu_xpu_max_difference=difference,
                      nan_inf=not bool(torch.isfinite(out).all() and torch.isfinite(loss)))
        result["checks"]["cpu_xpu_parity"] = torch.allclose(cpu_out, xpu_out, **result["tolerance"])
        result["status"] = "PASS" if all(result["checks"].values()) and not result["nan_inf"] else "XPU-BLOCKED"
    except Exception as exc:
        result.update(status="XPU-BLOCKED", reason=f"{type(exc).__name__}: {exc}")
    atomic_json(ROOT / "xpu_preflight.json", result)
    return result


class Progress:
    FIELDS = ("timestamp", "experiment_id", "phase", "task", "dataset_scale", "model_scale", "actual_parameters", "seed", "device", "dtype", "step", "maximum_steps", "samples_seen", "training_loss", "training_fidelity", "validation_fidelity", "composition_ood_fidelity", "depth_ood_fidelity", "learning_rate", "samples_per_second", "elapsed_seconds", "eta_seconds", "best_metric", "checkpoint", "state")
    def __init__(self, root: Path = ROOT, interval: float = 30.0, stream=None):
        self.root, self.interval, self.stream = root, interval, stream or sys.stdout
        self.last_print = 0.0

    def update(self, **values: Any) -> dict[str, Any]:
        row = {key: values.get(key) for key in self.FIELDS}
        row["timestamp"] = values.get("timestamp", time.time())
        append_jsonl(self.root / "progress.jsonl", row)
        atomic_json(self.root / "status.json", row)
        now = time.monotonic()
        if getattr(self.stream, "isatty", lambda: False)() or now - self.last_print >= self.interval:
            eta = "measuring..." if row["eta_seconds"] is None else f"{row['eta_seconds']:.0f}s (estimate)"
            line = f"[{row['phase']}] {row['experiment_id']} {row['step']}/{row['maximum_steps']} loss={row['training_loss']} fidelity={row['training_fidelity']} device={row['device']} ETA: {eta}"
            if getattr(self.stream, "isatty", lambda: False)():
                self.stream.write("\r\033[2K" + line)
            else:
                self.stream.write(line + "\n")
            self.stream.flush(); self.last_print = now
        return row


def status(root: Path = ROOT) -> dict[str, Any]:
    path = root / "status.json"
    latest = json.loads(path.read_text()) if path.exists() else {"state": "PENDING"}
    rows = [json.loads(x) for x in (root / "progress.jsonl").read_text().splitlines()] if (root / "progress.jsonl").exists() else []
    latest["completed_configurations"] = sorted({x["experiment_id"] for x in rows if x["state"] == "COMPLETED"})
    latest["failed_configurations"] = sorted({x["experiment_id"] for x in rows if x["state"] in ("FAILED", "BLOCKED")})
    latest["pending_grid"] = [f"{d}-{m}" for d, m in GRID if f"state-{d}-{m}" not in latest["completed_configurations"]]
    return latest


def _tensorize_circuit(circuit: list[Gate], max_depth: int = 16) -> tuple[np.ndarray, ...]:
    gates = np.zeros(max_depth, np.int64); qubits = np.full((max_depth, 2), N_QUBITS, np.int64); parameters = np.zeros((max_depth, 3), np.float32)
    for i, gate in enumerate(circuit):
        gates[i] = GATE_TO_ID[gate.name]; qubits[i, :len(gate.qubits)] = gate.qubits
        if gate.theta is not None: parameters[i] = (math.sin(gate.theta), math.cos(gate.theta), 1)
    return gates, qubits, parameters, gates != 0


def _save_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle: np.save(handle, value, allow_pickle=False)
    os.replace(tmp, path)


class GenerationProgress:
    def __init__(self, root: Path, stream=None):
        self.root, self.stream, self.started, self.last = root, stream or sys.stdout, time.monotonic(), 0.0
    def show(self, stage: str, current: int, total: int, split: str, shard: int | str = "-") -> None:
        now=time.monotonic(); tty=getattr(self.stream,"isatty",lambda:False)()
        if current not in (0,total) and now-self.last < (0.5 if tty else 30.0): return
        elapsed=now-self.started; rate=current/elapsed if current and elapsed else 0.; eta=(total-current)/rate if rate else None
        eta_text="measuring..." if eta is None else f"{eta:.0f}s (estimate)"
        line=f"stage={stage} shard={shard} samples={current}/{total} percent={100*current/max(total,1):.1f}% rate={rate:.1f}/s elapsed={elapsed:.1f}s ETA: {eta_text} split={split} workers=1 output={self.root}"
        self.stream.write(("\r\033[2K" if tty else "")+line+("" if tty else "\n")); self.stream.flush(); self.last=now


def _stats(values: Iterable[float]) -> dict[str, float]:
    x = np.asarray(list(values), float)
    return {"min": float(x.min()), "median": float(np.median(x)), "mean": float(x.mean()), "P95": float(np.percentile(x, 95)), "max": float(x.max())}


def _circuits(count: int, seed: int, depth_choices=(2, 4, 6), regime="train", motif=False, exclude_structures: set[str] | None = None) -> list[list[Gate]]:
    out, serials, structures = [], set(), set(exclude_structures or ())
    attempt = 0
    while len(out) < count:
        circuit = generate_circuit(seed + attempt, int(depth_choices[attempt % len(depth_choices)]), regime, motif); attempt += 1
        serial, structure = serialize_circuit(circuit), structural_signature(circuit)
        if serial in serials or structure in structures: continue
        serials.add(serial); structures.add(structure); out.append(circuit)
    return out


def generate_dataset(root: Path = ROOT / "datasets", master_samples: int = 1_000_000, eval_per_split: int = 10_000,
                     seed: int = 20260812, states_per_circuit: int = 16, shard_size: int = 50_000) -> dict[str, Any]:
    """Generate deduplicated, prefix-nested, memory-mappable CPU teacher data."""
    if master_samples < 1 or states_per_circuit < 2: raise ValueError("nonempty data and reusable contexts required")
    started = time.monotonic(); root.mkdir(parents=True, exist_ok=True); live=GenerationProgress(root)
    circuit_count = math.ceil(master_samples / states_per_circuit); live.show("circuits",0,circuit_count,"train")
    train_circuits = _circuits(circuit_count, seed); live.show("circuits",circuit_count,circuit_count,"train")
    train_structures = {structural_signature(c) for c in train_circuits}
    eval_circuit_count = max(1, math.ceil(eval_per_split / states_per_circuit))
    split_circuits: dict[str, list[list[Gate]]] = {"validation": _circuits(eval_circuit_count, seed + 2_000_000, exclude_structures=train_structures)}
    split_circuits["iid"] = split_circuits["validation"]
    split_circuits["state_ood"] = train_circuits[:eval_circuit_count]
    split_circuits["parameter_interpolation"] = _circuits(eval_circuit_count, seed + 3_000_000, regime="interpolation", exclude_structures=train_structures)
    split_circuits["parameter_extrapolation"] = _circuits(eval_circuit_count, seed + 4_000_000, regime="extrapolation", exclude_structures=train_structures)
    split_circuits["composition_ood"] = _circuits(eval_circuit_count, seed + 5_000_000, motif=True, exclude_structures=train_structures)
    split_circuits["depth_ood"] = _circuits(eval_circuit_count, seed + 6_000_000, depth_choices=(8,), exclude_structures=train_structures)
    all_circuits, circuit_split = list(train_circuits), ["train"] * len(train_circuits)
    offsets = {"train": (0, len(train_circuits))}
    for split in SPLITS:
        if split == "state_ood": offsets[split] = (0, eval_circuit_count); continue
        if split == "iid": offsets[split] = offsets["validation"]; continue
        start = len(all_circuits); all_circuits += split_circuits[split]; circuit_split += [split] * len(split_circuits[split]); offsets[split] = (start, len(all_circuits))
    circuit_rows=[]; gate_arrays=[]; qubit_arrays=[]; parameter_arrays=[]; masks=[]; unitaries=[]
    for i, circuit in enumerate(all_circuits):
        circuit_rows.append({"index": i, "circuit_id": circuit_id(circuit), "split": circuit_split[i], "depth": len(circuit), "serialized": serialize_circuit(circuit), "structural_signature": structural_signature(circuit)})
        g,q,p,m=_tensorize_circuit(circuit); gate_arrays.append(g); qubit_arrays.append(q); parameter_arrays.append(p); masks.append(m); unitaries.append(circuit_unitary(circuit))
    (root/"circuits.jsonl").write_text("".join(canonical(x)+"\n" for x in circuit_rows))
    _save_array(root/"circuit_gates.npy",np.asarray(gate_arrays)); _save_array(root/"circuit_qubits.npy",np.asarray(qubit_arrays)); _save_array(root/"circuit_parameters.npy",np.asarray(parameter_arrays)); _save_array(root/"circuit_masks.npy",np.asarray(masks)); _save_array(root/"unitaries.npy",np.asarray(unitaries,np.complex128))
    total_states = master_samples + len(SPLITS) * eval_per_split
    states=np.empty((total_states,DIM),np.complex128); state_rows=[]
    families=("product","random-local","entangled","Haar-random")
    live.show("states",0,total_states,"all")
    for i in range(total_states):
        family=families[i%4]; sseed=seed+20_000_000+i; states[i]=generate_state(sseed,family); state_rows.append({"index":i,"state_id":f"p45_s_{i}","family":family,"generator_seed":sseed})
        if (i+1)%1000==0 or i+1==total_states: live.show("states",i+1,total_states,"all")
    _save_array(root/"states.npy",states); (root/"states.jsonl").write_text("".join(canonical(x)+"\n" for x in state_rows))
    split_specs={"train":(master_samples,0)}
    for i,split in enumerate(SPLITS): split_specs[split]=(eval_per_split,master_samples+i*eval_per_split)
    shard_rows=[]; generated=0
    for split,(count,state_start) in split_specs.items():
        cstart,cend=offsets[split if split!="train" else "train"]
        for shard_start in range(0,count,shard_size):
            n=min(shard_size,count-shard_start); local=np.arange(shard_start,shard_start+n)
            ci=cstart+(local//states_per_circuit)%(cend-cstart); si=state_start+local
            target=np.einsum("nij,nj->ni",np.asarray(unitaries)[ci],states[si]).astype(np.complex128)
            stem=f"{split}_{shard_start//shard_size:05d}"; _save_array(root/f"pairs/{stem}.npy",np.column_stack((ci,si)).astype(np.int64)); _save_array(root/f"targets/{stem}.npy",target)
            shard_rows.append({"split":split,"pair_path":f"pairs/{stem}.npy","target_path":f"targets/{stem}.npy","count":n,"offset":shard_start}); generated+=n; live.show("targets",generated,master_samples+len(SPLITS)*eval_per_split,split,shard_start//shard_size)
    manifest={"schema_version":SCHEMA,"seed":seed,"n_qubits":4,"teacher":"cc_nqe.numpy_exact_statevector_v1","teacher_dtype":"complex128","master_samples":master_samples,"eval_per_split":eval_per_split,"states_per_circuit":states_per_circuit,"shard_size":shard_size,"unique_circuits":len(all_circuits),"train_unique_circuits":len(train_circuits),"unique_states":total_states,"circuit_offsets":offsets,"split_specs":split_specs,"shards":shard_rows,"generation_seconds":time.monotonic()-started,"samples_per_second":generated/max(time.monotonic()-started,1e-9)}
    atomic_json(root/"master_manifest.json",manifest)
    for scale,count in TRAIN_COUNTS.items():
        actual=min(count,master_samples); nested={"schema_version":SCHEMA,"scale":scale,"sample_count":actual,"pair_index_range":[0,actual],"master_manifest_sha256":sha256(root/"master_manifest.json")}; atomic_json(root/f"train_{scale}_manifest.json",nested)
    atomic_json(root/"evaluation_manifest.json",{"schema_version":SCHEMA,"frozen":True,"seed":seed,"split_counts":{s:eval_per_split for s in SPLITS},"master_manifest_sha256":sha256(root/"master_manifest.json")})
    return manifest


class CircuitDataset(torch.utils.data.Dataset):
    """Unique circuit/operator pairs; unitary targets are never duplicated."""
    def __init__(self, root: Path, split: str = "train", limit: int | None = None):
        self.root = root
        rows = [json.loads(x) for x in (root / "circuits.jsonl").read_text().splitlines()]
        table_split = "validation" if split == "iid" else ("train" if split == "state_ood" else split)
        self.indices = [x["index"] for x in rows if x["split"] == table_split][:limit]
        self.gates = np.load(root / "circuit_gates.npy", mmap_mode="r")
        self.qubits = np.load(root / "circuit_qubits.npy", mmap_mode="r")
        self.parameters = np.load(root / "circuit_parameters.npy", mmap_mode="r")
        self.masks = np.load(root / "circuit_masks.npy", mmap_mode="r")
        self.unitaries = np.load(root / "unitaries.npy", mmap_mode="r")
    def __len__(self): return len(self.indices)
    def __getitem__(self, index):
        i = self.indices[index]; unitary = self.unitaries[i]
        return self.gates[i], self.qubits[i], self.parameters[i], self.masks[i], np.stack((unitary.real, unitary.imag)).astype(np.float32)


class ShardedDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, split: str, limit: int | None = None):
        self.root=root; manifest=json.loads((root/"master_manifest.json").read_text()); self.shards=[]; total=0
        for row in manifest["shards"]:
            if row["split"]!=split: continue
            n=row["count"] if limit is None else min(row["count"],max(0,limit-total))
            if n: self.shards.append((total,total+n,row,n)); total+=n
            if limit is not None and total>=limit: break
        self.length=total; self.states=np.load(root/"states.npy",mmap_mode="r"); self.gates=np.load(root/"circuit_gates.npy",mmap_mode="r"); self.qubits=np.load(root/"circuit_qubits.npy",mmap_mode="r"); self.parameters=np.load(root/"circuit_parameters.npy",mmap_mode="r"); self.masks=np.load(root/"circuit_masks.npy",mmap_mode="r")
    def __len__(self): return self.length
    def __getitem__(self,index):
        if index<0: index+=self.length
        for start,end,row,n in self.shards:
            if start<=index<end:
                local=index-start; pair=np.load(self.root/row["pair_path"],mmap_mode="r")[local]; target=np.load(self.root/row["target_path"],mmap_mode="r")[local]; ci,si=map(int,pair)
                state=self.states[si]; return (self.gates[ci],self.qubits[ci],self.parameters[ci],self.masks[ci],np.r_[state.real,state.imag].astype(np.float32),np.r_[target.real,target.imag].astype(np.float32))
        raise IndexError(index)


def _classify_structural_duplicates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["structural_signature"], []).append(row)
    findings, counts = [], Counter()
    for signature, members in groups.items():
        if len(members) < 2:
            continue
        serialized = [row["serialized"] for row in members]
        parsed = [json.loads(value) for value in serialized]
        crosses_prohibited_boundary = any(row["split"] == "train" for row in members) and any(row["split"] not in ("train", "state_ood") for row in members)
        if crosses_prohibited_boundary:
            category = "D"
        elif len(set(serialized)) < len(serialized):
            category = "C"
        else:
            # The signature already proves identical ordered gate/qubit topology;
            # differing serialization can therefore only be continuous parameters.
            category = "A"
        counts[category] += len(members) - 1
        findings.append({
            "category": category,
            "reason": {"A": "same ordered gate/qubit topology with different continuous parameters", "C": "identical serialized circuit", "D": "topology crosses a prohibited train/OOD boundary"}[category],
            "structural_signature": signature,
            "members": [{"circuit_id": row["circuit_id"], "split": row["split"], "depth": row["depth"], "gate_sequence": [g["name"] for g in gates], "qubits": [g["qubits"] for g in gates], "continuous_parameters": [g.get("theta") for g in gates], "exact_circuit_signature": row["serialized"]} for row, gates in zip(members, parsed)],
        })
    return {"definition": "ordered gate type and ordered target/control qubits; parameters excluded; depth implicit in sequence length", "group_count": len(findings), "duplicate_excess_count": sum(counts.values()), "classification_counts": {key: counts.get(key, 0) for key in "ABCD"}, "groups": findings}


def audit_dataset(root: Path = ROOT/"datasets") -> dict[str,Any]:
    manifest=json.loads((root/"master_manifest.json").read_text()); rows=[json.loads(x) for x in (root/"circuits.jsonl").read_text().splitlines()]; state_rows=[json.loads(x) for x in (root/"states.jsonl").read_text().splitlines()]; states=np.load(root/"states.npy",mmap_mode="r"); unitaries=np.load(root/"unitaries.npy",mmap_mode="r")
    train=[x for x in rows if x["split"]=="train"]; train_ids={x["circuit_id"] for x in train}; train_sig={x["structural_signature"] for x in train}
    bysplit={s:[x for x in rows if x["split"]==s] for s in SPLITS}; pair_hashes=[]; sample_count=0; counts=Counter(); split_states={}; split_circuits={}; circuits_by_family={f:set() for f in ("product","random-local","entangled","Haar-random")}
    max_target_norm=0.0
    for shard in manifest["shards"]:
        pairs=np.load(root/shard["pair_path"],mmap_mode="r"); targets=np.load(root/shard["target_path"],mmap_mode="r"); sample_count+=len(pairs); counts.update(map(int,pairs[:,0])); pair_hashes += [hashlib.sha256(x.tobytes()).digest() for x in pairs]
        split_states.setdefault(shard["split"],set()).update(map(int,pairs[:,1])); split_circuits.setdefault(shard["split"],set()).update(map(int,pairs[:,0]))
        for ci,si in pairs: circuits_by_family[state_rows[int(si)]["family"]].add(int(ci))
        max_target_norm=max(max_target_norm,float(np.max(abs(np.linalg.norm(targets,axis=1)-1))))
    structural_audit = _classify_structural_duplicates(rows)
    exact_circuit_signatures = [x["serialized"] for x in rows]
    checks={
      "sample_count":sample_count==manifest["master_samples"]+len(SPLITS)*manifest["eval_per_split"],
      "nested_subsets":all(json.loads((root/f"train_{s}_manifest.json").read_text())["sample_count"]==min(n,manifest["master_samples"]) for s,n in TRAIN_COUNTS.items()),
      "state_normalization":float(np.max(abs(np.linalg.norm(states,axis=1)-1)))<1e-12,"target_normalization":max_target_norm<1e-12,
      "unitarity":float(np.max(np.linalg.norm(unitaries.conj().transpose(0,2,1)@unitaries-np.eye(DIM),axis=(1,2))))<1e-10,
      "exact_duplicates":len(pair_hashes)==len(set(pair_hashes)),
      "exact_circuit_duplicates":len(exact_circuit_signatures)==len(set(exact_circuit_signatures)),
      "state_id_leakage":all(not split_states.get("train",set()) & split_states.get(s,set()) for s in SPLITS),
      "circuit_id_leakage":all(not split_circuits.get("train",set()) & split_circuits.get(s,set()) for s in SPLITS if s != "state_ood"),
      "structural_leakage":all(not train_sig & {x["structural_signature"] for x in bysplit[s]} for s in SPLITS if s not in ("state_ood",)),
      "composition_holdout":all(has_composition_motif([Gate.from_dict(g) for g in json.loads(x["serialized"])]) for x in bysplit["composition_ood"]) and all(not has_composition_motif([Gate.from_dict(g) for g in json.loads(x["serialized"])]) for x in train),
      "depth_holdout":all(x["depth"]==8 for x in bysplit["depth_ood"]) and all(x["depth"]<=6 for x in train),
      "parameter_interpolation":_parameter_contract(bysplit["parameter_interpolation"],"interpolation"),"parameter_extrapolation":_parameter_contract(bysplit["parameter_extrapolation"],"extrapolation"),
      "training_excludes_parameter_holdouts":all(not any(low < g["theta"] < high for regions in (PARAM_REGIONS["interpolation"], PARAM_REGIONS["extrapolation"]) for low, high in regions) for row in train for g in json.loads(row["serialized"]) if g.get("theta") is not None),
      "fixed_evaluation":json.loads((root/"evaluation_manifest.json").read_text())["frozen"] is True,
      "deterministic_regeneration":serialize_circuit(_circuits(1,manifest["seed"])[0])==rows[0]["serialized"] and np.array_equal(generate_state(state_rows[0]["generator_seed"],state_rows[0]["family"]),states[0]),
    }
    all_gates=[g for row in rows for g in json.loads(row["serialized"])]; parameters={s:[g["theta"] for row in bysplit[s] for g in json.loads(row["serialized"]) if g["theta"] is not None] for s in SPLITS}
    multiplicity=_stats(counts.values()); result={"schema_version":SCHEMA,"gate":"G2","status":"PASS" if all(checks.values()) else "DATASET-BLOCKED","checks":checks,"sample_count":sample_count,"unique_circuit_count":len(rows),"unique_state_count":len(states),"states_per_circuit":multiplicity,"circuits_per_state_family":{k:len(v) for k,v in circuits_by_family.items()},"max_input_norm_error":float(np.max(abs(np.linalg.norm(states,axis=1)-1))),"max_target_norm_error":max_target_norm,"max_unitarity_error":float(np.max(unitarity_error(unitaries))),"exact_duplicate_count":len(pair_hashes)-len(set(pair_hashes)),"exact_circuit_duplicate_count":len(exact_circuit_signatures)-len(set(exact_circuit_signatures)),"structural_duplicate_count":structural_audit["duplicate_excess_count"],"structural_duplicates_informational":True,"structural_duplicate_classification":structural_audit["classification_counts"],"state_family_distribution":dict(Counter(x["family"] for x in state_rows)),"depth_distribution":dict(Counter(str(x["depth"]) for x in rows)),"gate_distribution":dict(Counter(g["name"] for g in all_gates)),"parameter_distribution":{s:{"count":len(v),"minimum":min(v) if v else None,"maximum":max(v) if v else None} for s,v in parameters.items()},"deterministic_regeneration":checks["deterministic_regeneration"]}
    atomic_json(root/"audit.json",result)
    atomic_json(root/"structural_duplicate_audit.json", structural_audit)
    hashes={str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name!="hashes.sha256"}
    (root/"hashes.sha256").write_text("".join(f"{value}  {name}\n" for name,value in hashes.items()))
    return result


def _parameter_contract(rows:list[dict],regime:str)->bool:
    regions=PARAM_REGIONS[regime]
    values=[g.get("theta") for row in rows for g in json.loads(row["serialized"]) if g.get("theta") is not None]
    return bool(values) and all(any(low<x<high for low,high in regions) for x in values)


def config_hash(config:dict[str,Any])->str: return digest(config)


def save_checkpoint(path:Path,model:nn.Module,optimizer:torch.optim.Optimizer,scheduler:Any,config:dict[str,Any],dataset_manifest_hash:str,step:int,samples_seen:int,best_metric:float,scaler:Any=None)->None:
    path.parent.mkdir(parents=True,exist_ok=True); payload={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict() if scheduler else None,"scaler":scaler.state_dict() if scaler else None,"step":step,"samples_seen":samples_seen,"best_metric":best_metric,"rng":{"torch":torch.get_rng_state(),"numpy":np.random.get_state(),"python":random.getstate()},"config":config,"config_hash":config_hash(config),"dataset_manifest_hash":dataset_manifest_hash}; tmp=path.with_name(path.name+".tmp"); torch.save(payload,tmp); os.replace(tmp,path)


def load_checkpoint(path:Path,model:nn.Module,optimizer:torch.optim.Optimizer,scheduler:Any,config:dict[str,Any],dataset_manifest_hash:str,scaler:Any=None)->dict[str,Any]:
    payload=torch.load(path,map_location="cpu",weights_only=False)
    if payload["config_hash"]!=config_hash(config) or payload["config"]!=config: raise ValueError("resume refused: config hash/definition differs")
    if payload["dataset_manifest_hash"]!=dataset_manifest_hash: raise ValueError("resume refused: dataset manifest differs")
    model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"])
    if scheduler and payload["scheduler"] is not None: scheduler.load_state_dict(payload["scheduler"])
    if scaler and payload["scaler"] is not None: scaler.load_state_dict(payload["scaler"])
    torch.set_rng_state(payload["rng"]["torch"]); np.random.set_state(payload["rng"]["numpy"]); random.setstate(payload["rng"]["python"])
    return payload


def artifact_hashes(root:Path=ROOT)->dict[str,str]:
    values={str(p):sha256(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name!="artifact_hashes.json"}; atomic_json(root/"artifact_hashes.json",values); return values
