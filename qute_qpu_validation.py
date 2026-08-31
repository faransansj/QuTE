"""Real-QPU validation for the frozen CC-NQE P1-P4 workload."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from cc_nqe import Gate, N_QUBITS, make_model, predict, tensorize

ROOT = Path("artifacts/cc_nqe_p1_p4")
DATASET_JSONL = ROOT / "dataset/samples.jsonl"
DATASET_NPZ = ROOT / "dataset/samples.npz"
CHECKPOINT = ROOT / "checkpoints/transformer_seed11.pt"
DEFAULT_ARTIFACT_ROOT = Path("artifacts/qute_qpu_validation")
SPLITS = {
    "IID": ("iid",),
    "Parameter-OOD": ("parameter_interpolation", "parameter_extrapolation"),
    "Composition-OOD": ("composition_ood",),
    "Depth-OOD": ("depth_ood",),
}
SCHEMA = "qute-qpu-validation-v1"
SEED_TRANSPILER = 20260811
BIT_ORDER = "benchmark q0..q3 = displayed bitstring MSB..LSB; benchmark q maps to Qiskit qubit 3-q"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def distribution(state: np.ndarray) -> dict[str, float]:
    state = np.asarray(state, np.complex128)
    norm = np.linalg.norm(state)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("statevector must have a finite nonzero norm")
    probabilities = np.abs(state / norm) ** 2
    return {format(i, f"0{N_QUBITS}b"): float(p) for i, p in enumerate(probabilities)}


def normalize_counts(counts: dict[str, int], shots: int, n_qubits: int = N_QUBITS) -> dict[str, float]:
    if shots <= 0 or sum(counts.values()) != shots:
        raise ValueError(f"counts total {sum(counts.values())} does not match shots {shots}")
    out = {format(i, f"0{n_qubits}b"): 0.0 for i in range(1 << n_qubits)}
    for raw, count in counts.items():
        key = raw.replace(" ", "").zfill(n_qubits)
        if len(key) != n_qubits or set(key) - {"0", "1"} or count < 0:
            raise ValueError(f"invalid count entry: {raw!r}={count}")
        out[key] += count / shots
    return out


def tvd(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    if abs(sum(left.values()) - 1) > 1e-9 or abs(sum(right.values()) - 1) > 1e-9:
        raise ValueError("TVD inputs must be normalized")
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def load_assets() -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    for path in (DATASET_JSONL, DATASET_NPZ, CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(f"frozen input missing: {path}")
    rows = [json.loads(line) for line in DATASET_JSONL.read_text().splitlines()]
    arrays = np.load(DATASET_NPZ)
    inputs, targets = arrays["input_state"], arrays["target_state"]
    if len(rows) != len(inputs) or len(rows) != len(targets):
        raise ValueError("frozen dataset row/array length mismatch")
    return rows, inputs, targets


def select_samples(rows: list[dict[str, Any]], num_per_split: int) -> list[dict[str, Any]]:
    if num_per_split < 1:
        raise ValueError("num_per_split must be positive")
    selected = []
    for label, source_splits in SPLITS.items():
        candidates = [row for row in rows if row["split_name"] in source_splits and row["state_family"] in {"product", "random-local"}]
        by_circuit: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            by_circuit[row["circuit_id"]].append(row)
        choices = [min(by_circuit[circuit_id], key=lambda row: row["sample_id"]) for circuit_id in sorted(by_circuit)]
        if len(choices) < num_per_split:
            raise ValueError(f"{label} has only {len(choices)} eligible unique circuits")
        for row in choices[:num_per_split]:
            selected.append({**row, "validation_split": label, "selection_rank": len([x for x in selected if x["validation_split"] == label])})
    return selected


def load_predictions(selected: list[dict[str, Any]], rows: list[dict[str, Any]], inputs: np.ndarray, targets: np.ndarray) -> list[np.ndarray]:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint.get("model") != "transformer" or checkpoint.get("seed") != 11:
        raise ValueError("unexpected frozen checkpoint identity")
    model = make_model("transformer")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    indices = [int(row["array_index"]) for row in selected]
    _, normalized = predict(model, tensorize(rows, inputs, targets, indices))
    return list(normalized)


def product_factors(state: np.ndarray) -> list[np.ndarray]:
    """Return q0..q3 factors for a product state, rejecting entangled inputs."""
    tensor = np.asarray(state, np.complex128).reshape((2,) * N_QUBITS)
    factors = []
    for axis in range(N_QUBITS):
        moved = np.moveaxis(tensor, axis, 0).reshape(2, -1)
        density = moved @ moved.conj().T
        values, vectors = np.linalg.eigh(density)
        factor = vectors[:, int(np.argmax(values))]
        factors.append(factor / np.linalg.norm(factor))
    rebuilt = factors[0]
    for factor in factors[1:]:
        rebuilt = np.kron(rebuilt, factor)
    overlap = np.vdot(rebuilt, np.asarray(state))
    if abs(overlap) < 1 - 1e-8:
        raise ValueError("input is not a product state; arbitrary preparation is outside smoke scope")
    return factors


def qiskit_modules():
    try:
        from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
        from qiskit.quantum_info import Statevector
        from qiskit.transpiler import generate_preset_pass_manager
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    except ImportError as error:
        raise RuntimeError("Qiskit dependencies missing; install the project qpu extra") from error
    return ClassicalRegister, QuantumRegister, QuantumCircuit, Statevector, generate_preset_pass_manager, QiskitRuntimeService, SamplerV2


def build_workload(row: dict[str, Any], input_state: np.ndarray):
    ClassicalRegister, QuantumRegister, QuantumCircuit, *_ = qiskit_modules()
    circuit = QuantumCircuit(QuantumRegister(N_QUBITS, "q"), ClassicalRegister(N_QUBITS, "meas"), name=row["sample_id"])
    for benchmark_q, factor in enumerate(product_factors(input_state)):
        circuit.prepare_state(factor, [N_QUBITS - 1 - benchmark_q])
    for value in row["gate_sequence_structured"]:
        gate = Gate.from_dict(value)
        q = [N_QUBITS - 1 - item for item in gate.qubits]
        if gate.name == "H": circuit.h(q[0])
        elif gate.name == "X": circuit.x(q[0])
        elif gate.name == "RX": circuit.rx(float(gate.theta), q[0])
        elif gate.name == "RY": circuit.ry(float(gate.theta), q[0])
        elif gate.name == "RZ": circuit.rz(float(gate.theta), q[0])
        elif gate.name == "CNOT": circuit.cx(q[0], q[1])
        else: raise ValueError(f"unsupported gate: {gate.name}")
    for q in range(N_QUBITS):
        circuit.measure(q, q)
    return circuit


def statevector_distribution(circuit) -> dict[str, float]:
    _, _, _, Statevector, *_ = qiskit_modules()
    bare = circuit.remove_final_measurements(inplace=False)
    return distribution(np.asarray(Statevector.from_instruction(bare).data))


def compact_transpiled_distribution(circuit) -> dict[str, float]:
    """Exactly simulate only physical wires touched by a transpiled circuit."""
    _, _, QuantumCircuit, Statevector, *_ = qiskit_modules()
    measurements: dict[int, int] = {}
    active = set()
    for instruction in circuit.data:
        qids = [circuit.find_bit(q).index for q in instruction.qubits]
        if instruction.operation.name == "measure":
            measurements[circuit.find_bit(instruction.clbits[0]).index] = qids[0]
        else:
            active.update(qids)
    if len(active) > 16:
        raise RuntimeError(f"transpiled ideal simulation needs {len(active)} active qubits; limit is 16")
    ordered, remap = sorted(active), {old: new for new, old in enumerate(sorted(active))}
    compact = QuantumCircuit(len(ordered))
    for instruction in circuit.data:
        if instruction.operation.name in {"measure", "barrier", "delay"}:
            continue
        qids = [remap[circuit.find_bit(q).index] for q in instruction.qubits]
        compact.append(instruction.operation, qids)
    probabilities = Statevector.from_instruction(compact).probabilities()
    out = {format(i, f"0{N_QUBITS}b"): 0.0 for i in range(1 << N_QUBITS)}
    for basis, probability in enumerate(probabilities):
        classical = [0] * N_QUBITS
        for cbit, physical in measurements.items():
            if physical in remap:
                classical[cbit] = (basis >> remap[physical]) & 1
        key = "".join(str(classical[i]) for i in reversed(range(N_QUBITS)))
        out[key] += float(probability)
    return out


def validate_bit_order() -> dict[str, Any]:
    ClassicalRegister, QuantumRegister, QuantumCircuit, Statevector, *_ = qiskit_modules()
    circuit = QuantumCircuit(QuantumRegister(N_QUBITS, "q"), ClassicalRegister(N_QUBITS, "meas"))
    circuit.x(N_QUBITS - 1)  # benchmark q0
    for q in range(N_QUBITS): circuit.measure(q, q)
    observed = statevector_distribution(circuit)
    passed = observed["1000"] > 1 - 1e-12
    return {"status": "PASS" if passed else "FAIL", "convention": BIT_ORDER, "basis_probe": "benchmark X(q0) -> 1000", "observed": observed}


def backend_metadata(backend) -> dict[str, Any]:
    status = backend.status() if hasattr(backend, "status") else None
    simulator = bool(getattr(backend, "simulator", False))
    operational = bool(getattr(status, "operational", True))
    if simulator:
        raise ValueError("selected backend is a simulator")
    if not operational:
        raise ValueError(f"backend is not operational: {getattr(status, 'status_msg', '')}")
    if int(backend.num_qubits) < N_QUBITS:
        raise ValueError(f"backend has {backend.num_qubits} qubits; {N_QUBITS} required")
    target = backend.target
    return {
        "backend_name": backend.name,
        "backend_version": str(getattr(backend, "backend_version", "unknown")),
        "num_qubits": int(backend.num_qubits),
        "simulator": simulator,
        "operational": operational,
        "status_message": getattr(status, "status_msg", None),
        "pending_jobs": getattr(status, "pending_jobs", None),
        "basis_gates": sorted(getattr(backend, "operation_names", [])),
        "target_summary": {name: len(target[name]) if target[name] is not None else 0 for name in sorted(target.operation_names)},
        "timestamp": utc_now(),
    }


def connect_backend(name: str, instance: str | None = None):
    *_, QiskitRuntimeService, _ = qiskit_modules()
    service = QiskitRuntimeService(instance=instance) if instance else QiskitRuntimeService()
    available = service.backends(name=name, min_num_qubits=N_QUBITS)
    if not available:
        raise ValueError(f"backend unavailable to this account: {name}")
    return service, available[0]


def layout_value(circuit) -> Any:
    layout = getattr(circuit, "layout", None)
    if layout is None: return None
    try: return list(layout.final_index_layout(filter_ancillas=True))
    except Exception: return str(layout)


def two_qubit_count(circuit) -> int:
    return sum(1 for item in circuit.data if len(item.qubits) == 2)


def circuit_metrics(row: dict[str, Any], original, transpiled) -> dict[str, Any]:
    original_bare = original.remove_final_measurements(inplace=False)
    transpiled_bare = transpiled.remove_final_measurements(inplace=False)
    counts = {str(k): int(v) for k, v in transpiled_bare.count_ops().items()}
    return {
        "sample_id": row["sample_id"], "circuit_id": row["circuit_id"], "split": row["validation_split"],
        "benchmark_depth": int(row["depth"]), "original_depth": int(original_bare.depth()),
        "original_2q_gate_count": two_qubit_count(original_bare), "original_total_gate_count": int(original_bare.size()),
        "transpiled_depth": int(transpiled_bare.depth()), "transpiled_2q_gate_count": two_qubit_count(transpiled_bare),
        "transpiled_total_gate_count": int(transpiled_bare.size()), "transpiled_gate_counts": counts,
        "native_2q_gate_counts": {name: count for name, count in counts.items() if any(item.operation.name == name and len(item.qubits) == 2 for item in transpiled_bare.data)},
        "physical_layout": layout_value(transpiled),
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
        out = {"count": len(rows)}
        for key in ("tvd_model_sim", "tvd_sim_qpu", "tvd_model_qpu", "tvd_canonical_transpiled"):
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            out[key] = {"mean": statistics.fmean(values), "median": statistics.median(values)} if values else None
        return out
    by_split = {split: block([row for row in results if row["split"] == split]) for split in SPLITS}
    overall = block(results)
    hardware_rows = [row for row in results if row.get("tvd_sim_qpu") is not None]
    def correlation(key: str) -> float | None:
        if len(hardware_rows) < 2: return None
        x = np.asarray([row[key] for row in hardware_rows], float)
        y = np.asarray([row["tvd_sim_qpu"] for row in hardware_rows], float)
        if np.std(x) == 0 or np.std(y) == 0: return None
        return float(np.corrcoef(x, y)[0, 1])
    return {"schema_version": SCHEMA, "by_split": by_split, "overall": overall,
            "complexity_correlations": {"transpiled_depth_vs_tvd_sim_qpu": correlation("transpiled_depth"),
                                         "transpiled_2q_gate_count_vs_tvd_sim_qpu": correlation("transpiled_2q_gate_count")},
            "interpretation": {"model_error": "CC-NQE ↔ canonical ideal simulator", "hardware_error": "canonical ideal simulator ↔ QPU", "combined_gap": "CC-NQE ↔ QPU"},
            "claim_boundary": "Descriptive smoke evidence only; the ideal-trained model did not learn QPU noise."}


def report(config: dict[str, Any], summary: dict[str, Any], metrics: list[dict[str, Any]], status: str, blocker: str | None = None) -> str:
    rows = []
    for split, values in summary["by_split"].items():
        def show(key):
            item = values[key]
            return "N/A" if item is None else f"{item['mean']:.6f} / {item['median']:.6f}"
        rows.append(f"| {split} | {show('tvd_model_sim')} | {show('tvd_sim_qpu')} | {show('tvd_model_qpu')} |")
    return f"""# QuTE / CC-NQE QPU Validation\n\n## Scope\n\nThis validation layer does not modify or retrain the frozen CC-NQE model, benchmark, splits, seeds, or prior results.\n\n## Research Question\n\n> How closely do QuTE / CC-NQE predictions and ideal simulator outputs reproduce measurement statistics observed on a real IBM QPU?\n\n## Experimental Setup\n\n- Model/checkpoint: `{config['checkpoint']}` (CC-NQE circuit transformer, frozen seed 11)\n- Benchmark source: `{config['dataset']}`\n- Selected splits: {', '.join(SPLITS)}\n- Circuits: {config['num_circuits']}\n- Backend: `{config['backend']}`\n- Shots: {config['shots']}\n- Transpilation: preset pass manager level {config['optimization_level']}, seed {config['seed_transpiler']}\n- Bit order: {BIT_ORDER}\n- Status: **{status}**{f' — {blocker}' if blocker else ''}\n\n## Results\n\nValues are mean / median TVD.\n\n| Split | QuTE-Sim | Sim-QPU | QuTE-QPU |\n|---|---:|---:|---:|\n{chr(10).join(rows)}\n\n## Hardware Complexity\n\nPer-circuit transpiled depth, 2Q gate count, native gate counts, and physical layout are in `circuit_metrics.json`. Complexity correlations are descriptive and stored in `summary.json`.\n\n## Interpretation\n\n- QuTE ↔ Simulator discrepancy = model approximation / generalization error.\n- Simulator ↔ QPU discrepancy = hardware execution gap.\n- QuTE ↔ QPU discrepancy = combined model + hardware gap.\n\nA small QuTE-QPU distance does not imply that QuTE learned QPU noise; this model was trained against ideal simulation. The smoke sample is too small for statistical claims.\n\n## Phase 2 Candidate\n\nOnly after Phase 1 succeeds: evaluate single-qubit X/Y/Z and selected two-qubit ZZ observables. No basis expansion is submitted by this implementation.\n"""


def checksums(root: Path) -> None:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    (root / "checksums.sha256").write_text("".join(f"{sha256(path)}  {path.relative_to(root)}\n" for path in files))


def preflight_matches(path: Path, config: dict[str, Any]) -> None:
    prior = json.loads((path / "config.json").read_text())
    if prior.get("status") != "QPU_SUBMISSION_READY":
        raise ValueError("preflight did not reach QPU_SUBMISSION_READY")
    for key in ("backend", "shots", "num_per_split", "selection_sha256", "checkpoint_sha256", "optimization_level", "seed_transpiler"):
        if prior.get(key) != config.get(key): raise ValueError(f"preflight mismatch: {key}")


def run(args: argparse.Namespace, backend=None, service=None) -> Path:
    rows, inputs, targets = load_assets()
    selected = select_samples(rows, args.num_per_split)
    selected_public = [{key: row[key] for key in ("sample_id", "circuit_id", "validation_split", "split_name", "state_id", "state_family", "depth", "array_index", "selection_rank", "gate_sequence_structured")} for row in selected]
    selection_bytes = json.dumps(selected_public, sort_keys=True, separators=(",", ":")).encode()
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + args.backend.replace(":", "_")
    resume_dir = getattr(args, "resume_dir", None)
    output = Path(resume_dir) if resume_dir else Path(args.output_dir) if args.output_dir else DEFAULT_ARTIFACT_ROOT / run_name
    if resume_dir:
        if not output.is_dir(): raise FileNotFoundError(f"resume artifact directory missing: {output}")
    else:
        if output.exists(): raise FileExistsError(f"refusing to overwrite artifact directory: {output}")
        output.mkdir(parents=True)
    config = {"schema_version": SCHEMA, "created_at": utc_now(), "dry_run": bool(args.dry_run), "backend": args.backend,
              "shots": args.shots, "num_per_split": args.num_per_split, "num_circuits": len(selected), "optimization_level": args.optimization_level,
              "seed_transpiler": SEED_TRANSPILER, "dataset": str(DATASET_JSONL), "dataset_sha256": sha256(DATASET_JSONL),
              "dataset_payload": str(DATASET_NPZ), "dataset_payload_sha256": sha256(DATASET_NPZ), "checkpoint": str(CHECKPOINT),
              "checkpoint_sha256": sha256(CHECKPOINT), "model": "transformer", "model_seed": 11,
              "selection_rule": "per requested split: product-state candidates only; circuit_id ascending; lowest sample_id per unique circuit; first N",
              "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(), "bit_order": BIT_ORDER, "status": "PREFLIGHT_RUNNING"}
    dump(output / "config.json", config); dump(output / "selected_circuits.json", {"selection_rule": config["selection_rule"], "circuits": selected_public})
    try:
        predictions = load_predictions(selected, rows, inputs, targets)
        if backend is None: service, backend = connect_backend(args.backend, args.instance)
        metadata = backend_metadata(backend); dump(output / "backend_metadata.json", metadata)
        bit_order = validate_bit_order()
        if bit_order["status"] != "PASS": raise RuntimeError("bit-order validation failed")
        _, _, _, _, generate_preset_pass_manager, _, SamplerV2 = qiskit_modules()
        originals = [build_workload(row, inputs[int(row["array_index"])]) for row in selected]
        canonical_qiskit = [statevector_distribution(circuit) for circuit in originals]
        canonical_frozen = [distribution(targets[int(row["array_index"])]) for row in selected]
        max_canonical_tvd = max(tvd(a, b) for a, b in zip(canonical_qiskit, canonical_frozen))
        if max_canonical_tvd > 1e-9: raise RuntimeError(f"bit-order/circuit reconstruction mismatch: max TVD={max_canonical_tvd}")
        manager = generate_preset_pass_manager(optimization_level=args.optimization_level, backend=backend, seed_transpiler=SEED_TRANSPILER)
        transpiled = list(manager.run(originals))
        metrics = [circuit_metrics(row, original, isa) for row, original, isa in zip(selected, originals, transpiled)]
        transpiled_ideal = [compact_transpiled_distribution(circuit) for circuit in transpiled]
        results = []
        for row, pred, sim, trans, metric in zip(selected, predictions, canonical_frozen, transpiled_ideal, metrics):
            model = distribution(pred)
            results.append({"schema_version": SCHEMA, "sample_id": row["sample_id"], "circuit_id": row["circuit_id"], "split": row["validation_split"],
                            "source_split": row["split_name"], "state_id": row["state_id"], "state_family": row["state_family"], "shots": args.shots,
                            "job_id": None, "model_distribution": model, "simulator_distribution": sim,
                            "transpiled_simulator_distribution": trans, "qpu_distribution": None, "raw_counts_file": None,
                            "tvd_model_sim": tvd(model, sim), "tvd_canonical_transpiled": tvd(sim, trans),
                            "tvd_sim_qpu": None, "tvd_model_qpu": None, **{key: metric[key] for key in ("original_depth", "transpiled_depth", "transpiled_2q_gate_count")}})
        dump(output / "circuit_metrics.json", {"bit_order_validation": bit_order, "canonical_reconstruction_max_tvd": max_canonical_tvd, "circuits": metrics})
        if args.dry_run:
            config["status"] = "QPU_SUBMISSION_READY"
            dump(output / "job_manifest.json", {"status": "NOT_SUBMITTED_DRY_RUN", "job_id": None, "backend": args.backend, "circuits": [row["sample_id"] for row in selected]})
        else:
            if args.preflight is None: raise ValueError("actual submission requires --preflight from a successful dry-run")
            preflight_matches(Path(args.preflight), config)
            if resume_dir:
                manifest = json.loads((output / "job_manifest.json").read_text())
                if not manifest.get("job_id"): raise ValueError("resume manifest has no job_id")
            else:
                manifest = {"status": "SUBMITTING", "job_id": None, "backend": args.backend, "shots": args.shots, "circuits": [row["sample_id"] for row in selected], "updated_at": utc_now()}
                dump(output / "job_manifest.json", manifest)
            try:
                if resume_dir:
                    if service is None: raise RuntimeError("resume requires IBM Runtime service")
                    job = service.job(manifest["job_id"])
                    manifest.update(status=str(job.status()), updated_at=utc_now()); dump(output / "job_manifest.json", manifest)
                else:
                    job = SamplerV2(mode=backend).run(transpiled, shots=args.shots)
                    manifest.update(status="SUBMITTED", job_id=job.job_id(), updated_at=utc_now()); dump(output / "job_manifest.json", manifest)
                if getattr(args, "no_wait", False):
                    status = str(job.status())
                    manifest.update(status=status, updated_at=utc_now()); dump(output / "job_manifest.json", manifest)
                    config["status"] = f"QPU_JOB_{status}"
                else:
                    pubs = job.result()
                    if len(pubs) != len(selected): raise RuntimeError(f"result count {len(pubs)} != circuit count {len(selected)}")
                    (output / "raw_counts").mkdir(exist_ok=True)
                    for row, result, pub in zip(results, selected, pubs):
                        counts = {str(key): int(value) for key, value in pub.data.meas.get_counts().items()}
                        raw_path = output / "raw_counts" / f"{result['sample_id']}.json"; dump(raw_path, counts)
                        qpu = normalize_counts(counts, args.shots)
                        row.update(job_id=manifest["job_id"], qpu_distribution=qpu, raw_counts_file=str(raw_path.relative_to(output)),
                                   tvd_sim_qpu=tvd(row["simulator_distribution"], qpu), tvd_model_qpu=tvd(row["model_distribution"], qpu))
                    manifest.update(status="COMPLETED", updated_at=utc_now()); dump(output / "job_manifest.json", manifest); config["status"] = "COMPLETED"
            except Exception as error:
                manifest.update(status="FAILED", error=f"{type(error).__name__}: {error}", updated_at=utc_now()); dump(output / "job_manifest.json", manifest)
                raise
        summary = aggregate(results)
        dump(output / "per_circuit_results.json", results); dump(output / "summary.json", summary)
        dump(output / "config.json", config); (output / "report.md").write_text(report(config, summary, metrics, config["status"]))
        checksums(output)
        print(config["status"])
        print(output)
        return output
    except Exception as error:
        config.update(status="BLOCKED", blocker=f"{type(error).__name__}: {error}")
        dump(output / "config.json", config)
        if not (output / "backend_metadata.json").exists(): dump(output / "backend_metadata.json", {"status": "UNAVAILABLE", "error": config["blocker"], "timestamp": utc_now()})
        if not (output / "circuit_metrics.json").exists(): dump(output / "circuit_metrics.json", {"status": "BLOCKED", "circuits": []})
        if not (output / "per_circuit_results.json").exists(): dump(output / "per_circuit_results.json", [])
        summary = aggregate([]); dump(output / "summary.json", summary)
        if not (output / "job_manifest.json").exists(): dump(output / "job_manifest.json", {"status": "NOT_SUBMITTED_BLOCKED", "job_id": None, "error": config["blocker"]})
        (output / "report.md").write_text(report(config, summary, [], "BLOCKED", config["blocker"]))
        checksums(output)
        print(f"QPU_SUBMISSION_BLOCKED: {config['blocker']}")
        print(output)
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--backend", required=True, help="real IBM backend name, for example ibm_pittsburgh")
    value.add_argument("--shots", type=int, default=4096)
    value.add_argument("--num-per-split", type=int, default=5)
    value.add_argument("--optimization-level", type=int, choices=range(4), default=2)
    value.add_argument("--instance", help="optional IBM Quantum instance CRN")
    value.add_argument("--output-dir")
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--preflight", help="successful dry-run artifact directory; required for submission")
    value.add_argument("--resume-dir", help="resume the IBM job recorded in an existing submission artifact directory")
    value.add_argument("--no-wait", action="store_true", help="record submission or current job status without waiting for results")
    return value


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.shots < 1: parser().error("--shots must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
