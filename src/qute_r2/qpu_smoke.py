from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from qute_r2.circuit_ir import CircuitIR, GateIR
from qute_r2.m3.model import M3Backend
from qute_r2.m3.run import outcome_metrics
from qute_r2.profiling.corpus import Workload, canonical_hash, graph_edges, parameter_points, qiskit_circuit
from qute_r2.qiskit_adapter import to_qiskit

ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def counts_to_outcomes(counts: dict[str, int]) -> np.ndarray:
    rows = [np.full(count, int(bits.replace(" ", ""), 2), dtype=np.uint32) for bits, count in sorted(counts.items())]
    return np.concatenate(rows) if rows else np.asarray([], dtype=np.uint32)


def tvd(a: dict[str, int], b: dict[str, int]) -> float:
    shots_a, shots_b = sum(a.values()), sum(b.values())
    keys = set(a) | set(b)
    return float(0.5 * sum(abs(a.get(k, 0) / shots_a - b.get(k, 0) / shots_b) for k in keys))


def qaoa_workload(item: dict[str, Any]) -> Workload:
    spec = item["qaoa_workload"]
    points = parameter_points(spec["parameter_index"] + 1, spec["p"])
    edges = graph_edges(spec["family"], spec["n_qubits"], spec["graph_seed"])
    gammas, betas = points[spec["parameter_index"]]
    payload = {"family": spec["family"], "n_qubits": spec["n_qubits"], "p": spec["p"], "edges": edges, "gammas": gammas[: spec["p"]], "betas": betas[: spec["p"]], "graph_seed": spec["graph_seed"], "point_index": spec["parameter_index"]}
    return Workload(item["logical_id"], spec["n_qubits"], spec["p"], spec["family"], spec["graph_seed"], spec["parameter_index"], edges, gammas[: spec["p"]], betas[: spec["p"]], canonical_hash({"family": spec["family"], "n_qubits": spec["n_qubits"], "edges": edges}), canonical_hash(payload))


def circuit_ir(item: dict[str, Any]) -> CircuitIR:
    if item.get("qaoa_workload"):
        return from_qaoa(item)
    return CircuitIR(item["n_qubits"], tuple(GateIR(g["name"], tuple(g["qubits"]), tuple(g.get("params", ()))) for g in item["gates"]), {"logical_id": item["logical_id"]})


def from_qaoa(item: dict[str, Any]) -> CircuitIR:
    row = qaoa_workload(item)
    gates = [GateIR("h", (q,)) for q in range(row.n_qubits)]
    for gamma, beta in zip(row.gammas, row.betas):
        gates.extend(GateIR("rzz", edge, (-gamma,)) for edge in row.edges)
        gates.extend(GateIR("rx", (q,), (2 * beta,)) for q in range(row.n_qubits))
    gates.append(GateIR("measure", tuple(range(row.n_qubits))))
    return CircuitIR(row.n_qubits, tuple(gates), {"logical_id": item["logical_id"], "qaoa_workload": item["qaoa_workload"]})


def load_circuits(config: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for item in config["circuits"]:
        ir = circuit_ir(item)
        result.append({"logical_id": item["logical_id"], "ir": ir, "qaoa": qaoa_workload(item) if item.get("qaoa_workload") else None})
    return result


def run_local(config: dict[str, Any], circuits: list[dict[str, Any]], out: Path) -> list[dict[str, Any]]:
    backend = None
    result = []
    for index, item in enumerate(circuits):
        if item["qaoa"] is None:
            result.append({"logical_id": item["logical_id"], "circuit_hash": item["ir"].circuit_hash(), "status": "ABSTAIN", "reason": "v13 backend only supports QAOA-shaped Workload"})
            continue
        try:
            backend = backend or M3Backend.from_pretrained(ROOT / config["checkpoint"])
        except FileNotFoundError:
            result.append({"logical_id": item["logical_id"], "circuit_hash": item["ir"].circuit_hash(), "status": "ABSTAIN", "reason": "v13 checkpoint file is not present in this checkout"})
            continue
        start = time.perf_counter_ns()
        counts, metadata = backend.sample(item["qaoa"], config["shots"], seed=71000 + index)
        row = {"logical_id": item["logical_id"], "circuit_hash": item["ir"].circuit_hash(), "status": "RECORDED", "counts": counts, "wall_ms": (time.perf_counter_ns() - start) / 1e6, "metadata": metadata}
        result.append(row); write_json(out / "local_counts" / f"{index:02d}.json", counts)
    return result


def run_aer(config: dict[str, Any], circuits: list[dict[str, Any]], out: Path) -> list[dict[str, Any]]:
    try:
        from qiskit import transpile
        from qiskit_aer import AerSimulator
    except Exception as exc:
        return [{"status": "SKIPPED", "reason": repr(exc)}]
    sim = AerSimulator(method="statevector", precision="double", max_parallel_threads=1, max_parallel_experiments=1)
    result = []
    for index, item in enumerate(circuits):
        circuit = qiskit_circuit(item["qaoa"]) if item["qaoa"] is not None else to_qiskit(item["ir"])
        circuit = transpile(circuit, sim, optimization_level=0, seed_transpiler=71000)
        start = time.perf_counter_ns()
        counts = dict(sim.run(circuit, shots=config["shots"], seed_simulator=72000 + index).result().get_counts())
        result.append({"logical_id": item["logical_id"], "counts": counts, "wall_ms": (time.perf_counter_ns() - start) / 1e6})
        write_json(out / "aer_counts" / f"{index:02d}.json", counts)
    return result


def select_backend(service, name: str):
    if name != "least_busy":
        return service.backend(name)
    return min(service.backends(simulator=False, operational=True, min_num_qubits=8), key=lambda b: getattr(b.status(), "pending_jobs", 10**9))


def submit_qpu(config: dict[str, Any], circuits: list[dict[str, Any]], out: Path) -> None:
    try:
        from qiskit import transpile
        from qiskit_ibm_runtime import QiskitRuntimeService
    except Exception as exc:
        write_json(out / "QPU_SMOKE_DECISION.json", {"decision": "QPU_SMOKE_BLOCKED", "reason": f"missing dependency: {exc!r}"}); return
    service = QiskitRuntimeService(channel=os.getenv("QISKIT_IBM_CHANNEL", "ibm_quantum"), token=os.getenv("QISKIT_IBM_TOKEN") or None)
    backend = select_backend(service, config["backend_name"])
    qiskit_circuits = [qiskit_circuit(item["qaoa"]) if item["qaoa"] is not None else to_qiskit(item["ir"]) for item in circuits]
    submitted_at = time.time(); job = backend.run(transpile(qiskit_circuits, backend, optimization_level=1, seed_transpiler=73000), shots=config["shots"])
    write_json(out / "backend_metadata.json", {"backend_name": backend.name, "simulator": False, "num_qubits": getattr(backend, "num_qubits", None), "pending_jobs_at_submit": getattr(backend.status(), "pending_jobs", None)})
    write_json(out / "job_manifest.json", {"job_id": job.job_id(), "backend_name": backend.name, "shots": config["shots"], "status": str(job.status()), "submitted_at_unix": submitted_at, "circuits": [item["logical_id"] for item in circuits]})


def collect_qpu(config: dict[str, Any], out: Path) -> list[dict[str, Any]] | None:
    manifest = json.loads((out / "job_manifest.json").read_text())
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except Exception as exc:
        write_json(out / "QPU_SMOKE_DECISION.json", {"decision": "QPU_SMOKE_BLOCKED", "reason": f"missing dependency: {exc!r}"}); return None
    job = QiskitRuntimeService(channel=os.getenv("QISKIT_IBM_CHANNEL", "ibm_quantum"), token=os.getenv("QISKIT_IBM_TOKEN") or None).job(manifest["job_id"])
    if str(job.status()).upper().split(".")[-1] not in {"DONE", "COMPLETED"}:
        write_json(out / "QPU_SMOKE_DECISION.json", {"decision": "QPU_SMOKE_BLOCKED", "reason": "job not complete", "job_id": manifest["job_id"], "status": str(job.status())}); return None
    raw = job.result(); rows = []
    for index, logical_id in enumerate(manifest["circuits"]):
        counts = dict(raw.get_counts(index)); write_json(out / "qpu_counts" / f"{index:02d}.json", counts); rows.append({"logical_id": logical_id, "counts": counts})
    manifest["completed_at_unix"] = time.time(); write_json(out / "job_manifest.json", manifest)
    return rows


def report(config: dict[str, Any], circuits: list[dict[str, Any]], out: Path) -> None:
    local = json.loads((out / "local_results.json").read_text())
    aer = json.loads((out / "aer_results.json").read_text()) if (out / "aer_results.json").exists() else []
    qpu = collect_qpu(config, out) if (out / "job_manifest.json").exists() else None
    if not qpu: return
    rows = []
    for index, item in enumerate(circuits):
        row = {"logical_id": item["logical_id"]}
        if aer and aer[0].get("status") != "SKIPPED": row["tvd_aer_qpu"] = tvd(aer[index]["counts"], qpu[index]["counts"])
        if local[index].get("status") == "RECORDED":
            row["tvd_local_qpu"] = tvd(local[index]["counts"], qpu[index]["counts"]); row["local_wall_ms"] = local[index]["wall_ms"]
            if item["qaoa"] is not None: row.update(outcome_metrics(counts_to_outcomes(qpu[index]["counts"]), counts_to_outcomes(local[index]["counts"]), item["qaoa"]))
        else:
            row["local_status"] = local[index]["status"]; row["local_reason"] = local[index]["reason"]
        rows.append(row)
    recorded = [r for r in rows if "local_wall_ms" in r]
    summary = {"decision": "QPU_SMOKE_RECORDED", "qpu_authorized": True, "qpu_replacement_claimed": False, "count": len(rows), "shots": config["shots"], "local_recorded": len(recorded), "local_abstained": len(rows) - len(recorded)}
    if recorded: summary["local_median_latency_ms"] = statistics.median(r["local_wall_ms"] for r in recorded)
    write_json(out / "per_circuit_comparison.json", rows); write_json(out / "summary.json", summary); write_json(out / "QPU_SMOKE_DECISION.json", {"decision": "QPU_SMOKE_RECORDED", "qpu_authorized": True, "qpu_replacement_claimed": False})
    (out / "QPU_SMOKE_REPORT.md").write_text("# QPU Smoke v1 Report\n\n" + json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, default=ROOT / "configs/r2/qpu_smoke_v1.json"); parser.add_argument("--submit-qpu", action="store_true"); parser.add_argument("--collect", action="store_true"); args = parser.parse_args()
    config = json.loads(args.config.read_text()); out = ROOT / config["artifact_root"]; out.mkdir(parents=True, exist_ok=True)
    circuits = load_circuits(config)
    write_json(out / "config.json", config); write_json(out / "circuits.json", [{"logical_id": c["logical_id"], "circuit_hash": c["ir"].circuit_hash(), "ir": c["ir"].to_dict(), "local_backend_supported": c["qaoa"] is not None} for c in circuits]); write_json(out / "environment.json", {"platform": platform.platform(), "pid": os.getpid()})
    if args.submit_qpu: submit_qpu(config, circuits, out); return
    local = run_local(config, circuits, out); write_json(out / "local_results.json", local); write_json(out / "aer_results.json", run_aer(config, circuits, out))
    if args.collect: report(config, circuits, out)
    else:
        recorded = [row for row in local if row.get("status") == "RECORDED"]; peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(out / "LOCAL_ONLY_DECISION.json", {"decision": "LOCAL_SMOKE_RECORDED", "qpu_pending": not (out / "job_manifest.json").exists(), "local_recorded": len(recorded), "local_abstained": len(local) - len(recorded), "local_median_latency_ms": statistics.median(row["wall_ms"] for row in recorded) if recorded else None, "local_peak_rss_raw": peak})


if __name__ == "__main__": main()
