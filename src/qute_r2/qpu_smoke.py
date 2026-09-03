from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from qute_r2.m3.model import M3Backend
from qute_r2.m3.run import outcome_metrics
from qute_r2.profiling.corpus import Workload, canonical_hash, graph_edges, parameter_points, qiskit_circuit

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


def make_workloads(config: dict[str, Any]) -> list[Workload]:
    points = parameter_points(max(row["parameter_index"] for row in config["circuits"]) + 1, max(row["p"] for row in config["circuits"]))
    rows = []
    for item in config["circuits"]:
        edges = graph_edges(item["family"], item["n_qubits"], item["graph_seed"])
        gammas, betas = points[item["parameter_index"]]
        payload = {"family": item["family"], "n_qubits": item["n_qubits"], "p": item["p"], "edges": edges, "gammas": gammas[: item["p"]], "betas": betas[: item["p"]], "graph_seed": item["graph_seed"], "point_index": item["parameter_index"]}
        rows.append(Workload(f"qpu-smoke:{item['family']}:n{item['n_qubits']}:p{item['p']}:g{item['graph_seed']}:a{item['parameter_index']}", item["n_qubits"], item["p"], item["family"], item["graph_seed"], item["parameter_index"], edges, gammas[: item["p"]], betas[: item["p"]], canonical_hash({"family": item["family"], "n_qubits": item["n_qubits"], "edges": edges}), canonical_hash(payload)))
    return rows


def run_local(config: dict[str, Any], rows: list[Workload], out: Path) -> list[dict[str, Any]]:
    backend = M3Backend.from_pretrained(ROOT / config["checkpoint"])
    result = []
    for index, row in enumerate(rows):
        start = time.perf_counter_ns()
        counts, metadata = backend.sample(row, config["shots"], seed=71000 + index)
        result.append({"logical_id": row.logical_id, "circuit_hash": row.circuit_hash, "counts": counts, "wall_ms": (time.perf_counter_ns() - start) / 1e6, "metadata": metadata})
        write_json(out / "local_counts" / f"{index:02d}.json", counts)
    return result


def run_aer(config: dict[str, Any], rows: list[Workload], out: Path) -> list[dict[str, Any]]:
    try:
        from qiskit import transpile
        from qiskit_aer import AerSimulator
    except Exception as exc:
        return [{"status": "SKIPPED", "reason": repr(exc)}]
    sim = AerSimulator(method="statevector", precision="double", max_parallel_threads=1, max_parallel_experiments=1)
    result = []
    for index, row in enumerate(rows):
        circuit = transpile(qiskit_circuit(row), sim, optimization_level=0, seed_transpiler=71000)
        start = time.perf_counter_ns()
        counts = dict(sim.run(circuit, shots=config["shots"], seed_simulator=72000 + index).result().get_counts())
        result.append({"logical_id": row.logical_id, "counts": counts, "wall_ms": (time.perf_counter_ns() - start) / 1e6})
        write_json(out / "aer_counts" / f"{index:02d}.json", counts)
    return result


def select_backend(service, name: str):
    if name != "least_busy":
        return service.backend(name)
    backends = service.backends(simulator=False, operational=True, min_num_qubits=8)
    return min(backends, key=lambda b: getattr(b.status(), "pending_jobs", 10**9))


def submit_qpu(config: dict[str, Any], rows: list[Workload], out: Path) -> None:
    try:
        from qiskit import transpile
        from qiskit_ibm_runtime import QiskitRuntimeService
    except Exception as exc:
        write_json(out / "QPU_SMOKE_DECISION.json", {"decision": "QPU_SMOKE_BLOCKED", "reason": f"missing dependency: {exc!r}"})
        return
    service = QiskitRuntimeService(channel=os.getenv("QISKIT_IBM_CHANNEL", "ibm_quantum"), token=os.getenv("QISKIT_IBM_TOKEN") or None)
    backend = select_backend(service, config["backend_name"])
    circuits = transpile([qiskit_circuit(row) for row in rows], backend, optimization_level=1, seed_transpiler=73000)
    job = backend.run(circuits, shots=config["shots"])
    write_json(out / "backend_metadata.json", {"backend_name": backend.name, "simulator": False, "num_qubits": getattr(backend, "num_qubits", None), "pending_jobs_at_submit": getattr(backend.status(), "pending_jobs", None)})
    write_json(out / "job_manifest.json", {"job_id": job.job_id(), "backend_name": backend.name, "shots": config["shots"], "status": str(job.status()), "circuits": [row.logical_id for row in rows]})


def collect_qpu(config: dict[str, Any], out: Path) -> list[dict[str, Any]] | None:
    manifest = json.loads((out / "job_manifest.json").read_text())
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except Exception as exc:
        write_json(out / "QPU_SMOKE_DECISION.json", {"decision": "QPU_SMOKE_BLOCKED", "reason": f"missing dependency: {exc!r}"})
        return None
    service = QiskitRuntimeService(channel=os.getenv("QISKIT_IBM_CHANNEL", "ibm_quantum"), token=os.getenv("QISKIT_IBM_TOKEN") or None)
    job = service.job(manifest["job_id"])
    if str(job.status()).upper().split(".")[-1] != "DONE":
        write_json(out / "QPU_SMOKE_DECISION.json", {"decision": "QPU_SMOKE_BLOCKED", "reason": "job not complete", "job_id": manifest["job_id"], "status": str(job.status())})
        return None
    raw = job.result()
    rows = []
    for index, logical_id in enumerate(manifest["circuits"]):
        counts = dict(raw.get_counts(index))
        write_json(out / "qpu_counts" / f"{index:02d}.json", counts)
        rows.append({"logical_id": logical_id, "counts": counts})
    return rows


def report(config: dict[str, Any], workloads: list[Workload], out: Path) -> None:
    local = json.loads((out / "local_results.json").read_text())
    aer = json.loads((out / "aer_results.json").read_text()) if (out / "aer_results.json").exists() else []
    qpu = collect_qpu(config, out) if (out / "job_manifest.json").exists() else None
    if not qpu:
        return
    rows = []
    for index, row in enumerate(workloads):
        qpu_counts, local_counts = qpu[index]["counts"], local[index]["counts"]
        item = {"logical_id": row.logical_id, "tvd_local_qpu": tvd(local_counts, qpu_counts), "local_wall_ms": local[index]["wall_ms"], **outcome_metrics(counts_to_outcomes(qpu_counts), counts_to_outcomes(local_counts), row)}
        if aer and aer[0].get("status") != "SKIPPED":
            item["tvd_aer_qpu"] = tvd(aer[index]["counts"], qpu_counts)
            item["tvd_local_aer"] = tvd(local_counts, aer[index]["counts"])
        rows.append(item)
    summary = {"decision": "QPU_SMOKE_RECORDED", "qpu_authorized": True, "qpu_replacement_claimed": False, "count": len(rows), "shots": config["shots"], "local_median_latency_ms": statistics.median(r["local_wall_ms"] for r in rows), "tvd_local_qpu_median": statistics.median(r["tvd_local_qpu"] for r in rows), "energy_error_per_edge_median": statistics.median(r["energy_error_per_edge"] for r in rows), "zz_mae_median": statistics.median(r["zz_mae"] for r in rows)}
    write_json(out / "per_circuit_comparison.json", rows)
    write_json(out / "summary.json", summary)
    text = "# QPU Smoke v1 Report\n\n" + json.dumps(summary, indent=2, sort_keys=True) + "\n\nDescriptive only: QPU noise and finite shots are not separated from model error.\n"
    (out / "QPU_SMOKE_REPORT.md").write_text(text)
    write_json(out / "QPU_SMOKE_DECISION.json", {"decision": "QPU_SMOKE_RECORDED", "qpu_authorized": True, "qpu_replacement_claimed": False})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/r2/qpu_smoke_v1.json")
    parser.add_argument("--submit-qpu", action="store_true")
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    out = ROOT / config["artifact_root"]
    out.mkdir(parents=True, exist_ok=True)
    workloads = make_workloads(config)
    write_json(out / "config.json", config)
    write_json(out / "workloads.json", [{"logical_id": r.logical_id, "n_qubits": r.n_qubits, "p": r.p, "graph_family": r.graph_family, "graph_seed": r.graph_seed, "parameter_index": r.parameter_index, "circuit_hash": r.circuit_hash} for r in workloads])
    write_json(out / "environment.json", {"platform": platform.platform(), "pid": os.getpid()})
    if args.submit_qpu:
        submit_qpu(config, workloads, out)
        return
    local = run_local(config, workloads, out)
    write_json(out / "local_results.json", local)
    write_json(out / "aer_results.json", run_aer(config, workloads, out))
    if args.collect:
        report(config, workloads, out)
    else:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(out / "LOCAL_ONLY_DECISION.json", {"decision": "LOCAL_SMOKE_RECORDED", "qpu_pending": not (out / "job_manifest.json").exists(), "local_median_latency_ms": statistics.median(row["wall_ms"] for row in local), "local_peak_rss_raw": peak})


if __name__ == "__main__":
    main()
