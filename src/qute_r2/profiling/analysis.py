from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Iterable

import numpy as np


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    x = np.asarray(list(values), dtype=float)
    if not len(x):
        raise ValueError("cannot summarize empty values")
    return {
        "count": len(x),
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
        "minimum": float(np.min(x)),
        "maximum": float(np.max(x)),
    }


def statevector_guard(
    n_qubits: int,
    available_bytes: int,
    *,
    bytes_per_amplitude: int = 16,
    overhead_factor: float = 4.0,
    payload_fraction: float = 0.5,
    peak_fraction: float = 0.7,
    projected_seconds: float | None = None,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    payload = bytes_per_amplitude * (1 << n_qubits)
    projected_peak = payload * overhead_factor
    if payload > available_bytes * payload_fraction or projected_peak > available_bytes * peak_fraction:
        status = "SKIPPED_RESOURCE_GUARD"
    elif projected_seconds is not None and projected_seconds > timeout_seconds:
        status = "SKIPPED_TIMEOUT_PROJECTION"
    else:
        status = "RUN"
    return {
        "status": status,
        "n_qubits": n_qubits,
        "predicted_state_payload_bytes": payload,
        "available_host_memory_bytes": available_bytes,
        "expected_overhead_factor": overhead_factor,
        "projected_peak_bytes": projected_peak,
        "projected_seconds": projected_seconds,
    }


def aggregate_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = ("backend", "graph_family", "n_qubits", "p", "shots", "timing_mode", "batch_size")
    for row in rows:
        if row.get("status") == "OK":
            row.setdefault("batch_size", 1)
            groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for group_key, items in groups.items():
        row = dict(zip(keys, group_key))
        for field in ("total_ms", "execute_ms", "transpile_ms", "peak_rss_bytes", "incremental_peak_rss_bytes"):
            values = [float(item[field]) for item in items if item.get(field) not in (None, "")]
            if values:
                row.update({f"{field}_{key}": value for key, value in summarize(values).items()})
        row["circuits_per_second"] = 1000.0 * int(row["batch_size"]) / row["total_ms_median"]
        row["sample_count"] = len(items)
        result.append(row)
    return sorted(result, key=lambda r: tuple(r[k] for k in keys))


def best_classical(aggregate: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in aggregate:
        if row["timing_mode"] == "warm_single":
            key = (row["graph_family"], row["n_qubits"], row["p"], row["shots"])
            cells[key].append(row)
    result = []
    for key, rows in cells.items():
        if {row["backend"] for row in rows} >= {"statevector", "matrix_product_state"}:
            winner = min(rows, key=lambda r: r["total_ms_median"])
            result.append(
                {
                    "graph_family": key[0], "n_qubits": key[1], "p": key[2], "shots": key[3],
                    "best_classical_backend": winner["backend"],
                    "median_classical_latency_ms": winner["total_ms_median"],
                    "classical_peak_memory_bytes": winner.get("peak_rss_bytes_median"),
                    "validation_method": "statevector_and_mps_overlap" if key[1] <= 20 else "mps_with_guarded_statevector_overlap",
                    "evidence_type": "measured",
                }
            )
    return sorted(result, key=lambda r: (r["graph_family"], r["n_qubits"], r["p"], r["shots"]))


def budget_map(cells: list[dict[str, Any]], latency_budgets: list[float], memory_budgets: list[float]) -> list[dict[str, Any]]:
    rows = []
    for cell in cells:
        for latency in latency_budgets:
            speedup = cell["median_classical_latency_ms"] / latency
            for memory_mib in memory_budgets:
                classical_mib = (cell["classical_peak_memory_bytes"] or 0) / 2**20
                rows.append({
                    **{key: cell[key] for key in ("graph_family", "n_qubits", "p", "shots", "best_classical_backend")},
                    "neural_latency_budget_ms": latency,
                    "neural_peak_memory_budget_mib": memory_mib,
                    "speedup": speedup,
                    "speedup_ge_2x": speedup >= 2,
                    "speedup_ge_5x": speedup >= 5,
                    "speedup_ge_10x": speedup >= 10,
                    "memory_lower_than_classical": classical_mib > memory_mib,
                })
    return rows


def candidate_regions(cells: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    target_ms = config["candidate"]["planning_neural_latency_ms"]
    target_mib = config["candidate"]["planning_neural_memory_mib"]
    max_teacher_hours = config["candidate"]["max_teacher_10000_hours"]
    candidates = []
    for cell in cells:
        latency = cell["median_classical_latency_ms"]
        peak_mib = (cell["classical_peak_memory_bytes"] or 0) / 2**20
        teacher_hours = latency * 10_000 / 3_600_000
        if latency / target_ms < config["candidate"]["minimum_speedup"] or peak_mib <= target_mib or teacher_hours > max_teacher_hours:
            continue
        candidates.append({
            **cell,
            "required_neural_latency_2x_ms": latency / 2,
            "required_neural_latency_5x_ms": latency / 5,
            "required_neural_latency_10x_ms": latency / 10,
            "required_neural_memory_mib": min(target_mib, peak_mib * 0.8),
            "teacher_10000_serial_hours": teacher_hours,
            "teacher_data_feasibility": "FEASIBLE_SMALL_STUDY",
            "confidence": "moderate_measured",
            "limitations": "No scale-capable QuTE accuracy measurement exists at this width.",
        })
    return candidates


def decision(correctness_pass: bool, backends: set[str], candidates: list[dict[str, Any]], mps_dominates: bool) -> str:
    if not correctness_pass or not {"statevector", "matrix_product_state"}.issubset(backends):
        return "BLOCKED"
    if candidates:
        return "PROCEED_TO_M3"
    if mps_dominates:
        return "PIVOT_WORKLOAD_BEFORE_M3"
    return "NO_FEASIBLE_REGION_FOUND"


def scaling_fits(aggregate: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in aggregate:
        if row["timing_mode"] == "warm_single":
            groups[(row["backend"], row["graph_family"], row["p"], row["shots"])].append(row)
    fits: dict[str, Any] = {}
    for key, rows in groups.items():
        rows = sorted(rows, key=lambda r: r["n_qubits"])
        if len(rows) < 4:
            continue
        n = np.asarray([r["n_qubits"] for r in rows], dtype=float)
        y = np.log(np.asarray([max(r["total_ms_median"], 1e-9) for r in rows]))
        hold = np.arange(len(n)) % 4 == 3
        train = ~hold
        coefficients, covariance = np.polyfit(n[train], y[train], 1, cov=True)
        predicted = np.exp(np.polyval(coefficients, n))
        observed = np.exp(y)
        residuals = observed - predicted
        key_name = ":".join(map(str, key))
        fits[key_name] = {
            "fit_family": "log_linear_empirical" if key[0] == "statevector" else "log_linear_descriptive_not_assumed_physical",
            "parameters": {"log_slope_per_qubit": float(coefficients[0]), "log_intercept": float(coefficients[1])},
            "parameter_95pct_ci": (1.96 * np.sqrt(np.diag(covariance))).tolist(),
            "residuals_ms": residuals.tolist(),
            "held_out_mean_absolute_percentage_error": float(np.mean(np.abs((observed[hold] - predicted[hold]) / observed[hold]))) if hold.any() else None,
            "fitted_range": [int(n.min()), int(n.max())],
            "extrapolated_range": [int(n.max() + 2), int(n.max() + 4)],
            "extrapolated_ms": {str(int(x)): float(math.exp(np.polyval(coefficients, x))) for x in (n.max() + 2, n.max() + 4)},
        }
    return fits
