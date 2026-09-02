from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import networkx as nx
from scipy.stats import qmc


@dataclass(frozen=True)
class Workload:
    logical_id: str
    n_qubits: int
    p: int
    graph_family: str
    graph_seed: int
    parameter_index: int
    edges: tuple[tuple[int, int], ...]
    gammas: tuple[float, ...]
    betas: tuple[float, ...]
    graph_hash: str
    circuit_hash: str

    def row(self) -> dict[str, Any]:
        row = asdict(self)
        row["edges"] = json.dumps(row["edges"], separators=(",", ":"))
        row["gammas"] = json.dumps(row["gammas"], separators=(",", ":"))
        row["betas"] = json.dumps(row["betas"], separators=(",", ":"))
        return row


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def graph_edges(family: str, n: int, seed: int, edge_probability: float = 0.5) -> tuple[tuple[int, int], ...]:
    if family == "cycle":
        edges = nx.cycle_graph(n).edges()
    elif family == "random_3_regular":
        edges = nx.random_regular_graph(3, n, seed=seed).edges()
    elif family == "erdos_renyi":
        edges = nx.gnp_random_graph(n, edge_probability, seed=seed).edges()
    else:
        raise ValueError(f"unknown graph family: {family}")
    return tuple(sorted((min(a, b), max(a, b)) for a, b in edges))


def parameter_points(count: int, max_depth: int) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
    # One preregistered sequence is sliced for shallower depths; no backend gets different angles.
    samples = qmc.Sobol(2 * max_depth, scramble=False).random_base2(math.ceil(math.log2(count)))[:count]
    return [
        (
            tuple(float(x * math.pi) for x in row[:max_depth]),
            tuple(float(x * math.pi / 2) for x in row[max_depth:]),
        )
        for row in samples
    ]


def generate_corpus(config: dict[str, Any], widths: list[int] | None = None, families: list[str] | None = None) -> list[Workload]:
    widths = widths or list(config["base_qubits"])
    families = families or list(config["graph_families"])
    points = parameter_points(config["parameter_points"], max(config["depths"]))
    result: list[Workload] = []
    for family in families:
        for n in widths:
            for p in config["depths"]:
                for seed in config["graph_seeds"]:
                    edges = graph_edges(family, n, seed, config["stress_family"]["edge_probability"])
                    graph_payload = {"family": family, "n_qubits": n, "seed": seed, "edges": edges}
                    graph_hash = canonical_hash(graph_payload)
                    for parameter_index, (all_gammas, all_betas) in enumerate(points):
                        payload = {
                            **graph_payload,
                            "p": p,
                            "gammas": all_gammas[:p],
                            "betas": all_betas[:p],
                            "parameter_index": parameter_index,
                        }
                        circuit_hash = canonical_hash(payload)
                        logical_id = f"m2:{family}:n{n}:p{p}:g{seed}:a{parameter_index}"
                        result.append(
                            Workload(
                                logical_id,
                                n,
                                p,
                                family,
                                seed,
                                parameter_index,
                                edges,
                                all_gammas[:p],
                                all_betas[:p],
                                graph_hash,
                                circuit_hash,
                            )
                        )
    return result


def qiskit_circuit(workload: Workload, measure: bool = True):
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(workload.n_qubits, workload.n_qubits if measure else 0)
    circuit.h(range(workload.n_qubits))
    for gamma, beta in zip(workload.gammas, workload.betas):
        for a, b in workload.edges:
            circuit.rzz(-gamma, a, b)
        circuit.rx(2 * beta, range(workload.n_qubits))
    if measure:
        circuit.measure(range(workload.n_qubits), range(workload.n_qubits))
    circuit.metadata = {"logical_workload_id": workload.logical_id, "circuit_hash": workload.circuit_hash}
    return circuit
