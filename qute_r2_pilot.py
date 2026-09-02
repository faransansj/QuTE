"""Small M1 pilot: one pretrained QAOA-conditioned autoregressive sampler.

The exact simulator is development-only. QuTEBackend.run never calls it.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

MAX_QUBITS = 6
_EXACT_CALLS = 0


@dataclass(frozen=True)
class QAOACircuit:
    n_qubits: int
    edges: tuple[tuple[int, int], ...]
    gamma: float
    beta: float
    p: int = 1

    def __post_init__(self) -> None:
        clean = tuple(sorted((min(a, b), max(a, b)) for a, b in self.edges))
        if clean != self.edges or len(set(clean)) != len(clean):
            raise ValueError("edges must be unique, sorted canonical pairs")
        if any(a < 0 or b >= self.n_qubits or a == b for a, b in clean):
            raise ValueError("invalid edge")

    def canonical(self) -> dict[str, Any]:
        return {
            "beta": float(self.beta),
            "edges": [list(edge) for edge in self.edges],
            "gamma": float(self.gamma),
            "n_qubits": self.n_qubits,
            "p": self.p,
        }

    @property
    def circuit_hash(self) -> str:
        raw = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    @property
    def graph_hash(self) -> str:
        raw = json.dumps(
            {"edges": self.canonical()["edges"], "n_qubits": self.n_qubits},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(raw).hexdigest()


def cycle_edges(n_qubits: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((min(i, (i + 1) % n_qubits), max(i, (i + 1) % n_qubits)) for i in range(n_qubits)))


def cycle_plus_chord(n_qubits: int, chord: tuple[int, int] | None) -> tuple[tuple[int, int], ...]:
    edges = set(cycle_edges(n_qubits))
    if chord is not None:
        edges.add((min(chord), max(chord)))
    return tuple(sorted(edges))


def available_chords(n_qubits: int) -> tuple[tuple[int, int], ...]:
    cycle = set(cycle_edges(n_qubits))
    return tuple((a, b) for a in range(n_qubits) for b in range(a + 1, n_qubits) if (a, b) not in cycle)


def exact_probabilities(circuit: QAOACircuit) -> np.ndarray:
    """Development-only exact QAOA probabilities in Qiskit count-key order."""
    global _EXACT_CALLS
    _EXACT_CALLS += 1
    if circuit.p != 1:
        raise ValueError("pilot exact teacher supports p=1")
    n = circuit.n_qubits
    size = 1 << n
    indices = np.arange(size, dtype=np.uint64)
    state = np.full(size, 1 / math.sqrt(size), dtype=np.complex128)
    cut = np.zeros(size, dtype=np.float64)
    for a, b in circuit.edges:
        cut += ((indices >> a) & 1) ^ ((indices >> b) & 1)
    state *= np.exp(-1j * circuit.gamma * cut)
    c, s = math.cos(circuit.beta), -1j * math.sin(circuit.beta)
    for qubit in range(n):
        stride = 1 << qubit
        view = state.reshape(-1, 2 * stride)
        zero, one = view[:, :stride].copy(), view[:, stride:].copy()
        view[:, :stride] = c * zero + s * one
        view[:, stride:] = s * zero + c * one
    probabilities = np.abs(state) ** 2
    return probabilities / probabilities.sum()


def sample_counts(probabilities: np.ndarray, n_qubits: int, shots: int, seed: int) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    outcomes = rng.choice(len(probabilities), size=shots, p=probabilities)
    return dict(sorted(Counter(format(int(x), f"0{n_qubits}b") for x in outcomes).items()))


def _adjacency_and_degrees(circuit: QAOACircuit, max_qubits: int = MAX_QUBITS) -> np.ndarray:
    adjacency: list[float] = []
    edge_set = set(circuit.edges)
    for a in range(max_qubits):
        for b in range(a + 1, max_qubits):
            adjacency.append(float((a, b) in edge_set))
    degree = np.zeros(max_qubits, dtype=np.float32)
    for a, b in circuit.edges:
        degree[a] += 1
        degree[b] += 1
    return np.asarray(
        adjacency
        + list(degree / max(1, circuit.n_qubits - 1))
        + [
            math.sin(circuit.gamma),
            math.cos(circuit.gamma),
            math.sin(circuit.beta),
            math.cos(circuit.beta),
            circuit.n_qubits / max_qubits,
        ],
        dtype=np.float32,
    )


def autoregressive_feature(
    circuit: QAOACircuit,
    prefix: Iterable[int],
    position: int,
    max_qubits: int = MAX_QUBITS,
) -> np.ndarray:
    prefix_values = np.zeros(max_qubits, dtype=np.float32)
    prefix_list = list(prefix)
    prefix_values[: len(prefix_list)] = np.asarray(prefix_list, dtype=np.float32) * 2 - 1
    position_values = np.zeros(max_qubits, dtype=np.float32)
    position_values[position] = 1
    return np.concatenate((_adjacency_and_degrees(circuit, max_qubits), prefix_values, position_values))


class ConditionalAutoregressiveSampler(nn.Module):
    def __init__(self, max_qubits: int = MAX_QUBITS, hidden: int = 96) -> None:
        super().__init__()
        adjacency = max_qubits * (max_qubits - 1) // 2
        self.max_qubits = max_qubits
        self.hidden = hidden
        feature_size = adjacency + max_qubits + 5 + max_qubits * 2
        self.network = nn.Sequential(
            nn.Linear(feature_size, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def training_rows(
    teacher_counts: list[tuple[QAOACircuit, dict[str, int]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features: list[np.ndarray] = []
    targets: list[float] = []
    weights: list[float] = []
    for circuit, counts in teacher_counts:
        shots = sum(counts.values())
        for bitstring, count in counts.items():
            bits = [int(bit) for bit in bitstring]
            for position, target in enumerate(bits):
                features.append(autoregressive_feature(circuit, bits[:position], position))
                targets.append(float(target))
                weights.append(count / shots)
    return (
        torch.from_numpy(np.stack(features)),
        torch.tensor(targets, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )


def train_sampler(
    teacher_counts: list[tuple[QAOACircuit, dict[str, int]]],
    *,
    seed: int = 2026,
    epochs: int = 40,
    learning_rate: float = 3e-3,
) -> tuple[ConditionalAutoregressiveSampler, list[float]]:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = ConditionalAutoregressiveSampler()
    features, targets, weights = training_rows(teacher_counts)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        per_row = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = (per_row * weights).sum() / weights.sum()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    return model, losses


def save_checkpoint(model: ConditionalAutoregressiveSampler, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": {"hidden": model.hidden, "max_qubits": model.max_qubits}, "state_dict": model.state_dict()},
        path,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checkpoint(path: str | Path) -> ConditionalAutoregressiveSampler:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = ConditionalAutoregressiveSampler(**payload["config"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def model_distribution(model: ConditionalAutoregressiveSampler, circuit: QAOACircuit) -> np.ndarray:
    """Small-width evaluation only; the backend never enumerates this table."""
    probabilities = []
    with torch.inference_mode():
        for value in range(1 << circuit.n_qubits):
            bits = [int(bit) for bit in format(value, f"0{circuit.n_qubits}b")]
            probability = 1.0
            for position, target in enumerate(bits):
                feature = torch.from_numpy(autoregressive_feature(circuit, bits[:position], position)).unsqueeze(0)
                one = float(torch.sigmoid(model(feature))[0])
                probability *= one if target else 1 - one
            probabilities.append(probability)
    result = np.asarray(probabilities, dtype=np.float64)
    return result / result.sum()


def distribution_metrics(exact: np.ndarray, predicted: np.ndarray, circuit: QAOACircuit) -> dict[str, float]:
    midpoint = (exact + predicted) / 2
    nz_exact = exact > 0
    nz_predicted = predicted > 0
    js = 0.5 * np.sum(exact[nz_exact] * np.log(exact[nz_exact] / midpoint[nz_exact]))
    js += 0.5 * np.sum(predicted[nz_predicted] * np.log(predicted[nz_predicted] / midpoint[nz_predicted]))
    cuts = cut_values(circuit)
    exact_energy = float(exact @ cuts)
    predicted_energy = float(predicted @ cuts)
    optimum = cuts == cuts.max()
    return {
        "tvd": float(0.5 * np.abs(exact - predicted).sum()),
        "hellinger_fidelity": float(np.square(np.sqrt(exact * predicted).sum())),
        "jensen_shannon": float(js),
        "cut_energy_error": abs(exact_energy - predicted_energy),
        "cut_energy_error_per_edge": abs(exact_energy - predicted_energy) / len(circuit.edges),
        "optimal_probability_error": abs(float(exact[optimum].sum() - predicted[optimum].sum())),
    }


def cut_values(circuit: QAOACircuit) -> np.ndarray:
    indices = np.arange(1 << circuit.n_qubits, dtype=np.uint64)
    result = np.zeros(len(indices), dtype=np.float64)
    for a, b in circuit.edges:
        result += ((indices >> a) & 1) ^ ((indices >> b) & 1)
    return result


class QuTEUnsupportedCircuitError(ValueError):
    pass


class QuTEResult:
    def __init__(self, counts: list[dict[str, int]], metadata: list[dict[str, Any]]) -> None:
        self._counts = counts
        self.metadata = metadata

    def get_counts(self, experiment: int | None = None) -> dict[str, int] | list[dict[str, int]]:
        if experiment is not None:
            return self._counts[experiment]
        return self._counts[0] if len(self._counts) == 1 else self._counts


class QuTEJob:
    def __init__(self, result: QuTEResult) -> None:
        self._result = result

    def result(self) -> QuTEResult:
        return self._result


class QuTEBackend:
    def __init__(self, model: ConditionalAutoregressiveSampler, model_hash: str) -> None:
        self.model = model
        self.model_hash = model_hash

    @classmethod
    def from_pretrained(cls, model_name: str | Path) -> "QuTEBackend":
        path = Path(model_name)
        return cls(load_checkpoint(path), hashlib.sha256(path.read_bytes()).hexdigest())

    @staticmethod
    def _parse(circuit: QAOACircuit | Any) -> QAOACircuit:
        if isinstance(circuit, QAOACircuit):
            return circuit
        metadata = getattr(circuit, "metadata", None) or {}
        payload = metadata.get("qute_qaoa")
        if not payload:
            raise QuTEUnsupportedCircuitError("Qiskit circuit requires metadata['qute_qaoa']")
        return QAOACircuit(
            n_qubits=int(payload["n_qubits"]),
            edges=tuple(sorted(tuple(sorted(map(int, edge))) for edge in payload["edges"])),
            gamma=float(payload["gamma"]),
            beta=float(payload["beta"]),
            p=int(payload.get("p", 1)),
        )

    @staticmethod
    def _supported(circuit: QAOACircuit) -> bool:
        if circuit.n_qubits != MAX_QUBITS or circuit.p != 1:
            return False
        base = set(cycle_edges(circuit.n_qubits))
        extras = set(circuit.edges) - base
        return base.issubset(circuit.edges) and len(extras) <= 1 and all(edge in available_chords(circuit.n_qubits) for edge in extras)

    def _sample(self, circuit: QAOACircuit, shots: int, seed: int) -> dict[str, int]:
        rng = np.random.default_rng(seed)
        prefixes = np.zeros((shots, self.model.max_qubits), dtype=np.int8)
        with torch.inference_mode():
            for position in range(circuit.n_qubits):
                features = np.stack(
                    [autoregressive_feature(circuit, row[:position], position) for row in prefixes]
                )
                probabilities = torch.sigmoid(self.model(torch.from_numpy(features))).numpy()
                prefixes[:, position] = rng.random(shots) < probabilities
        return dict(sorted(Counter("".join(map(str, row[: circuit.n_qubits])) for row in prefixes).items()))

    def run(
        self,
        circuits: QAOACircuit | Any | list[QAOACircuit | Any],
        *,
        shots: int = 4096,
        seed_simulator: int = 2026,
        fallback: str = "reject",
    ) -> QuTEJob:
        if shots <= 0:
            raise ValueError("shots must be positive")
        if fallback not in {"reject", "warn", "aer"}:
            raise ValueError("fallback must be reject, warn, or aer")
        items = circuits if isinstance(circuits, list) else [circuits]
        all_counts: list[dict[str, int]] = []
        all_metadata: list[dict[str, Any]] = []
        for index, raw in enumerate(items):
            total_start = time.perf_counter_ns()
            parse_start = total_start
            circuit = self._parse(raw)
            parse_ms = (time.perf_counter_ns() - parse_start) / 1e6
            support_start = time.perf_counter_ns()
            supported = self._supported(circuit)
            support_ms = (time.perf_counter_ns() - support_start) / 1e6
            if not supported:
                message = "circuit is outside the frozen six-qubit p=1 cycle-plus-chord envelope"
                if fallback == "aer":
                    raise QuTEUnsupportedCircuitError(message + "; Aer fallback is not packaged in the pilot")
                raise QuTEUnsupportedCircuitError(message)
            exact_before = _EXACT_CALLS
            encode_start = time.perf_counter_ns()
            _adjacency_and_degrees(circuit)
            encode_ms = (time.perf_counter_ns() - encode_start) / 1e6
            inference_start = time.perf_counter_ns()
            counts = self._sample(circuit, shots, seed_simulator + index)
            inference_ms = (time.perf_counter_ns() - inference_start) / 1e6
            package_start = time.perf_counter_ns()
            metadata = {
                "schema_version": "qute-result-v1",
                "model_hash": self.model_hash,
                "circuit_hash": circuit.circuit_hash,
                "graph_hash": circuit.graph_hash,
                "num_qubits": circuit.n_qubits,
                "qaoa_p": circuit.p,
                "support_score": 1.0,
                "estimated_error": None,
                "calibration_status": "UNAVAILABLE_PILOT",
                "execution_route": "qute",
                "shots": shots,
                "seed_simulator": seed_simulator + index,
                "per_circuit_optimizer_steps": 0,
                "exact_simulator_calls_on_neural_route": _EXACT_CALLS - exact_before,
            }
            package_ms = (time.perf_counter_ns() - package_start) / 1e6
            metadata["timing_ms"] = {
                "parse": parse_ms,
                "support": support_ms,
                "encode": encode_ms,
                "inference_and_sampling": inference_ms,
                "packaging": package_ms,
                "total": (time.perf_counter_ns() - total_start) / 1e6,
            }
            all_counts.append(counts)
            all_metadata.append(metadata)
        return QuTEJob(QuTEResult(all_counts, all_metadata))


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def exact_call_count() -> int:
    return _EXACT_CALLS
