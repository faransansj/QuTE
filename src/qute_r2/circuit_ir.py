from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

SUPPORTED_GATES = {"h", "x", "rx", "ry", "rz", "cx", "cz", "rzz", "measure"}


@dataclass(frozen=True)
class GateIR:
    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        name = self.name.lower()
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "qubits", tuple(self.qubits))
        object.__setattr__(self, "params", tuple(float(x) for x in self.params))
        if name not in SUPPORTED_GATES:
            raise ValueError(f"unsupported gate: {name}")
        if name in {"h", "x", "rx", "ry", "rz"} and len(self.qubits) != 1:
            raise ValueError(f"{name} expects 1 qubit")
        if name in {"cx", "cz", "rzz"} and len(self.qubits) != 2:
            raise ValueError(f"{name} expects 2 qubits")
        if name == "measure" and not self.qubits:
            raise ValueError("measure expects at least 1 qubit")
        if name in {"rx", "ry", "rz", "rzz"} and len(self.params) != 1:
            raise ValueError(f"{name} expects 1 parameter")
        if name not in {"rx", "ry", "rz", "rzz"} and self.params:
            raise ValueError(f"{name} expects no parameters")


@dataclass(frozen=True)
class CircuitIR:
    n_qubits: int
    gates: tuple[GateIR, ...]
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "gates", tuple(self.gates))
        if self.n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        for gate in self.gates:
            if any(q < 0 or q >= self.n_qubits for q in gate.qubits):
                raise ValueError(f"gate qubit outside circuit width: {gate}")

    def to_dict(self) -> dict[str, Any]:
        data = {"n_qubits": self.n_qubits, "gates": [asdict(gate) for gate in self.gates]}
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CircuitIR":
        return cls(
            n_qubits=int(data["n_qubits"]),
            gates=tuple(GateIR(g["name"], tuple(g["qubits"]), tuple(g.get("params", ()))) for g in data["gates"]),
            metadata=data.get("metadata"),
        )

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def circuit_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


def bell_ir() -> CircuitIR:
    return CircuitIR(2, (GateIR("h", (0,)), GateIR("cx", (0, 1)), GateIR("measure", (0, 1))))


def ghz_ir(n_qubits: int) -> CircuitIR:
    return CircuitIR(n_qubits, (GateIR("h", (0,)), *(GateIR("cx", (i, i + 1)) for i in range(n_qubits - 1)), GateIR("measure", tuple(range(n_qubits)))))
