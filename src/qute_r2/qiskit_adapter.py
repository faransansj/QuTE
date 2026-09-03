from __future__ import annotations

from qute_r2.circuit_ir import CircuitIR, GateIR


def to_qiskit(ir: CircuitIR, measure: bool = True):
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(ir.n_qubits, ir.n_qubits if measure else 0)
    for gate in ir.gates:
        if gate.name == "measure":
            if measure:
                circuit.measure(list(gate.qubits), list(gate.qubits))
        elif gate.name == "h":
            circuit.h(gate.qubits[0])
        elif gate.name == "x":
            circuit.x(gate.qubits[0])
        elif gate.name == "rx":
            circuit.rx(gate.params[0], gate.qubits[0])
        elif gate.name == "ry":
            circuit.ry(gate.params[0], gate.qubits[0])
        elif gate.name == "rz":
            circuit.rz(gate.params[0], gate.qubits[0])
        elif gate.name == "cx":
            circuit.cx(*gate.qubits)
        elif gate.name == "cz":
            circuit.cz(*gate.qubits)
        elif gate.name == "rzz":
            circuit.rzz(gate.params[0], *gate.qubits)
    circuit.metadata = {"circuit_hash": ir.circuit_hash(), **(ir.metadata or {})}
    return circuit


def from_qiskit(circuit) -> CircuitIR:
    gates: list[GateIR] = []
    pending_measure: list[int] = []

    def flush_measure() -> None:
        if pending_measure:
            gates.append(GateIR("measure", tuple(pending_measure)))
            pending_measure.clear()

    for instruction in circuit.data:
        op = instruction.operation
        name = op.name.lower()
        if name == "barrier":
            continue
        qubits = tuple(circuit.find_bit(q).index for q in instruction.qubits)
        if name == "measure":
            pending_measure.extend(qubits)
        elif name in {"h", "x", "cx", "cz"}:
            flush_measure(); gates.append(GateIR(name, qubits))
        elif name in {"rx", "ry", "rz", "rzz"}:
            flush_measure(); gates.append(GateIR(name, qubits, (float(op.params[0]),)))
        else:
            raise ValueError(f"unsupported qiskit gate: {name}")
    flush_measure()
    return CircuitIR(circuit.num_qubits, tuple(gates), dict(circuit.metadata or {}))
