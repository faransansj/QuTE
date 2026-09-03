import math

import pytest

from qute_r2.circuit_ir import CircuitIR, GateIR, bell_ir, ghz_ir


def test_circuit_ir_hash_is_stable() -> None:
    circuit = CircuitIR(2, (GateIR("h", (0,)), GateIR("cx", (0, 1)), GateIR("measure", (0, 1))))
    assert circuit.circuit_hash() == CircuitIR.from_dict(circuit.to_dict()).circuit_hash()
    assert circuit.canonical_json() == '{"gates":[{"name":"h","params":[],"qubits":[0]},{"name":"cx","params":[],"qubits":[0,1]},{"name":"measure","params":[],"qubits":[0,1]}],"n_qubits":2}'


def test_builtin_bell_and_ghz_shapes() -> None:
    assert [gate.name for gate in bell_ir().gates] == ["h", "cx", "measure"]
    assert [gate.name for gate in ghz_ir(4).gates] == ["h", "cx", "cx", "cx", "measure"]


def test_validation_rejects_bad_ir() -> None:
    with pytest.raises(ValueError):
        CircuitIR(1, (GateIR("cx", (0, 1)),))
    with pytest.raises(ValueError):
        GateIR("rx", (0,))


def test_qiskit_roundtrip_when_available() -> None:
    pytest.importorskip("qiskit")
    from qute_r2.qiskit_adapter import from_qiskit, to_qiskit

    circuit = CircuitIR(2, (GateIR("h", (0,)), GateIR("rzz", (0, 1), (math.pi / 3,)), GateIR("measure", (0, 1))))
    assert from_qiskit(to_qiskit(circuit)).circuit_hash() == circuit.circuit_hash()
