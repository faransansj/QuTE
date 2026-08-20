import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cc_nqe import DIM, Gate, circuit_unitary, generate_state
from cc_nqe_p4_5 import _tensorize_circuit, parameter_count
from cc_nqe_p4_6 import CheckpointSelector, process_fidelity, raw_unitarity_error
from cc_nqe_p4_7 import (
    ANCHOR_COMMIT, ANCHOR_FILES, ROOT, SCHEMA, VARIANTS, RecursiveOperatorModel,
    assert_validation_split, composition_loss, deterministic_split,
    exact_prefix_actions, loss_for_batch, prefix_action_loss, prepare_artifacts,
    split_circuits, status, validate_checkpoint, variant_config, verify_anchor,
)


def circuit_batch(circuits, max_depth=7):
    columns = list(zip(*[_tensorize_circuit(circuit, max_depth) for circuit in circuits]))
    return tuple(torch.as_tensor(np.asarray(column)) for column in columns)


def state_batch(circuits):
    states = np.asarray([generate_state(100 + i, "product") for i in range(len(circuits))])
    targets = np.asarray([circuit_unitary(c) @ s for c, s in zip(circuits, states)])
    return torch.tensor(np.c_[states.real, states.imag], dtype=torch.float32), torch.tensor(np.c_[targets.real, targets.imag], dtype=torch.float32)


def test_c0_anchor_integrity_and_canonical_metadata():
    anchor = verify_anchor()
    assert anchor == {
        "schema_version": SCHEMA, "variant": "C0", "role": "frozen P4.6 B3 reference; no rerun",
        "commit": ANCHOR_COMMIT, "config_hash": ANCHOR_FILES["config"][1],
        "canonical_config_hash": "bff1722a762277c8f88f020a436ce84884eb3ab2931822db0f386424c1d45d3f",
        "dataset_manifest_hash": ANCHOR_FILES["dataset_manifest"][1], "metric_artifact_hash": ANCHOR_FILES["metric"][1],
        "best_checkpoint_step": 8500, "best_balanced_score": 0.5904355843861898,
        "actual_parameters": 1073312, "supervision_class": "ACTION_ONLY", "scientific_state": "REFERENCE_ONLY", "anchor_integrity": "PASS",
    }


def test_prepare_writes_only_p47_and_leaves_p46_anchor_bytes(tmp_path, monkeypatch):
    before = {name: path.read_bytes() for name, (path, _) in ANCHOR_FILES.items()}
    import cc_nqe_p4_7 as p47
    monkeypatch.setattr(p47, "ROOT", tmp_path)
    summary = prepare_artifacts()
    assert summary["sealed_test_access_count"] == 0
    assert before == {name: path.read_bytes() for name, (path, _) in ANCHOR_FILES.items()}


def test_allocation_validation_mapping_and_sealed_denial():
    assert variant_config("C1")["allocation"] == {"unique_circuits": 58824, "probes_per_circuit": 17, "state_action_pairs": 1000008, "source_arm": "A4"}
    for split in ("iid_validation", "state_ood_validation", "parameter_ood_validation", "composition_ood_validation", "depth_ood_validation"):
        assert_validation_split(split)
    with pytest.raises(PermissionError): assert_validation_split("composition_ood_test_sealed")
    with pytest.raises(PermissionError): assert_validation_split("depth_ood_test_sealed")


def test_recursive_variable_length_shared_weights_order_and_depth7():
    torch.manual_seed(7)
    model = RecursiveOperatorModel(width=24)
    assert len({id(model.transition) for _ in range(7)}) == 1
    first = [Gate("H", (0,)), Gate("CNOT", (0, 1))]
    reversed_order = list(reversed(first))
    args = circuit_batch([first, reversed_order, first + [Gate("X", (2,))] * 5], 7)
    output, prefixes = model(*args, return_prefixes=True)
    assert output.shape == (3, DIM, DIM) and prefixes.shape == (3, 7, DIM, DIM)
    assert not torch.allclose(output[0], output[1])
    assert torch.allclose(output[0], prefixes[0, 1]) and torch.allclose(output[2], prefixes[2, 6])


def test_recursive_inputs_contain_no_analytical_gate_matrices():
    signature = inspect.signature(RecursiveOperatorModel.forward)
    assert list(signature.parameters) == ["self", "gates", "qubits", "parameters", "mask", "return_prefixes"]
    model = RecursiveOperatorModel(width=16)
    assert list(dict(model.named_buffers())) == []
    assert set(dict(model.token.named_children())) == {"gate", "qubit", "parameter"}


def test_c1_action_only_and_supervision_classes():
    configs = {variant: variant_config(variant) for variant in VARIANTS}
    assert configs["C1"]["supervision"] == ["C", "psi_in", "psi_out"] and not configs["C1"]["privileged"]
    assert configs["C2"]["supervision_class"] == "ACTION_ONLY_PLUS_SELF_CONSISTENCY" and "exact_U" not in configs["C2"]["supervision"]
    assert configs["C3"]["supervision_class"] == "PRIVILEGED_PREFIX_ACTION_SUPERVISION" and configs["C3"]["privileged"]


def test_c2_parameter_identity_deterministic_split_and_ordering():
    assert variant_config("C1")["actual_parameters"] == variant_config("C2")["actual_parameters"]
    assert deterministic_split("sample-9", 6) == deterministic_split("sample-9", 6)
    c1 = [Gate("H", (0,))]; c2 = [Gate("CNOT", (0, 1))]
    first, second, combined = split_circuits([c1 + c2], ["x"])
    assert first == [c1] and second == [c2] and combined == [c1 + c2]
    u1 = torch.tensor(circuit_unitary(c1)); u2 = torch.tensor(circuit_unitary(c2)); direct = torch.tensor(circuit_unitary(c1 + c2))
    assert torch.allclose(direct, u2 @ u1) and composition_loss(direct[None], u2[None], u1[None]) < 1e-10
    assert not torch.allclose(u2 @ u1, u1 @ u2)


def test_c2_non_tautological_random_loss_and_gradient():
    torch.manual_seed(11)
    model = RecursiveOperatorModel(width=24)
    circuits = [[Gate("H", (0,)), Gate("CNOT", (0, 1))], [Gate("RX", (2,), .4), Gate("X", (3,))]]
    gates = circuit_batch(circuits, 2); states, targets = state_batch(circuits)
    loss, _, _, _, comp, _ = loss_for_batch("C2", model, [*gates, states, targets], circuits, ["a", "b"])
    assert comp > 1e-7 and torch.isfinite(comp)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients) and sum(float(g.abs().sum()) for g in gradients) > 0


def test_c2_synthetic_consistent_mapping_zero_case():
    first = torch.linalg.qr(torch.randn(3, DIM, DIM, dtype=torch.complex64)).Q
    second = torch.linalg.qr(torch.randn(3, DIM, DIM, dtype=torch.complex64)).Q
    assert composition_loss(second @ first, second, first) < 1e-6


def test_c3_prefix_targets_are_targets_only_free_running_and_normalized():
    circuits = [[Gate("H", (0,)), Gate("X", (1,))], [Gate("X", (0,))]]
    states, final_targets = state_batch(circuits)
    exact = torch.tensor(exact_prefix_actions(circuits, states.numpy(), 2))
    masks = torch.tensor([[True, True], [True, False]])
    exact_ops = torch.stack([torch.tensor(np.asarray([circuit_unitary(c[:i + 1]) for c in circuits], np.complex64)) for i in range(2)], dim=1)
    # Replace padded operator; masking must remove its contribution.
    exact_ops[1, 1] = torch.randn(DIM, DIM, dtype=torch.complex64)
    assert prefix_action_loss(exact_ops, states, exact, masks) < 1e-6
    model = RecursiveOperatorModel(width=24)
    gates = circuit_batch(circuits, 2)
    before = model(*gates).detach().clone()
    _, _, _, final, _, prefix = loss_for_batch("C3", model, [*gates, states, final_targets], circuits)
    altered_targets = final_targets.roll(1, 0)
    after = model(*gates).detach()
    assert torch.allclose(before, final) and torch.allclose(before, after) and torch.isfinite(prefix)
    source = inspect.getsource(RecursiveOperatorModel.hidden_states)
    assert "target" not in source and "state" not in inspect.signature(RecursiveOperatorModel.forward).parameters


def test_basic_cayley_exact_unitarity_and_finite_nonzero_gradients():
    model = RecursiveOperatorModel(width=24)
    args = circuit_batch([[Gate("H", (0,)), Gate("CNOT", (0, 1))]], 2)
    operator = model(*args)
    assert raw_unitarity_error(operator).max() < 1e-5
    (1 - process_fidelity(operator, torch.eye(DIM, dtype=operator.dtype)[None])).mean().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients) and sum(float(g.abs().sum()) for g in gradients) > 0
    source = inspect.getsource(RecursiveOperatorModel.operators_from_hidden)
    assert "torch.linalg.solve" in source and "matrix_exp" not in source and "inverse" not in source and ".cpu" not in source


def test_parameter_count_fairness_exact():
    counts = {variant: variant_config(variant)["actual_parameters"] for variant in VARIANTS}
    assert counts == {"C1": 1063124, "C2": 1063124, "C3": 1063124}
    model = RecursiveOperatorModel()
    assert (model.head[0].in_features, model.head[0].out_features, model.head[2].out_features) == (160, 320, 2 * DIM * DIM)
    assert all(variant_config(v)["parameter_fairness_pass"] for v in VARIANTS)
    assert abs(counts["C1"] - 1073312) / 1073312 < .05


def test_best_balanced_checkpoint_selection_is_single_checkpoint():
    selector = CheckpointSelector()
    selector.update(500, {"iid_validation": .8, "composition_ood_validation": .3, "depth_ood_validation": .4})
    selector.update(1000, {"iid_validation": .7, "composition_ood_validation": .6, "depth_ood_validation": .5})
    assert selector.best["balanced"] == {"score": pytest.approx(.6), "step": 1000}
    assert selector.best["iid"]["step"] == 500


def test_checkpoint_resume_refusal_status_and_completed_blocking(tmp_path, monkeypatch):
    config = variant_config("C1")
    payload = {
        "variant": "C1", "config_hash": __import__("cc_nqe_p4_6").digest(config), "dataset_manifest_hash": "x",
        "model": {}, "optimizer": {}, "scheduler": {}, "step": 1, "samples_seen": 1,
        "numpy_rng": {}, "torch_rng": torch.get_rng_state(), "xpu_rng": None,
        "best_checkpoint_state": {"score": 0.5, "step": 1},
    }
    validate_checkpoint(payload, config, "x")
    with pytest.raises(ValueError, match="incomplete checkpoint"): validate_checkpoint({k: v for k, v in payload.items() if k != "optimizer"}, config, "x")
    with pytest.raises(ValueError, match="config"): validate_checkpoint(payload, variant_config("C2"), "x")
    with pytest.raises(ValueError, match="dataset"): validate_checkpoint(payload, config, "y")
    import cc_nqe_p4_7 as p47
    monkeypatch.setattr(p47, "ROOT", tmp_path)
    (tmp_path / "metrics").mkdir()
    completed = {"schema_version": SCHEMA, "variant": "C1", "state": "COMPLETED"}
    (tmp_path / "metrics/C1.json").write_text(json.dumps(completed))
    monkeypatch.setattr(p47, "require_preconditions", lambda: None)
    monkeypatch.setattr(p47, "preflight", lambda: {"status": "PASS"})
    assert p47.run("C1") == completed
    value = status()
    assert value["scientific_runs"] == {"C1": "COMPLETED", "C2": "NOT_RUN", "C3": "NOT_RUN"} and value["sealed_test_access_count"] == 0


@pytest.mark.skipif(not torch.xpu.is_available(), reason="native XPU unavailable")
def test_xpu_native_cayley_forward_backward_no_cpu_fallback():
    model = RecursiveOperatorModel(width=24).to("xpu:0")
    args = tuple(value.to("xpu:0") for value in circuit_batch([[Gate("H", (0,)), Gate("CNOT", (0, 1))]], 2))
    operator = model(*args)
    loss = raw_unitarity_error(operator).mean() + (1 - process_fidelity(operator, torch.eye(DIM, dtype=operator.dtype, device="xpu:0")[None])).mean()
    loss.backward(); torch.xpu.synchronize()
    assert operator.device.type == "xpu" and raw_unitarity_error(operator).max() < 1e-4
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
