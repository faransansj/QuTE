import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cc_nqe import ACCEL, ACCEL_DEVICE, DIM, Gate, accel_synchronize, circuit_unitary, generate_state
from cc_nqe_p4_5 import _tensorize_circuit, parameter_count
from cc_nqe_p4_6 import CheckpointSelector, process_fidelity, raw_unitarity_error
from cc_nqe_p4_7 import (
    ANCHOR_COMMIT, ANCHOR_FILES, CONFIRM_SCHEMA, ROOT, SCHEMA, VARIANTS,
    RecursiveOperatorModel, aggregate_confirmatory, assert_validation_split,
    composition_loss, confirm_all, confirmatory_config, confirmatory_model,
    deterministic_split, exact_prefix_actions, initialization_digest,
    loss_for_batch, prefix_action_loss, prepare_artifacts,
    prepare_confirmatory_artifacts, split_circuits, status,
    validate_checkpoint, validate_confirmatory_checkpoint, variant_config,
    verify_anchor, verify_c0_confirmatory_config,
)
from run_p4_7 import build_parser


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


def test_confirmatory_cli_parsing_and_seed_gate():
    parser = build_parser()
    args = parser.parse_args(["confirm", "--seed", "2027"])
    assert (args.command, args.seed) == ("confirm", 2027)
    assert parser.parse_args(["confirm-all"]).command == "confirm-all"
    with pytest.raises(SystemExit):
        parser.parse_args(["confirm", "--seed", "2026"])


def test_c0_fresh_training_config_equivalence_and_seed_control():
    first = confirmatory_config("C0", 2027)
    second = confirmatory_config("C0", 2028)
    verify_c0_confirmatory_config(first)
    assert first["recipe"]["seed"] == 2027 and second["recipe"]["seed"] == 2028
    assert parameter_count(confirmatory_model("C0")) == 1_073_312
    assert initialization_digest("C0", 2027) == initialization_digest("C0", 2027)
    assert initialization_digest("C0", 2027) != initialization_digest("C0", 2028)


def test_confirmatory_checkpoint_identity_and_resume_rejection():
    config = confirmatory_config("C1", 2027)
    payload = {
        "variant": "C1", "seed": 2027, "run_kind": "confirmatory",
        "config_hash": __import__("cc_nqe_p4_6").digest(config), "dataset_manifest_hash": "data",
        "model": {}, "optimizer": {}, "scheduler": {}, "step": 1, "samples_seen": 1024,
        "numpy_rng": {}, "torch_rng": torch.get_rng_state(), "xpu_rng": None,
        "best_checkpoint_state": {"score": .5, "step": 1},
    }
    validate_confirmatory_checkpoint(payload, config, "data")
    with pytest.raises(ValueError, match="seed"):
        validate_confirmatory_checkpoint({**payload, "seed": 2028}, config, "data")
    with pytest.raises(ValueError, match="config"):
        validate_confirmatory_checkpoint(payload, confirmatory_config("C2", 2027), "data")
    with pytest.raises(ValueError, match="dataset"):
        validate_confirmatory_checkpoint(payload, config, "other")


def test_confirmatory_artifact_namespace_preserves_screening(tmp_path, monkeypatch):
    import cc_nqe_p4_7 as p47
    before = {variant: (ROOT / f"metrics/{variant}.json").read_bytes() for variant in VARIANTS}
    monkeypatch.setattr(p47, "CONFIRM_ROOT", tmp_path / "confirmatory")
    summary = prepare_confirmatory_artifacts()
    assert summary["scientific_runs"] == {f"{variant}-seed{seed}": "NOT_RUN" for seed in (2027, 2028) for variant in ("C0", "C1", "C2", "C3")}
    assert (tmp_path / "confirmatory/configs/C0-seed2027.json").exists()
    assert before == {variant: (ROOT / f"metrics/{variant}.json").read_bytes() for variant in VARIANTS}


def test_completed_confirmatory_run_is_not_reexecuted(tmp_path, monkeypatch):
    import cc_nqe_p4_7 as p47
    monkeypatch.setattr(p47, "CONFIRM_ROOT", tmp_path)
    monkeypatch.setattr(p47, "confirmatory_preflight", lambda: {"status": "PASS"})
    monkeypatch.setattr(p47, "prepare_confirmatory_artifacts", lambda: {})
    (tmp_path / "metrics").mkdir()
    completed = {"variant": "C2", "seed": 2027, "run_kind": "confirmatory", "state": "COMPLETED"}
    (tmp_path / "metrics/C2-seed2027.json").write_text(json.dumps(completed))
    monkeypatch.setattr(p47, "confirmatory_model", lambda *_: pytest.fail("completed run was reinitialized"))
    assert p47.confirm_run("C2", 2027) == completed


def test_confirm_all_orders_runs_and_skips_are_delegated(monkeypatch):
    import cc_nqe_p4_7 as p47
    calls = []
    def fake_confirm(seed):
        calls.append(seed)
        return {variant: {"state": "COMPLETED"} for variant in ("C0", "C1", "C2", "C3")}
    monkeypatch.setattr(p47, "confirm", fake_confirm)
    monkeypatch.setattr(p47, "aggregate_confirmatory", lambda: {"state": "COMPLETE"})
    result = confirm_all()
    assert calls == [2027, 2028] and result["aggregate"]["state"] == "COMPLETE"


def _metric(variant, seed, base):
    validation = {split: {"normalized_action_fidelity": base + offset} for split, offset in (
        ("iid_validation", .01), ("state_ood_validation", .02), ("parameter_ood_validation", .03),
        ("composition_ood_validation", .04), ("depth_ood_validation", .05))}
    return {"variant": variant, "seed": seed, "run_kind": "confirmatory", "state": "COMPLETED",
            "best_balanced_validation": base, "validation_at_best_balanced_checkpoint": validation}


def test_aggregation_three_seeds_and_paired_deltas(tmp_path, monkeypatch):
    import cc_nqe_p4_7 as p47
    root = tmp_path / "p47"; confirm_root = root / "confirmatory"
    (root / "metrics").mkdir(parents=True); (confirm_root / "metrics").mkdir(parents=True)
    c0_2026 = _metric("C0", 2026, .50)
    c0_2026.pop("run_kind")
    c0_2026["latest_validation"] = {k: {"predicted_operator_state_fidelity": v["normalized_action_fidelity"]} for k, v in c0_2026.pop("validation_at_best_balanced_checkpoint").items()}
    anchor_path = tmp_path / "B3.json"; anchor_path.write_text(json.dumps(c0_2026))
    monkeypatch.setitem(p47.ANCHOR_FILES, "metric", (anchor_path, "unused"))
    monkeypatch.setattr(p47, "ROOT", root); monkeypatch.setattr(p47, "CONFIRM_ROOT", confirm_root)
    for variant_index, variant in enumerate(("C1", "C2", "C3"), 1):
        screening = _metric(variant, 2026, .50 + variant_index * .01)
        (root / f"metrics/{variant}.json").write_text(json.dumps(screening))
    for seed_index, seed in enumerate((2027, 2028), 1):
        for variant_index, variant in enumerate(("C0", "C1", "C2", "C3")):
            (confirm_root / f"metrics/{variant}-seed{seed}.json").write_text(json.dumps(_metric(variant, seed, .50 + variant_index * .01 + seed_index * .001)))
    result = aggregate_confirmatory()
    delta = result["paired_deltas"]["C1_minus_C0"]["S_balanced"]
    assert result["state"] == "COMPLETE" and delta["mean_delta"] == pytest.approx(.01)
    assert delta["sign_consistency"] == "consistent" and set(delta["per_seed"]) == {"2026", "2027", "2028"}


def test_confirmatory_status_schema_and_sealed_zero(tmp_path, monkeypatch):
    import cc_nqe_p4_7 as p47
    root = tmp_path / "p47"; confirm_root = root / "confirmatory"
    (root / "metrics").mkdir(parents=True); (confirm_root / "metrics").mkdir(parents=True)
    (root / "metrics/C1.json").write_text(json.dumps({"state": "COMPLETED"}))
    (confirm_root / "metrics/C0-seed2027.json").write_text(json.dumps({"state": "COMPLETED", "step": 10000}))
    (confirm_root / "status.json").write_text(json.dumps({"variant": "C0", "seed": 2027, "state": "RUNNING", "step": 50, "maximum_updates": 10000}))
    monkeypatch.setattr(p47, "ROOT", root); monkeypatch.setattr(p47, "CONFIRM_ROOT", confirm_root)
    value = status()
    assert value["screening_runs"]["C0"] == "COMPLETED"
    assert value["confirmatory_runs"]["C0-seed2027"] == "COMPLETED"
    assert value["confirmatory_runs"]["C1-seed2028"] == "NOT_RUN" and value["sealed_test_access_count"] == 0
    assert value["state"] == "COMPLETED" and value["step"] == 10000


@pytest.mark.skipif(ACCEL == "cpu", reason="no native accelerator")
def test_xpu_native_cayley_forward_backward_no_cpu_fallback():
    model = RecursiveOperatorModel(width=24).to(ACCEL_DEVICE)
    args = tuple(value.to(ACCEL_DEVICE) for value in circuit_batch([[Gate("H", (0,)), Gate("CNOT", (0, 1))]], 2))
    operator = model(*args)
    loss = raw_unitarity_error(operator).mean() + (1 - process_fidelity(operator, torch.eye(DIM, dtype=operator.dtype, device=ACCEL_DEVICE)[None])).mean()
    loss.backward(); accel_synchronize()
    assert operator.device.type == ACCEL and raw_unitarity_error(operator).max() < 1e-4
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
