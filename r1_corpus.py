"""Deterministic R1 semantic-pair corpus generator and exact oracle audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from cc_nqe import Gate, circuit_id, circuit_unitary, generate_circuit


SCHEMA_VERSION = "qute-r1-corpus-v1"
SEEN_FAMILIES = ("inverse_cancellation", "commuting_reorder", "rotation_fusion_split")
HELD_OUT_SPLITS = {
    "rewrite_family_ood": "identity_insertion_removal",
    "cross_decomposition_ood": "cross_decomposition",
    "nonlocal_semantics_ood": "nonlocal_operator_semantics",
}


def _json(value: Any, *, pretty: bool = False) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _seed(protocol_hash: str, *parts: object) -> int:
    value = "||".join((protocol_hash, *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _gate_dicts(circuit: Iterable[Gate]) -> list[dict[str, Any]]:
    return [gate.to_dict() for gate in circuit]


def _phase_aligned_errors(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
    overlap = np.vdot(left, right)
    phase = overlap / abs(overlap) if abs(overlap) else 1.0 + 0.0j
    aligned = right * phase.conjugate()
    relative_frobenius = float(np.linalg.norm(left - aligned) / np.linalg.norm(left))
    maximum_probe_l2 = float(np.max(np.linalg.norm(left - aligned, axis=0)))
    left_probabilities = np.abs(left) ** 2
    right_probabilities = np.abs(right) ** 2
    maximum_probability_tvd = float(np.max(0.5 * np.sum(np.abs(left_probabilities - right_probabilities), axis=0)))
    return relative_frobenius, maximum_probe_l2, maximum_probability_tvd


def _operator_hash(unitary: np.ndarray) -> str:
    flat = unitary.ravel()
    pivot = next(value for value in flat if abs(value) > 1e-14)
    canonical = unitary * np.exp(-1j * np.angle(pivot))
    rounded = np.stack((canonical.real.round(10), canonical.imag.round(10)), axis=-1)
    rounded[rounded == 0] = 0.0  # canonicalize negative zero before hashing
    return _sha256_bytes(rounded.astype("<f8", copy=False).tobytes())


def _parameter_angle(protocol: dict[str, Any], region: str, seed: int) -> float:
    ranges = protocol["scope"]["parameter_regions_radians"][region]
    low, high = ranges[seed % len(ranges)]
    fraction = ((seed >> 8) % 1_000_003 + 0.5) / 1_000_003
    return float(low + (high - low) * fraction)


def _qubits(seed: int) -> tuple[int, int, int, int]:
    order = list(range(4))
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    return tuple(order)  # type: ignore[return-value]


def _insert(base: list[Gate], gates: list[Gate], seed: int) -> list[Gate]:
    position = seed % (len(base) + 1)
    return [*base[:position], *gates, *base[position:]]


def _equivalent_pair(
    base: list[Gate], family: str, protocol: dict[str, Any], seed: int, *, instance_ood: bool = False, parameter_region: str = "train"
) -> tuple[list[Gate], list[Gate], str]:
    q0, q1, q2, _ = _qubits(seed)
    angle_a = _parameter_angle(protocol, parameter_region, seed)
    angle_b = _parameter_angle(protocol, parameter_region, seed ^ 0x9E3779B97F4A7C15)

    if family == "inverse_cancellation":
        if instance_ood:
            identity = [Gate("CNOT", (q0, q1)), Gate("CNOT", (q0, q1))]
            template = "inverse_cnot"
        else:
            choice = seed % 3
            if choice == 0:
                identity, template = [Gate("H", (q0,)), Gate("H", (q0,))], "inverse_h"
            elif choice == 1:
                identity, template = [Gate("X", (q0,)), Gate("X", (q0,))], "inverse_x"
            else:
                identity = [Gate("RY", (q0,), angle_a), Gate("RY", (q0,), -angle_a)]
                template = "inverse_rotation"
        return base, _insert(base, identity, seed), template

    if family == "commuting_reorder":
        if instance_ood:
            first, second = Gate("CNOT", (q0, q1)), Gate("H", (q2,))
            template = "commute_cnot_disjoint_h"
        else:
            first, second = Gate("H", (q0,)), Gate("RZ", (q1,), angle_a)
            template = "commute_disjoint_single"
        position = seed % (len(base) + 1)
        prefix, suffix = base[:position], base[position:]
        return [*prefix, first, second, *suffix], [*prefix, second, first, *suffix], template

    if family == "rotation_fusion_split":
        first = Gate("RZ", (q0,), angle_a)
        second = Gate("RZ", (q0,), angle_b)
        fused = Gate("RZ", (q0,), (angle_a + angle_b) % (2 * math.pi))
        position = seed % (len(base) + 1)
        prefix, suffix = base[:position], base[position:]
        split, compact = [*prefix, first, second, *suffix], [*prefix, fused, *suffix]
        return (compact, split, "rotation_split_heldout_direction") if instance_ood else (split, compact, "rotation_fusion")

    if family == "identity_insertion_removal":
        identity = [Gate("H", (q0,)), Gate("X", (q0,)), Gate("X", (q0,)), Gate("H", (q0,))]
        return base, _insert(base, identity, seed), "identity_hxxh"

    if family == "cross_decomposition":
        direct = [Gate("CNOT", (q0, q1))]
        reverse = [Gate("H", (q0,)), Gate("H", (q1,)), Gate("CNOT", (q1, q0)), Gate("H", (q0,)), Gate("H", (q1,))]
        return _insert(base, direct, seed), _insert(base, reverse, seed), "reverse_direction_cnot"

    if family == "nonlocal_operator_semantics":
        first = [Gate("CNOT", (q0, q1)), Gate("CNOT", (q1, q0)), Gate("CNOT", (q0, q1))]
        second = [Gate("CNOT", (q1, q0)), Gate("CNOT", (q0, q1)), Gate("CNOT", (q1, q0))]
        return _insert(base, first, seed), _insert(base, second, seed), "swap_three_cnot"

    raise ValueError(f"unknown rewrite family: {family}")


def _mutations(circuit: list[Gate]) -> Iterable[tuple[list[Gate], str]]:
    for index, gate in enumerate(circuit):
        changed = list(circuit)
        if gate.name in {"RX", "RY", "RZ"}:
            changed[index] = Gate(gate.name, gate.qubits, (float(gate.theta) + math.pi / 2) % (2 * math.pi))
            yield changed, "parameter_perturbation"
        elif gate.name == "CNOT":
            changed[index] = Gate("CNOT", (gate.qubits[1], gate.qubits[0]))
            yield changed, "uncompensated_control_target_reversal"
        elif gate.name in {"H", "X"}:
            changed[index] = Gate("X" if gate.name == "H" else "H", gate.qubits)
            yield changed, "gate_substitution"
    yield [*circuit, Gate("X", (0,))], "operator_changing_insertion"


def _negative_right(left_unitary: np.ndarray, right: list[Gate]) -> tuple[list[Gate], str, float, np.ndarray]:
    for candidate, control_type in _mutations(right):
        candidate_unitary = circuit_unitary(candidate)
        distance = _phase_aligned_errors(left_unitary, candidate_unitary)[0]
        if distance >= 0.1:
            return candidate, control_type, distance, candidate_unitary
    raise RuntimeError("could not construct a non-equivalent control")


def _plans(protocol: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []

    def add(partition: str, split: str, family: str, depth: int, count: int, *, region: str = "train", instance_ood: bool = False) -> None:
        for local_index in range(count):
            plans.append({
                "partition": partition,
                "split": split,
                "family": family,
                "base_depth": depth,
                "parameter_region": region,
                "instance_ood": instance_ood,
                "local_index": local_index,
            })

    if mode == "smoke":
        for partition in ("train", "development"):
            for family in SEEN_FAMILIES:
                add(partition, partition, family, 2, 1)
        for family in SEEN_FAMILIES:
            add("final_test", "semantic_iid", family, 2, 1)
            add("final_test", "rewrite_instance_ood", family, 4, 1, instance_ood=True)
        for split, family in HELD_OUT_SPLITS.items():
            add("final_test", split, family, 6, 1)
        for region in ("interpolation_ood", "extrapolation_ood"):
            for family in SEEN_FAMILIES:
                add("final_test", "parameter_ood", family, 4, 1, region=region)
        for depth in (8, 10):
            for family in SEEN_FAMILIES:
                add("final_test", "depth_ood", family, depth, 1)
        return plans

    if mode != "full":
        raise ValueError("mode must be 'smoke' or 'full'")

    allocation = protocol["corpus_allocation"]
    for partition in ("train", "development"):
        values = allocation[partition]
        for family in values["families"]:
            for depth in values["depths"]:
                add(partition, partition, family, depth, values["pairs_per_family_depth"])

    final = allocation["final_test"]
    for family in SEEN_FAMILIES:
        for depth in (2, 4, 6):
            add("final_test", "semantic_iid", family, depth, final["semantic_iid"]["pairs_per_family_depth"])
            add("final_test", "rewrite_instance_ood", family, depth, final["rewrite_instance_ood"]["pairs_per_family_depth"], instance_ood=True)
    for split, family in HELD_OUT_SPLITS.items():
        for depth in (2, 4, 6):
            add("final_test", split, family, depth, final[split]["pairs_per_family_depth"])
    for region in ("interpolation_ood", "extrapolation_ood"):
        for family in SEEN_FAMILIES:
            for depth in (2, 4, 6):
                add("final_test", "parameter_ood", family, depth, final["parameter_ood"]["pairs_per_stratum"], region=region)
    for family in SEEN_FAMILIES:
        for depth in (8, 10):
            add("final_test", "depth_ood", family, depth, final["depth_ood"]["pairs_per_family_depth"])
    return plans


def _probe(protocol_hash: str, partition: str, base_id: str, index: int) -> dict[str, Any]:
    probe_seed = _seed(protocol_hash, partition, base_id, index)
    family = "product" if partition != "final_test" else ("entangled" if probe_seed % 2 == 0 else "haar")
    return {"probe_id": f"p_{probe_seed:016x}", "family": family, "seed": probe_seed, "index": index}


def _record_pair(
    protocol: dict[str, Any], protocol_hash: str, plan: dict[str, Any], global_index: int, attempt: int
) -> tuple[dict[str, Any], dict[str, Any], str, str, str]:
    item_seed = _seed(protocol_hash, plan["partition"], plan["split"], plan["family"], plan["base_depth"], plan["parameter_region"], plan["local_index"], attempt)
    regime = {"train": "train", "interpolation_ood": "interpolation", "extrapolation_ood": "extrapolation"}[plan["parameter_region"]]
    base = generate_circuit(item_seed, plan["base_depth"], regime=regime)
    base_id = circuit_id(base)
    left, right, template_id = _equivalent_pair(
        base,
        plan["family"],
        protocol,
        item_seed,
        instance_ood=plan["instance_ood"],
        parameter_region=plan["parameter_region"],
    )
    left_unitary, right_unitary = circuit_unitary(left), circuit_unitary(right)
    frobenius, probe_l2, probability_tvd = _phase_aligned_errors(left_unitary, right_unitary)
    tolerances = protocol["oracle"]["tolerances"]
    equivalent_pass = (
        frobenius <= tolerances["phase_aligned_relative_frobenius"]
        and probe_l2 <= tolerances["maximum_probe_l2"]
        and probability_tvd <= tolerances["maximum_probability_tvd"]
    )
    if not equivalent_pass:
        raise RuntimeError(f"oracle rejected {plan['family']}/{template_id}: {(frobenius, probe_l2, probability_tvd)}")

    left_hash, right_hash = _operator_hash(left_unitary), _operator_hash(right_unitary)
    if left_hash != right_hash:
        raise RuntimeError(f"canonical operator hash mismatch for {plan['family']}/{template_id}")

    negative_right, control_type, negative_distance, negative_unitary = _negative_right(left_unitary, right)
    negative_hash = _operator_hash(negative_unitary)
    pair_root = _sha256_bytes(f"{protocol_hash}|{plan['partition']}|{plan['split']}|{global_index}|{item_seed}".encode())[:20]
    common = {
        "schema_version": SCHEMA_VERSION,
        "partition": plan["partition"],
        "split": plan["split"],
        "rewrite_family": plan["family"],
        "template_id": template_id,
        "base_circuit_id": base_id,
        "base_depth": plan["base_depth"],
        "parameter_region": plan["parameter_region"],
        "generation_seed": item_seed,
        "protocol_sha256": protocol_hash,
        "probe": _probe(protocol_hash, plan["partition"], base_id, 0),
    }
    positive = {
        **common,
        "pair_id": f"r1p_{pair_root}",
        "label": "equivalent",
        "left_circuit_id": circuit_id(left),
        "right_circuit_id": circuit_id(right),
        "left_circuit": _gate_dicts(left),
        "right_circuit": _gate_dicts(right),
        "expanded_depth": {"left": len(left), "right": len(right)},
        "operator_hash": left_hash,
        "oracle": {
            "phase_aligned_relative_frobenius": frobenius,
            "maximum_probe_l2": probe_l2,
            "maximum_probability_tvd": probability_tvd,
            "symbolic_preconditions": True,
            "pass": True,
        },
    }
    negative = {
        **common,
        "pair_id": f"r1n_{pair_root}",
        "label": "non_equivalent",
        "control_type": control_type,
        "left_circuit_id": circuit_id(left),
        "right_circuit_id": circuit_id(negative_right),
        "left_circuit": _gate_dicts(left),
        "right_circuit": _gate_dicts(negative_right),
        "expanded_depth": {"left": len(left), "right": len(negative_right)},
        "left_operator_hash": left_hash,
        "right_operator_hash": negative_hash,
        "oracle": {"phase_aligned_relative_frobenius": negative_distance, "minimum_required": 0.1, "pass": negative_distance >= 0.1},
    }
    return positive, negative, base_id, left_hash, negative_hash


def build_corpus(protocol_path: str | Path, *, mode: str = "smoke", authorize_full: bool = False) -> dict[str, Any]:
    protocol_path = Path(protocol_path)
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol_hash = _sha256_bytes(protocol_bytes)
    if protocol["status"] != "FROZEN_NOT_RUN" or protocol["frozen"] is not True:
        raise ValueError("R1 protocol must be frozen and not run")
    if mode == "full" and not authorize_full:
        raise PermissionError("full corpus generation requires explicit authorize_full=True")

    plans = _plans(protocol, mode)
    records: dict[str, list[dict[str, Any]]] = {"train": [], "development": [], "final_test": []}
    base_ids: set[str] = set()
    operator_partitions: dict[str, set[str]] = defaultdict(set)
    maxima = {"phase_aligned_relative_frobenius": 0.0, "maximum_probe_l2": 0.0, "maximum_probability_tvd": 0.0}
    minimum_negative_distance = math.inf

    for global_index, plan in enumerate(plans):
        for attempt in range(10_000):
            positive, negative, base_id, positive_hash, negative_hash = _record_pair(protocol, protocol_hash, plan, global_index, attempt)
            partition = plan["partition"]
            if base_id in base_ids:
                continue
            if any(operator_partitions[value] - {partition} for value in (positive_hash, negative_hash)):
                continue
            break
        else:
            raise RuntimeError(f"could not produce leakage-free record for plan {plan}")
        base_ids.add(base_id)
        operator_partitions[positive_hash].add(partition)
        operator_partitions[negative_hash].add(partition)
        records[partition].extend((positive, negative))
        for key in maxima:
            maxima[key] = max(maxima[key], positive["oracle"][key])
        minimum_negative_distance = min(minimum_negative_distance, negative["oracle"]["phase_aligned_relative_frobenius"])

    counts = {
        partition: dict(sorted(Counter(record["label"] for record in values).items()))
        for partition, values in records.items()
    }
    expected = {
        partition: {"equivalent": sum(plan["partition"] == partition for plan in plans), "non_equivalent": sum(plan["partition"] == partition for plan in plans)}
        for partition in records
    }
    partition_leakage = sorted(operator_hash for operator_hash, partitions in operator_partitions.items() if len(partitions) > 1)
    pair_ids = [record["pair_id"] for values in records.values() for record in values]
    audit = {
        "schema_version": "qute-r1-corpus-audit-v1",
        "status": "PASS" if counts == expected and not partition_leakage and len(pair_ids) == len(set(pair_ids)) else "FAIL",
        "mode": mode,
        "protocol_sha256": protocol_hash,
        "counts": counts,
        "expected_counts": expected,
        "positive_oracle_failures": 0,
        "negative_oracle_failures": 0,
        "partition_leakage": partition_leakage,
        "duplicate_pair_ids": len(pair_ids) - len(set(pair_ids)),
        "maximum_positive_errors": maxima,
        "minimum_negative_distance": minimum_negative_distance,
        "final_test_scientific": mode == "full",
        "final_test_access_count": 0,
    }
    if audit["status"] != "PASS":
        raise RuntimeError(f"R1 corpus audit failed: {audit}")
    return {"schema_version": SCHEMA_VERSION, "mode": mode, "protocol_sha256": protocol_hash, "records": records, "audit": audit}


def write_corpus(
    output_dir: str | Path, protocol_path: str | Path, *, mode: str = "smoke", authorize_full: bool = False
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty corpus directory: {output_dir}")
    corpus = build_corpus(protocol_path, mode=mode, authorize_full=authorize_full)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_hashes: dict[str, str] = {}
    names = {"train": "train.jsonl", "development": "development.jsonl", "final_test": "final_test.smoke.jsonl" if mode == "smoke" else "final_test.sealed.jsonl"}
    for partition, filename in names.items():
        payload = "".join(_json(record) + "\n" for record in corpus["records"][partition])
        path = output_dir / filename
        path.write_text(payload)
        file_hashes[filename] = _sha256_file(path)

    audit_path = output_dir / "audit.json"
    audit_path.write_text(_json(corpus["audit"], pretty=True) + "\n")
    file_hashes[audit_path.name] = _sha256_file(audit_path)
    manifest = {
        "schema_version": "qute-r1-corpus-manifest-v1",
        "mode": mode,
        "protocol_sha256": corpus["protocol_sha256"],
        "counts": corpus["audit"]["counts"],
        "file_hashes": dict(sorted(file_hashes.items())),
        "scientific_final_test_generated": mode == "full",
        "final_test_access_count": 0,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(_json(manifest, pretty=True) + "\n")
    file_hashes[manifest_path.name] = _sha256_file(manifest_path)
    checksum_path = output_dir / "checksums.sha256"
    checksum_path.write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(file_hashes.items())))
    if mode == "full":
        (output_dir / "FINAL_TEST_ACCESS_LOG.jsonl").write_text("")
    return manifest


def load_partition(root: str | Path, partition: str, *, allow_final: bool = False) -> list[dict[str, Any]]:
    root = Path(root)
    if partition == "final_test" and not allow_final:
        raise PermissionError("ordinary development access to final_test is prohibited")
    manifest = json.loads((root / "manifest.json").read_text())
    names = {"train": "train.jsonl", "development": "development.jsonl", "final_test": "final_test.smoke.jsonl" if manifest["mode"] == "smoke" else "final_test.sealed.jsonl"}
    if partition not in names:
        raise ValueError(f"unknown partition: {partition}")
    return [json.loads(line) for line in (root / names[partition]).read_text().splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("artifacts/r1_operator_semantic_benchmark/protocol.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--authorize-full-protocol-run", action="store_true")
    args = parser.parse_args()
    manifest = write_corpus(args.output, args.protocol, mode=args.mode, authorize_full=args.authorize_full_protocol_run)
    print(_json(manifest, pretty=True))


if __name__ == "__main__":
    main()
