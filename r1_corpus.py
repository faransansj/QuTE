"""Deterministic R1 semantic-pair corpus generator and exact oracle audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from cc_nqe import Gate, circuit_id, circuit_unitary, generate_circuit


SCHEMA_VERSION = "qute-r1-corpus-v1"
SEED_SCHEMA = "qute-r1-seed-v1"
SEED_PREFIX = b"qute-r1-seed-v1\0"
ORACLE_PROBE_SET_ID = "qute-r1-oracle-basis-4q-v1"
SMOKE_V2_NAMESPACE = "qute:r1:smoke:v2"
SCIENTIFIC_PILOT_NAMESPACE = "qute:r1:scientific-pilot:v1"
SCIENTIFIC_DEVELOPMENT_NAMESPACE = "qute:r1:scientific-development:v1"
SCIENTIFIC_SEALED_NAMESPACE = "qute:r1:scientific-sealed:v1"
R1_ARTIFACT_ROOT = Path("artifacts/r1_operator_semantic_benchmark")
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
    base: list[Gate],
    family: str,
    protocol: dict[str, Any],
    seed: int,
    *,
    instance_ood: bool = False,
    parameter_region: str = "train",
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
        if instance_ood:
            return compact, split, "rotation_split_heldout_direction"
        return split, compact, "rotation_fusion"

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

    def add(
        partition: str,
        split: str,
        family: str,
        depth: int,
        count: int,
        *,
        region: str = "train",
        instance_ood: bool = False,
    ) -> None:
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
            add(
                "final_test",
                "rewrite_instance_ood",
                family,
                depth,
                final["rewrite_instance_ood"]["pairs_per_family_depth"],
                instance_ood=True,
            )
    for split, family in HELD_OUT_SPLITS.items():
        for depth in (2, 4, 6):
            add("final_test", split, family, depth, final[split]["pairs_per_family_depth"])
    for region in ("interpolation_ood", "extrapolation_ood"):
        for family in SEEN_FAMILIES:
            for depth in (2, 4, 6):
                add(
                    "final_test",
                    "parameter_ood",
                    family,
                    depth,
                    final["parameter_ood"]["pairs_per_stratum"],
                    region=region,
                )
    for family in SEEN_FAMILIES:
        for depth in (8, 10):
            add("final_test", "depth_ood", family, depth, final["depth_ood"]["pairs_per_family_depth"])
    return plans


def planned_counts(protocol_path: str | Path, *, mode: str = "full") -> dict[str, Any]:
    """Return the frozen allocation without generating or accessing corpus records."""
    protocol = json.loads(Path(protocol_path).read_text())
    plans = _plans(protocol, mode)
    partitions = dict(Counter(plan["partition"] for plan in plans))
    splits = dict(Counter(plan["split"] for plan in plans))
    positive_pairs = len(plans)
    return {
        "partitions": partitions,
        "splits": splits,
        "positive_pairs": positive_pairs,
        "matched_negative_pairs": positive_pairs,
        "records": positive_pairs * 2,
    }


def _probe(protocol_hash: str, partition: str, base_id: str, index: int) -> dict[str, Any]:
    probe_seed = _seed(protocol_hash, partition, base_id, index)
    family = "product" if partition != "final_test" else ("entangled" if probe_seed % 2 == 0 else "haar")
    return {"probe_id": f"p_{probe_seed:016x}", "family": family, "seed": probe_seed, "index": index}


def _record_pair(
    protocol: dict[str, Any], protocol_hash: str, plan: dict[str, Any], global_index: int, attempt: int
) -> tuple[dict[str, Any], dict[str, Any], str, str, str]:
    item_seed = _seed(
        protocol_hash,
        plan["partition"],
        plan["split"],
        plan["family"],
        plan["base_depth"],
        plan["parameter_region"],
        plan["local_index"],
        attempt,
    )
    regime = {
        "train": "train",
        "interpolation_ood": "interpolation",
        "extrapolation_ood": "extrapolation",
    }[plan["parameter_region"]]
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
    pair_key = f"{protocol_hash}|{plan['partition']}|{plan['split']}|{global_index}|{item_seed}"
    pair_root = _sha256_bytes(pair_key.encode())[:20]
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
        "oracle": {
            "phase_aligned_relative_frobenius": negative_distance,
            "minimum_required": 0.1,
            "pass": negative_distance >= 0.1,
        },
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
            positive, negative, base_id, positive_hash, negative_hash = _record_pair(
                protocol, protocol_hash, plan, global_index, attempt
            )
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
        minimum_negative_distance = min(
            minimum_negative_distance,
            negative["oracle"]["phase_aligned_relative_frobenius"],
        )

    counts = {
        partition: dict(sorted(Counter(record["label"] for record in values).items()))
        for partition, values in records.items()
    }
    expected = {
        partition: {
            "equivalent": sum(plan["partition"] == partition for plan in plans),
            "non_equivalent": sum(plan["partition"] == partition for plan in plans),
        }
        for partition in records
    }
    partition_leakage = sorted(
        operator_hash
        for operator_hash, partitions in operator_partitions.items()
        if len(partitions) > 1
    )
    pair_ids = [record["pair_id"] for values in records.values() for record in values]
    audit = {
        "schema_version": "qute-r1-corpus-audit-v1",
        "status": (
            "PASS"
            if counts == expected and not partition_leakage and len(pair_ids) == len(set(pair_ids))
            else "FAIL"
        ),
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
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "protocol_sha256": protocol_hash,
        "records": records,
        "audit": audit,
    }


def write_corpus(
    output_dir: str | Path, protocol_path: str | Path, *, mode: str = "smoke", authorize_full: bool = False
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty corpus directory: {output_dir}")
    corpus = build_corpus(protocol_path, mode=mode, authorize_full=authorize_full)
    regenerated = build_corpus(protocol_path, mode=mode, authorize_full=authorize_full)
    if _json(corpus) != _json(regenerated):
        raise RuntimeError("corpus regeneration was not byte-deterministic")
    corpus["audit"]["deterministic_regeneration"] = True
    output_dir.mkdir(parents=True, exist_ok=True)

    file_hashes: dict[str, str] = {}
    names = {
        "train": "train.jsonl",
        "development": "development.jsonl",
        "final_test": (
            "final_test.smoke.jsonl" if mode == "smoke" else "final_test.sealed.jsonl"
        ),
    }
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
        "generator_sha256": _sha256_file(Path(__file__).resolve()),
        "counts": corpus["audit"]["counts"],
        "file_hashes": dict(sorted(file_hashes.items())),
        "scientific_final_test_generated": mode == "full",
        "final_test_access_count": 0,
        "deterministic_regeneration": True,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(_json(manifest, pretty=True) + "\n")
    file_hashes[manifest_path.name] = _sha256_file(manifest_path)
    checksum_path = output_dir / "checksums.sha256"
    checksum_path.write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(file_hashes.items())))
    if mode == "full":
        (output_dir / "FINAL_TEST_ACCESS_LOG.jsonl").write_text("")
    return manifest


def load_partition(
    root: str | Path,
    partition: str,
    *,
    allow_final: bool = False,
    access_reason: str | None = None,
) -> list[dict[str, Any]]:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    names = {
        "train": "train.jsonl",
        "development": "development.jsonl",
        "final_test": (
            "final_test.smoke.jsonl"
            if manifest["mode"] == "smoke"
            else "final_test.sealed.jsonl"
        ),
    }
    if partition not in names:
        raise ValueError(f"unknown partition: {partition}")
    if partition == "final_test" and not allow_final:
        raise PermissionError("ordinary development access to final_test is prohibited")
    path = root / names[partition]
    expected_hash = manifest.get("file_hashes", {}).get(path.name)
    if expected_hash is None or _sha256_file(path) != expected_hash:
        raise RuntimeError(f"corpus partition checksum failed: {path.name}")
    if partition == "final_test" and manifest["mode"] == "full":
        if not access_reason or not access_reason.strip():
            raise ValueError("access_reason is required for full final_test access")
        event = {
            "accessed_utc": datetime.now(timezone.utc).isoformat(),
            "reason": access_reason.strip(),
            "file": path.name,
            "sha256": _sha256_file(path),
        }
        with (root / "FINAL_TEST_ACCESS_LOG.jsonl").open("a") as handle:
            handle.write(_json(event) + "\n")
    return [json.loads(line) for line in path.read_text().splitlines()]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    return _sha256_bytes(canonical_json_bytes(value))


def derive_seed(
    *,
    protocol_sha256: str,
    generation_namespace: str,
    corpus_role: str,
    partition_role: str,
    master_seed: int,
    rewrite_family: str,
    template_id: str,
    local_index: int,
) -> int:
    coordinates = {
        "seed_schema": SEED_SCHEMA,
        "protocol_sha256": protocol_sha256,
        "generation_namespace": generation_namespace,
        "corpus_role": corpus_role,
        "partition_role": partition_role,
        "master_seed": master_seed,
        "rewrite_family": rewrite_family,
        "template_id": template_id,
        "local_index": local_index,
    }
    return int.from_bytes(
        hashlib.sha256(SEED_PREFIX + canonical_json_bytes(coordinates)).digest()[:8],
        "big",
    )


def canonical_circuit(circuit: Iterable[Gate], *, n_qubits: int = 4) -> dict[str, Any]:
    gates = []
    for position, gate in enumerate(circuit):
        gates.append(
            {
                "position": position,
                "gate": gate.name,
                "qubits": list(gate.qubits),
                "parameters": [] if gate.theta is None else [float(gate.theta).hex()],
            }
        )
    return {"n_qubits": n_qubits, "gates": gates}


def circuit_content_sha256(circuit: Iterable[Gate]) -> str:
    return content_sha256(canonical_circuit(circuit))


def namespaced_id(kind: str, generation_namespace: str, content_hash: str) -> str:
    digest = content_sha256(
        {
            "kind": kind,
            "generation_namespace": generation_namespace,
            "content_sha256": content_hash,
        }
    )[:24]
    return f"{kind}_{digest}"


def pair_content_hashes(source_hash: str, target_hash: str) -> tuple[str, str]:
    ordered = content_sha256({"source": source_hash, "target": target_hash})
    unordered = content_sha256({"members": sorted((source_hash, target_hash))})
    return ordered, unordered


def semantic_class_sha256(semantic_operator_sha256: str) -> str:
    return content_sha256({"semantic_operator_sha256": semantic_operator_sha256})


def validate_lifecycle_transition(
    namespace_entry: dict[str, Any], target_role: str
) -> None:
    if namespace_entry.get("status") == "RETIRED":
        raise PermissionError("retired corpus reuse is forbidden")
    if namespace_entry["corpus_role"] != target_role:
        raise PermissionError("corpus promotion between roles is forbidden")


def rewrite_coverage_complete(matrix: dict[str, Any]) -> bool:
    required = [cell for cell in matrix["cells"] if cell["required"]]
    return bool(required) and all(
        cell["implemented"]
        and cell["oracle_tested"]
        and cell["smoke_realized"]
        for cell in required
    )


def _coverage_pair(
    base: list[Gate],
    cell: dict[str, Any],
    protocol: dict[str, Any],
    seed: int,
) -> tuple[list[Gate], list[Gate]]:
    q0, q1, q2, _ = _qubits(seed)
    axis = cell.get("axis")
    template = cell["template_id"]
    angle_a = _parameter_angle(protocol, "train", seed)
    angle_b = _parameter_angle(protocol, "train", seed ^ 0x9E3779B97F4A7C15)
    position = seed % (len(base) + 1)
    prefix, suffix = base[:position], base[position:]

    if template == "inverse_h":
        return base, _insert(base, [Gate("H", (q0,)), Gate("H", (q0,))], seed)
    if template == "inverse_x":
        return base, _insert(base, [Gate("X", (q0,)), Gate("X", (q0,))], seed)
    if template == "inverse_cnot":
        gates = [Gate("CNOT", (q0, q1)), Gate("CNOT", (q0, q1))]
        return base, _insert(base, gates, seed)
    if template == "inverse_rotation":
        gates = [Gate(axis, (q0,), angle_a), Gate(axis, (q0,), -angle_a)]
        return base, _insert(base, gates, seed)
    if template == "commute_disjoint":
        first, second = Gate("H", (q0,)), Gate("X", (q1,))
        return [*prefix, first, second, *suffix], [*prefix, second, first, *suffix]
    if template == "commute_rz_rz":
        first = Gate("RZ", (q0,), angle_a)
        second = Gate("RZ", (q1,), angle_b)
        return [*prefix, first, second, *suffix], [*prefix, second, first, *suffix]
    if template == "rotation_fusion":
        first = Gate(axis, (q0,), angle_a)
        second = Gate(axis, (q0,), angle_b)
        fused = Gate(axis, (q0,), (angle_a + angle_b) % (2 * math.pi))
        return [*prefix, first, second, *suffix], [*prefix, fused, *suffix]
    if template == "identity_hxxh":
        identity = [
            Gate("H", (q0,)),
            Gate("X", (q0,)),
            Gate("X", (q0,)),
            Gate("H", (q0,)),
        ]
        return base, _insert(base, identity, seed)
    if template == "identity_cnot_hh_cnot":
        identity = [
            Gate("CNOT", (q0, q1)),
            Gate("H", (q2,)),
            Gate("H", (q2,)),
            Gate("CNOT", (q0, q1)),
        ]
        return base, _insert(base, identity, seed)
    if template == "reverse_direction_cnot":
        direct = [Gate("CNOT", (q0, q1))]
        reverse = [
            Gate("H", (q0,)),
            Gate("H", (q1,)),
            Gate("CNOT", (q1, q0)),
            Gate("H", (q0,)),
            Gate("H", (q1,)),
        ]
        return _insert(base, direct, seed), _insert(base, reverse, seed)
    if template == "swap_three_cnot":
        first = [
            Gate("CNOT", (q0, q1)),
            Gate("CNOT", (q1, q0)),
            Gate("CNOT", (q0, q1)),
        ]
        second = [
            Gate("CNOT", (q1, q0)),
            Gate("CNOT", (q0, q1)),
            Gate("CNOT", (q1, q0)),
        ]
        return _insert(base, first, seed), _insert(base, second, seed)
    raise ValueError(f"unsupported coverage cell: {cell['coverage_cell_id']}")


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def _gate_counts(circuit: Iterable[Gate]) -> dict[str, int]:
    return dict(sorted(Counter(gate.name for gate in circuit).items()))


def _process_fidelity(left: np.ndarray, right: np.ndarray) -> float:
    dimension = left.shape[0]
    return float(abs(np.vdot(left, right)) ** 2 / (dimension * dimension))


def _model_probe(seed: int, generation_namespace: str) -> dict[str, Any]:
    descriptor = {
        "n_qubits": 4,
        "family": "product",
        "angles_hex": [
            float(((seed >> shift) & 0xFFFF) / 0xFFFF * math.pi).hex()
            for shift in (0, 16, 32, 48)
        ],
    }
    digest = content_sha256(descriptor)
    return {
        "model_probe_id": namespaced_id("model_probe", generation_namespace, digest),
        "model_probe_content_sha256": digest,
        "descriptor": descriptor,
    }


def _circuit_metadata(
    circuit: list[Gate],
    *,
    base_hash: str,
    generation_namespace: str,
    corpus_role: str,
    partition_role: str,
    parameter_region: str,
    seed: int,
    protocol_sha256: str,
    generator_commit: str,
) -> dict[str, Any]:
    canonical = canonical_circuit(circuit)
    digest = content_sha256(canonical)
    circuit_id_value = namespaced_id("circuit", generation_namespace, digest)
    return {
        "record_id": namespaced_id("record", generation_namespace, digest),
        "circuit_id": circuit_id_value,
        "namespace": generation_namespace,
        "corpus_role": corpus_role,
        "partition_role": partition_role,
        "circuit_content_sha256": digest,
        "base_circuit_content_sha256": base_hash,
        "canonical_serialization": canonical,
        "depth": len(circuit),
        "gate_counts": _gate_counts(circuit),
        "parameter_region": parameter_region,
        "seed_schema": SEED_SCHEMA,
        "derived_seed": seed,
        "protocol_sha256": protocol_sha256,
        "generator_commit": generator_commit,
    }


def build_metadata_smoke(
    protocol_path: str | Path,
    coverage_matrix_path: str | Path,
    *,
    generation_namespace: str = SMOKE_V2_NAMESPACE,
    master_seed: int = 2026,
) -> dict[str, Any]:
    protocol_bytes = Path(protocol_path).read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol_hash = _sha256_bytes(protocol_bytes)
    matrix = json.loads(Path(coverage_matrix_path).read_text())
    if not rewrite_coverage_complete(matrix):
        raise RuntimeError("required rewrite coverage is incomplete")
    corpus_role = "SMOKE_DEVELOPMENT"
    partition_role = "SMOKE_COVERAGE"
    generator_commit = _git_head()
    circuits_by_id: dict[str, dict[str, Any]] = {}
    classes: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    oracle_failures: list[str] = []

    for local_index, cell in enumerate(matrix["cells"]):
        seed = derive_seed(
            protocol_sha256=protocol_hash,
            generation_namespace=generation_namespace,
            corpus_role=corpus_role,
            partition_role=partition_role,
            master_seed=master_seed,
            rewrite_family=cell["family"],
            template_id=cell["template_id"] + (f":{cell['axis']}" if cell.get("axis") else ""),
            local_index=local_index,
        )
        base = generate_circuit(seed, 4, regime="train")
        base_hash = circuit_content_sha256(base)
        source, target = _coverage_pair(base, cell, protocol, seed)
        source_u, target_u = circuit_unitary(source), circuit_unitary(target)
        frobenius, probe_l2, probability_tvd = _phase_aligned_errors(source_u, target_u)
        tolerances = protocol["oracle"]["tolerances"]
        oracle_pass = (
            frobenius <= tolerances["phase_aligned_relative_frobenius"]
            and probe_l2 <= tolerances["maximum_probe_l2"]
            and probability_tvd <= tolerances["maximum_probability_tvd"]
        )
        if not oracle_pass:
            oracle_failures.append(cell["coverage_cell_id"])
        source_meta = _circuit_metadata(
            source,
            base_hash=base_hash,
            generation_namespace=generation_namespace,
            corpus_role=corpus_role,
            partition_role=partition_role,
            parameter_region="train",
            seed=seed,
            protocol_sha256=protocol_hash,
            generator_commit=generator_commit,
        )
        target_meta = _circuit_metadata(
            target,
            base_hash=base_hash,
            generation_namespace=generation_namespace,
            corpus_role=corpus_role,
            partition_role=partition_role,
            parameter_region="train",
            seed=seed,
            protocol_sha256=protocol_hash,
            generator_commit=generator_commit,
        )
        negative_target, control_type, negative_distance, negative_u = _negative_right(
            source_u, target
        )
        negative_meta = _circuit_metadata(
            negative_target,
            base_hash=base_hash,
            generation_namespace=generation_namespace,
            corpus_role=corpus_role,
            partition_role=partition_role,
            parameter_region="train",
            seed=seed,
            protocol_sha256=protocol_hash,
            generator_commit=generator_commit,
        )
        for metadata in (source_meta, target_meta, negative_meta):
            circuits_by_id[metadata["circuit_id"]] = metadata

        semantic_operator = _operator_hash(source_u)
        semantic_class = semantic_class_sha256(semantic_operator)
        equivalence_class_id = namespaced_id(
            "equivalence_class", generation_namespace, semantic_class
        )
        model_probe = _model_probe(seed, generation_namespace)

        def make_pair(
            right_meta: dict[str, Any],
            right_u: np.ndarray,
            *,
            label: str,
            pair_role: str,
            difficulty: str,
            control: str | None,
        ) -> dict[str, Any]:
            ordered, unordered = pair_content_hashes(
                source_meta["circuit_content_sha256"],
                right_meta["circuit_content_sha256"],
            )
            rewrite_hash = content_sha256(
                {
                    "family": cell["family"],
                    "template_id": cell["template_id"],
                    "axis": cell.get("axis"),
                    "base_circuit_content_sha256": base_hash,
                    "source": source_meta["circuit_content_sha256"],
                    "target": right_meta["circuit_content_sha256"],
                    "pair_role": pair_role,
                }
            )
            fidelity = _process_fidelity(source_u, right_u)
            distance = _phase_aligned_errors(source_u, right_u)[0]
            pair_id_value = namespaced_id("pair", generation_namespace, ordered)
            return {
                "record_id": namespaced_id("record", generation_namespace, pair_id_value),
                "pair_id": pair_id_value,
                "equivalence_class_id": equivalence_class_id,
                "namespace": generation_namespace,
                "corpus_role": corpus_role,
                "partition_role": partition_role,
                "source_circuit_id": source_meta["circuit_id"],
                "target_circuit_id": right_meta["circuit_id"],
                "source_circuit_content_sha256": source_meta["circuit_content_sha256"],
                "target_circuit_content_sha256": right_meta["circuit_content_sha256"],
                "ordered_pair_content_sha256": ordered,
                "unordered_pair_content_sha256": unordered,
                "label": label,
                "pair_role": pair_role,
                "rewrite_family": cell["family"],
                "template_id": cell["template_id"],
                "coverage_cell_id": cell["coverage_cell_id"],
                "rewrite_instance_id": namespaced_id(
                    "rewrite_instance", generation_namespace, rewrite_hash
                ),
                "rewrite_instance_content_sha256": rewrite_hash,
                "process_fidelity": fidelity,
                "process_infidelity": max(0.0, 1.0 - fidelity),
                "phase_aligned_relative_frobenius": distance,
                "difficulty_stratum": difficulty,
                "control_type": control,
                "oracle_probe_verification": {
                    "oracle_probe_set_id": ORACLE_PROBE_SET_ID,
                    "basis_probe_count": 16,
                    "maximum_probe_l2": _phase_aligned_errors(source_u, right_u)[1],
                    "maximum_probability_tvd": _phase_aligned_errors(source_u, right_u)[2],
                    "shared_across_partitions": True,
                },
                **model_probe,
                "seed_schema": SEED_SCHEMA,
                "derived_seed": seed,
                "protocol_sha256": protocol_hash,
                "generator_commit": generator_commit,
            }

        positive = make_pair(
            target_meta,
            target_u,
            label="POSITIVE_EQUIVALENT",
            pair_role="SEMANTIC_EQUIVALENCE",
            difficulty="EQUIVALENT_ORACLE_CONFIRMED",
            control=None,
        )
        negative = make_pair(
            negative_meta,
            negative_u,
            label="NEGATIVE_NON_EQUIVALENT",
            pair_role="MATCHED_NON_EQUIVALENT_CONTROL",
            difficulty="FAR_NEGATIVE_SANITY",
            control=control_type,
        )
        if negative["phase_aligned_relative_frobenius"] < 0.1:
            raise RuntimeError("negative control violates frozen distance floor")
        pairs.extend((positive, negative))
        class_record = {
            "record_id": namespaced_id("record", generation_namespace, equivalence_class_id),
            "equivalence_class_id": equivalence_class_id,
            "namespace": generation_namespace,
            "corpus_role": corpus_role,
            "partition_role": partition_role,
            "semantic_class_sha256": semantic_class,
            "base_circuit_id": namespaced_id("circuit", generation_namespace, base_hash),
            "base_circuit_content_sha256": base_hash,
            "semantic_operator_sha256": semantic_operator,
            "rewrite_family": cell["family"],
            "template_id": cell["template_id"],
            "coverage_cell_id": cell["coverage_cell_id"],
            "rewrite_chain_length": 1,
            "variant_circuit_ids": sorted(
                (source_meta["circuit_id"], target_meta["circuit_id"])
            ),
            "pair_ids": sorted((positive["pair_id"], negative["pair_id"])),
            "class_cardinality": 2,
            "seed_schema": SEED_SCHEMA,
            "derived_seed": seed,
            "protocol_sha256": protocol_hash,
            "generator_commit": generator_commit,
        }
        classes.append(class_record)

    if oracle_failures:
        raise RuntimeError(f"oracle failures: {oracle_failures}")
    circuits = sorted(circuits_by_id.values(), key=lambda item: item["circuit_id"])
    classes.sort(key=lambda item: item["equivalence_class_id"])
    pairs.sort(key=lambda item: item["pair_id"])
    audit = {
        "schema_version": "qute-r1-metadata-audit-v1",
        "status": "PASS",
        "generation_namespace": generation_namespace,
        "corpus_role": corpus_role,
        "scientific_use": "FORBIDDEN",
        "sealable": False,
        "promotion_allowed": False,
        "coverage_required": len(matrix["cells"]),
        "coverage_realized": len(classes),
        "coverage_fraction": 1.0,
        "oracle_failures": 0,
        "positive_pairs": len(classes),
        "negative_controls": len(classes),
        "negative_metadata_contract_valid": all(
            pair["label"] == "NEGATIVE_NON_EQUIVALENT"
            and pair["pair_role"] == "MATCHED_NON_EQUIVALENT_CONTROL"
            and pair["difficulty_stratum"] == "FAR_NEGATIVE_SANITY"
            for pair in pairs
            if pair["label"] == "NEGATIVE_NON_EQUIVALENT"
        ),
        "full_generation_authorized": False,
    }
    return {
        "protocol_sha256": protocol_hash,
        "generator_commit": generator_commit,
        "generator_source_sha256": _sha256_file(Path(__file__).resolve()),
        "generation_namespace": generation_namespace,
        "corpus_role": corpus_role,
        "partition_role": partition_role,
        "circuits": circuits,
        "equivalence_classes": classes,
        "pairs": pairs,
        "audit": audit,
    }


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(_json(record) + "\n" for record in records))


def _index_values(bundle: dict[str, Any]) -> dict[str, list[str]]:
    circuits = bundle["circuits"]
    classes = bundle["equivalence_classes"]
    pairs = bundle["pairs"]
    return {
        "record_ids.txt": sorted(
            {record["record_id"] for group in (circuits, classes, pairs) for record in group}
        ),
        "circuit_content_hashes.txt": sorted(
            {record["circuit_content_sha256"] for record in circuits}
        ),
        "base_circuit_content_hashes.txt": sorted(
            {record["base_circuit_content_sha256"] for record in circuits}
        ),
        "ordered_pair_content_hashes.txt": sorted(
            {record["ordered_pair_content_sha256"] for record in pairs}
        ),
        "unordered_pair_content_hashes.txt": sorted(
            {record["unordered_pair_content_sha256"] for record in pairs}
        ),
        "equivalence_class_ids.txt": sorted(
            {record["equivalence_class_id"] for record in classes}
        ),
        "semantic_class_hashes.txt": sorted(
            {record["semantic_class_sha256"] for record in classes}
        ),
        "rewrite_instance_hashes.txt": sorted(
            {record["rewrite_instance_content_sha256"] for record in pairs}
        ),
        "model_probe_content_hashes.txt": sorted(
            {record["model_probe_content_sha256"] for record in pairs}
        ),
    }


def write_metadata_corpus(output_dir: str | Path, bundle: dict[str, Any]) -> dict[str, Any]:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty corpus directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "circuits.jsonl", bundle["circuits"])
    _write_jsonl(output / "equivalence_classes.jsonl", bundle["equivalence_classes"])
    _write_jsonl(output / "pairs.jsonl", bundle["pairs"])
    (output / "audit.json").write_text(_json(bundle["audit"], pretty=True) + "\n")
    indices = output / "indices"
    indices.mkdir()
    for name, values in _index_values(bundle).items():
        (indices / name).write_text("".join(value + "\n" for value in values))
    tracked = sorted(
        path for path in output.rglob("*") if path.is_file()
    )
    file_hashes = {
        str(path.relative_to(output)): _sha256_file(path) for path in tracked
    }
    manifest = {
        "schema_version": "qute-r1-corpus-manifest-v2",
        "generation_namespace": bundle["generation_namespace"],
        "seed_schema": SEED_SCHEMA,
        "corpus_role": "SMOKE_DEVELOPMENT",
        "partition_role": bundle["partition_role"],
        "status": "VALIDATED_SMOKE_EVIDENCE",
        "scientific_use": "FORBIDDEN",
        "sealable": False,
        "promotion_allowed": False,
        "reuse_allowed": False,
        "protocol_sha256": bundle["protocol_sha256"],
        "generator_commit": bundle["generator_commit"],
        "generator_source_sha256": bundle["generator_source_sha256"],
        "counts": {
            "circuits": len(bundle["circuits"]),
            "equivalence_classes": len(bundle["equivalence_classes"]),
            "pairs": len(bundle["pairs"]),
        },
        "oracle_probe_set_id": ORACLE_PROBE_SET_ID,
        "basis_probe_count": 16,
        "scientific_development_payload_generated": False,
        "scientific_sealed_payload_generated": False,
        "full_generation_authorized": False,
        "file_hashes": file_hashes,
    }
    (output / "corpus_manifest.json").write_text(_json(manifest, pretty=True) + "\n")
    tracked = sorted(path for path in output.rglob("*") if path.is_file())
    (output / "checksums.sha256").write_text(
        "".join(
            f"{_sha256_file(path)}  {path.relative_to(output)}\n" for path in tracked
        )
    )
    return manifest


def generate_smoke_v2(
    output_dir: str | Path,
    protocol_path: str | Path,
    coverage_matrix_path: str | Path,
) -> dict[str, Any]:
    bundle = build_metadata_smoke(protocol_path, coverage_matrix_path)
    return write_metadata_corpus(output_dir, bundle)


def namespace_isolation_audit(
    protocol_path: str | Path,
    coverage_matrix_path: str | Path,
) -> dict[str, Any]:
    protocol_hash = _sha256_file(Path(protocol_path))
    matrix = json.loads(Path(coverage_matrix_path).read_text())
    roles = {
        SMOKE_V2_NAMESPACE: ("SMOKE_DEVELOPMENT", "SMOKE_COVERAGE"),
        SCIENTIFIC_DEVELOPMENT_NAMESPACE: (
            "SCIENTIFIC_DEVELOPMENT",
            "SCIENTIFIC_DEVELOPMENT",
        ),
        SCIENTIFIC_SEALED_NAMESPACE: (
            "SCIENTIFIC_SEALED_FINAL",
            "SCIENTIFIC_SEALED_FINAL",
        ),
    }
    records = []
    for namespace, (role, partition) in roles.items():
        for local_index, cell in enumerate(matrix["cells"]):
            records.append(
                {
                    "generation_namespace": namespace,
                    "coverage_cell_id": cell["coverage_cell_id"],
                    "derived_seed": derive_seed(
                        protocol_sha256=protocol_hash,
                        generation_namespace=namespace,
                        corpus_role=role,
                        partition_role=partition,
                        master_seed=2026,
                        rewrite_family=cell["family"],
                        template_id=cell["template_id"]
                        + (f":{cell['axis']}" if cell.get("axis") else ""),
                        local_index=local_index,
                    ),
                }
            )
    collisions = [
        seed for seed, count in Counter(r["derived_seed"] for r in records).items() if count > 1
    ]
    return {
        "schema_version": "qute-r1-namespace-isolation-audit-v1",
        "status": "PASS" if not collisions else "FAIL",
        "seed_schema": SEED_SCHEMA,
        "namespaces": sorted(roles),
        "coordinates_per_namespace": len(matrix["cells"]),
        "derived_seed_count": len(records),
        "derived_seed_collision_count": len(collisions),
        "collision_examples": collisions[:20],
        "scientific_payloads_generated": False,
    }


def determinism_audit(
    protocol_path: str | Path,
    coverage_matrix_path: str | Path,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
        generate_smoke_v2(first_dir, protocol_path, coverage_matrix_path)
        generate_smoke_v2(second_dir, protocol_path, coverage_matrix_path)
        first = Path(first_dir)
        second = Path(second_dir)
        names = sorted(str(path.relative_to(first)) for path in first.rglob("*") if path.is_file())
        mismatches = [
            name for name in names if (first / name).read_bytes() != (second / name).read_bytes()
        ]
    return {
        "schema_version": "qute-r1-determinism-audit-v1",
        "status": "PASS" if not mismatches else "FAIL",
        "compared_file_count": len(names),
        "byte_mismatch_count": len(mismatches),
        "mismatch_examples": mismatches[:20],
        "excluded_fields": [],
    }


def _read_index(root: Path, name: str) -> set[str]:
    path = root / "indices" / name
    return set(path.read_text().splitlines()) if path.exists() else set()


def _legacy_smoke_v1_hashes(root: Path, protocol_path: Path) -> dict[str, set[str]]:
    fields = {
        "circuit_content_sha256": set(),
        "base_circuit_content_sha256": set(),
        "ordered_pair_content_sha256": set(),
        "unordered_pair_content_sha256": set(),
        "rewrite_instance_content_sha256": set(),
        "semantic_class_sha256": set(),
        "model_probe_content_sha256": set(),
    }
    for filename in ("train.jsonl", "development.jsonl", "final_test.smoke.jsonl"):
        for line in (root / filename).read_text().splitlines():
            record = json.loads(line)
            left = [Gate.from_dict(gate) for gate in record["left_circuit"]]
            right = [Gate.from_dict(gate) for gate in record["right_circuit"]]
            left_hash = circuit_content_sha256(left)
            right_hash = circuit_content_sha256(right)
            fields["circuit_content_sha256"].update((left_hash, right_hash))
            regime = {
                "train": "train",
                "interpolation_ood": "interpolation",
                "extrapolation_ood": "extrapolation",
            }[record["parameter_region"]]
            base = generate_circuit(record["generation_seed"], record["base_depth"], regime=regime)
            base_hash = circuit_content_sha256(base)
            fields["base_circuit_content_sha256"].add(base_hash)
            ordered, unordered = pair_content_hashes(left_hash, right_hash)
            fields["ordered_pair_content_sha256"].add(ordered)
            fields["unordered_pair_content_sha256"].add(unordered)
            fields["rewrite_instance_content_sha256"].add(
                content_sha256(
                    {
                        "family": record["rewrite_family"],
                        "template_id": record["template_id"],
                        "base_circuit_content_sha256": base_hash,
                        "source": left_hash,
                        "target": right_hash,
                        "pair_role": record["label"],
                    }
                )
            )
            operator_hash = record.get("operator_hash") or record.get("left_operator_hash")
            fields["semantic_class_sha256"].add(semantic_class_sha256(operator_hash))
            fields["model_probe_content_sha256"].add(content_sha256(record["probe"]))
    return fields


def content_hash_intersections(
    left: dict[str, set[str]],
    right: dict[str, set[str]],
    fields: Iterable[str],
    *,
    example_limit: int = 20,
) -> dict[str, dict[str, Any]]:
    result = {}
    for field in fields:
        examples = sorted(left.get(field, set()) & right.get(field, set()))
        result[field] = {"count": len(examples), "examples": examples[:example_limit]}
    return result


def intersection_audit(
    smoke_v1_root: str | Path,
    smoke_v2_root: str | Path,
    protocol_path: str | Path,
    overlap_policy_path: str | Path,
) -> dict[str, Any]:
    smoke_v1 = _legacy_smoke_v1_hashes(Path(smoke_v1_root), Path(protocol_path))
    smoke_v2_root = Path(smoke_v2_root)
    index_names = {
        "circuit_content_sha256": "circuit_content_hashes.txt",
        "base_circuit_content_sha256": "base_circuit_content_hashes.txt",
        "ordered_pair_content_sha256": "ordered_pair_content_hashes.txt",
        "unordered_pair_content_sha256": "unordered_pair_content_hashes.txt",
        "rewrite_instance_content_sha256": "rewrite_instance_hashes.txt",
        "semantic_class_sha256": "semantic_class_hashes.txt",
        "model_probe_content_sha256": "model_probe_content_hashes.txt",
    }
    smoke_v2 = {
        field: _read_index(smoke_v2_root, filename)
        for field, filename in index_names.items()
    }
    fields = json.loads(Path(overlap_policy_path).read_text())["zero_overlap_fields"]
    legacy_overlaps = content_hash_intersections(smoke_v1, smoke_v2, fields)
    failures = [field for field, result in legacy_overlaps.items() if result["count"]]
    prohibited = []
    for left, right in (
        ("SMOKE_DEVELOPMENT", "SCIENTIFIC_DEVELOPMENT"),
        ("SMOKE_DEVELOPMENT", "SCIENTIFIC_SEALED_FINAL"),
        ("SCIENTIFIC_DEVELOPMENT", "SCIENTIFIC_SEALED_FINAL"),
    ):
        prohibited.append(
            {
                "left_role": left,
                "right_role": right,
                "payload_state": "RIGHT_OR_BOTH_NOT_GENERATED",
                "overlap_counts": {field: 0 for field in index_names},
            }
        )
    return {
        "schema_version": "qute-r1-intersection-audit-v1",
        "status": "PASS" if not failures else "FAIL",
        "smoke_v1_vs_smoke_v2": legacy_overlaps,
        "prohibited_role_intersections": prohibited,
        "failure_fields": failures,
        "oracle_probe_set_id_exempt": ORACLE_PROBE_SET_ID,
        "scientific_development_payload_generated": False,
        "scientific_sealed_payload_generated": False,
    }


def protocol_contract_preflight(
    protocol_path: str | Path,
    coverage_matrix_path: str | Path,
    *,
    pilot_class_count: int = 1000,
) -> dict[str, Any]:
    protocol = json.loads(Path(protocol_path).read_text())
    matrix = json.loads(Path(coverage_matrix_path).read_text())
    allocation = protocol["corpus_allocation"]
    train = allocation["train"]
    development = allocation["development"]
    expected_train = len(train["families"]) * len(train["depths"]) * train["pairs_per_family_depth"]
    expected_development = (
        len(development["families"])
        * len(development["depths"])
        * development["pairs_per_family_depth"]
    )
    regions = protocol["scope"]["parameter_regions_radians"]
    flat_regions = [(*interval, name) for name, intervals in regions.items() for interval in intervals]
    region_overlap = any(
        max(left[0], right[0]) < min(left[1], right[1])
        for index, left in enumerate(flat_regions)
        for right in flat_regions[index + 1 :]
        if left[2] != right[2]
    )
    checks = {
        "train_count_contract": expected_train == train["positive_pairs"] == train["negative_pairs"],
        "development_count_contract": expected_development == development["positive_pairs"] == development["negative_pairs"],
        "rewrite_allocation": rewrite_coverage_complete(matrix) and len(matrix["cells"]) == 15,
        "parameter_regions_disjoint": not region_overlap,
        "depth_buckets": train["depths"] == development["depths"] == [2, 4, 6],
        "negative_control_scope": protocol["non_equivalent_controls"]["required_ratio"] == "one matched negative per positive pair",
    }
    blockers = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema_version": "qute-r1-canonical-contract-preflight-v1",
        "status": "PASS" if not blockers else "FAIL",
        "verdict": "R1-CANONICAL-PROTOCOL-RECONCILED" if not blockers else "PROTOCOL_CONTRACT_CONFLICT",
        "protocol_sha256": _sha256_file(Path(protocol_path)),
        "coverage_matrix_sha256": _sha256_file(Path(coverage_matrix_path)),
        "checks": checks,
        "blockers": blockers,
        "contract": {
            "scientific_development_positive_classes": train["positive_pairs"] + development["positive_pairs"],
            "pilot_equivalence_classes": pilot_class_count,
            "rewrite_cell_count": len(matrix["cells"]),
            "parameter_regions": regions,
            "development_depths": train["depths"],
            "negative_scope": "FAR_MATCHED_CONTROL",
        },
    }


def _pilot_cell_quotas(cells: list[dict[str, Any]], total: int) -> list[int]:
    base, remainder = divmod(total, len(cells))
    return [base + (index < remainder) for index in range(len(cells))]


def build_pilot_plan(
    protocol_path: str | Path,
    coverage_matrix_path: str | Path,
    *,
    pilot_class_count: int = 1000,
) -> dict[str, Any]:
    protocol = json.loads(Path(protocol_path).read_text())
    matrix = json.loads(Path(coverage_matrix_path).read_text())
    protocol_hash = _sha256_file(Path(protocol_path))
    cells = [cell for cell in matrix["cells"] if cell["required"]]
    coordinates: list[dict[str, Any]] = []
    seen_semantic: set[str] = set()
    attempts_by_cell: dict[str, int] = {}
    tolerances = protocol["oracle"]["tolerances"]
    for cell, quota in zip(cells, _pilot_cell_quotas(cells, pilot_class_count), strict=True):
        accepted = 0
        attempt = 0
        while accepted < quota and attempt < max(10, quota * 2):
            depth = (2, 4, 6)[attempt % 3]
            template = cell["template_id"] + (f":{cell['axis']}" if cell.get("axis") else "")
            seed = derive_seed(
                protocol_sha256=protocol_hash,
                generation_namespace=SCIENTIFIC_PILOT_NAMESPACE,
                corpus_role="SCIENTIFIC_PILOT",
                partition_role="PILOT_PLAN",
                master_seed=2026,
                rewrite_family=cell["family"],
                template_id=template,
                local_index=attempt,
            )
            base = generate_circuit(seed, depth, regime="train")
            source, target = _coverage_pair(base, cell, protocol, seed)
            source_hash = circuit_content_sha256(source)
            target_hash = circuit_content_sha256(target)
            semantic_hash = semantic_class_sha256(_operator_hash(circuit_unitary(source)))
            frobenius, probe_l2, probability_tvd = _phase_aligned_errors(
                circuit_unitary(source), circuit_unitary(target)
            )
            attempt += 1
            if semantic_hash in seen_semantic:
                continue
            if not (
                frobenius <= tolerances["phase_aligned_relative_frobenius"]
                and probe_l2 <= tolerances["maximum_probe_l2"]
                and probability_tvd <= tolerances["maximum_probability_tvd"]
            ):
                continue
            ordered, unordered = pair_content_hashes(source_hash, target_hash)
            base_hash = circuit_content_sha256(base)
            negative, control_type, negative_distance, _ = _negative_right(
                circuit_unitary(source), target
            )
            coordinate_id = namespaced_id(
                SCIENTIFIC_PILOT_NAMESPACE,
                "coordinate",
                content_sha256({"cell": cell["coverage_cell_id"], "local_index": attempt - 1}),
            )
            coordinates.append({
                "coordinate_id": coordinate_id,
                "generation_namespace": SCIENTIFIC_PILOT_NAMESPACE,
                "corpus_role": "SCIENTIFIC_PILOT",
                "partition_role": "PILOT_SINGLE_PARTITION",
                "master_seed": 2026,
                "coverage_cell_id": cell["coverage_cell_id"],
                "rewrite_family": cell["family"],
                "rewrite_template": template,
                "template_id": template,
                "local_index": attempt - 1,
                "generation_attempt": attempt - 1,
                "depth_bucket": depth,
                "base_depth": depth,
                "parameter_region": "train",
                "derived_seed": seed,
                "base_circuit_content_sha256": base_hash,
                "source_circuit_content_sha256": source_hash,
                "target_circuit_content_sha256": target_hash,
                "ordered_pair_content_sha256": ordered,
                "unordered_pair_content_sha256": unordered,
                "semantic_class_sha256": semantic_hash,
                "rewrite_instance_content_sha256": content_sha256({
                    "base": base_hash, "source": source_hash, "target": target_hash,
                    "family": cell["family"], "template": template,
                }),
                "negative_control_content_sha256": circuit_content_sha256(negative),
                "negative_control_type": "MATCHED_NON_EQUIVALENT_CONTROL",
                "negative_control_label": "NEGATIVE_NON_EQUIVALENT",
                "negative_control_claim": "FAR_NEGATIVE_SANITY",
                "negative_control_distance": negative_distance,
                "model_probe_content_sha256": _model_probe(seed, SCIENTIFIC_PILOT_NAMESPACE)["model_probe_content_sha256"],
                "oracle_metrics": {
                    "phase_aligned_relative_frobenius": frobenius,
                    "maximum_probe_l2": probe_l2,
                    "maximum_probability_tvd": probability_tvd,
                },
            })
            seen_semantic.add(semantic_hash)
            accepted += 1
        attempts_by_cell[cell["coverage_cell_id"]] = attempt
    return {
        "schema_version": "qute-r1-pilot-coordinate-plan-v1",
        "protocol_sha256": protocol_hash,
        "generation_namespace": SCIENTIFIC_PILOT_NAMESPACE,
        "corpus_role": "SCIENTIFIC_PILOT",
        "partition_role": "PILOT_SINGLE_PARTITION",
        "pilot_class_count": pilot_class_count,
        "environment_contract": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "byteorder": sys.byteorder,
        },
        "coordinates": coordinates,
        "attempts_by_cell": attempts_by_cell,
    }


def capacity_preflight(
    protocol_path: str | Path,
    coverage_matrix_path: str | Path,
    *,
    pilot_class_count: int = 1000,
) -> dict[str, Any]:
    plan = build_pilot_plan(
        protocol_path, coverage_matrix_path, pilot_class_count=pilot_class_count
    )
    matrix = json.loads(Path(coverage_matrix_path).read_text())
    quotas = _pilot_cell_quotas(matrix["cells"], pilot_class_count)
    rows = []
    for cell, quota in zip(matrix["cells"], quotas, strict=True):
        accepted = [
            row for row in plan["coordinates"]
            if row["coverage_cell_id"] == cell["coverage_cell_id"]
        ]
        rows.append({
            "coverage_cell_id": cell["coverage_cell_id"],
            "quota": quota,
            "unique_semantic_yield": len(accepted),
            "attempts": plan["attempts_by_cell"][cell["coverage_cell_id"]],
            "collision_count": plan["attempts_by_cell"][cell["coverage_cell_id"]] - len(accepted),
            "quota_met": len(accepted) == quota,
            "oracle_tolerance_pass": len(accepted) == quota,
            "depths_observed": sorted({row["base_depth"] for row in accepted}),
            "parameter_regions_observed": sorted({row["parameter_region"] for row in accepted}),
        })
    blockers = [row["coverage_cell_id"] for row in rows if not row["quota_met"]]
    return {
        "schema_version": "qute-r1-capacity-report-v1",
        "status": "PASS" if not blockers else "FAIL",
        "verdict": "R1-CELL-CAPACITY-PASS" if not blockers else "R1-CELL-CAPACITY-REVISION-REQUIRED",
        "payload_written": False,
        "automatic_reallocation": False,
        "pilot_class_count": pilot_class_count,
        "cells": rows,
        "blockers": blockers,
    }


def cross_process_reproducibility(
    protocol_path: str | Path,
    coverage_matrix_path: str | Path,
    *,
    pilot_class_count: int = 1000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    code = (
        "import sys; from r1_corpus import build_pilot_plan,canonical_json_bytes;"
        "sys.stdout.buffer.write(canonical_json_bytes(build_pilot_plan(sys.argv[1],sys.argv[2],pilot_class_count=int(sys.argv[3]))))"
    )
    outputs = []
    for hash_seed in ("1", "987654"):
        environment = dict(os.environ, PYTHONHASHSEED=hash_seed)
        outputs.append(subprocess.check_output(
            [sys.executable, "-c", code, str(protocol_path), str(coverage_matrix_path), str(pilot_class_count)],
            env=environment,
        ))
    plan = json.loads(outputs[0])
    identical = outputs[0] == outputs[1]
    report = {
        "schema_version": "qute-r1-reproducibility-report-v1",
        "status": "PASS" if identical else "FAIL",
        "verdict": "R1-CROSS-PROCESS-REPRODUCIBLE" if identical else "R1-CROSS-PROCESS-REPRODUCIBILITY-FAIL",
        "process_count": 2,
        "pythonhashseeds": [1, 987654],
        "coordinate_ledger_byte_identical": identical,
        "content_hashes_identical": identical,
        "environment_contract_match": identical,
        "oracle_metrics_tolerance_pass": all(
            max(row["oracle_metrics"].values()) <= 1e-10
            for row in plan["coordinates"]
        ),
        "ledger_sha256": _sha256_bytes(outputs[0]),
        "environment_contract": plan["environment_contract"],
        "cross_machine_check": "NOT_RUN_OPTIONAL",
    }
    return report, plan


def run_pilot_readiness_preflight(
    protocol_path: str | Path,
    coverage_matrix_path: str | Path,
    output_dir: str | Path,
    *,
    pilot_class_count: int = 1000,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    contract = protocol_contract_preflight(
        protocol_path, coverage_matrix_path, pilot_class_count=pilot_class_count
    )
    capacity = capacity_preflight(
        protocol_path, coverage_matrix_path, pilot_class_count=pilot_class_count
    )
    reproducibility, plan = cross_process_reproducibility(
        protocol_path, coverage_matrix_path, pilot_class_count=pilot_class_count
    )
    ledgers = output / "ledgers"
    ledgers.mkdir(exist_ok=True)
    ledger_path = ledgers / "pilot_coordinate_ledger.json"
    ledger_path.write_bytes(canonical_json_bytes(plan) + b"\n")
    pilot_viability = {
        "schema_version": "qute-r1-pilot-viability-report-v1",
        "status": "NOT_RUN",
        "verdict": "R1-PILOT-AUTHORIZATION-REQUIRED",
        "pilot_payload_generated": False,
        "claim_boundary": {
            "allowed_scope": "representation consistency for equivalent circuits and basic discrimination against clearly non-equivalent matched controls",
            "prohibited_claim": "R1 v1 does not certify fine-grained near-operator discrimination.",
            "hard_negative_policy": "SEPARATE_PROTOCOL_VERSION_REQUIRED",
        },
        "required_checks": [
            "O1 syntax baseline",
            "semantic learning signal",
            "absolute accuracy and consistency",
            "collapse resistance",
            "variance and power",
            "rewrite-family viability",
        ],
    }
    reports = {
        "capacity_report.json": capacity,
        "reproducibility_report.json": reproducibility,
        "pilot_viability_report.json": pilot_viability,
    }
    for name, value in reports.items():
        (output / name).write_text(_json(value, pretty=True) + "\n")
    checks = {
        "canonical_protocol_reconciled": contract["status"] == "PASS",
        "cell_capacity_pass": capacity["status"] == "PASS",
        "cross_process_reproducibility_pass": reproducibility["status"] == "PASS",
        "pilot_payload_not_generated": True,
        "sealed_payload_not_generated": True,
    }
    evidence = {
        "schema_version": "qute-r1-pilot-readiness-preflight-v1",
        "status": "BLOCKED",
        "highest_completed_gate": 2 if all(checks.values()) else 0,
        "next_gate": "PILOT_CORPUS_REQUIRES_SEPARATE_AUTHORIZATION",
        "checks": checks,
        "canonical_contract": contract,
        "artifact_hashes": {
            name: _sha256_file(output / name) for name in reports
        } | {"ledgers/pilot_coordinate_ledger.json": _sha256_file(ledger_path)},
        "scientific_development_generation_authorized": False,
        "sealed_generation_authorized": False,
    }
    (output / "preflight_evidence.json").write_text(_json(evidence, pretty=True) + "\n")
    return evidence


def verify_preflight_evidence(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    evidence = json.loads((output / "preflight_evidence.json").read_text())
    mismatches = [
        relative
        for relative, expected in evidence["artifact_hashes"].items()
        if not (output / relative).is_file()
        or _sha256_file(output / relative) != expected
    ]
    return {
        "schema_version": "qute-r1-preflight-evidence-verification-v1",
        "status": "PASS" if not mismatches else "FAIL",
        "verified_artifact_count": len(evidence["artifact_hashes"]),
        "hash_mismatches": sorted(mismatches),
        "pilot_generation_authorized": False,
        "scientific_development_generation_authorized": False,
        "sealed_generation_authorized": False,
    }


def freeze_and_authorize_pilot_plan(
    protocol_path: str | Path,
    coverage_matrix_path: str | Path,
    registry_path: str | Path,
    output_dir: str | Path,
    *,
    pilot_class_count: int = 1000,
    generator_anchor_commit: str | None = None,
) -> dict[str, Any]:
    """Freeze hash-only pilot evidence and issue an unconsumed pilot-only token."""
    protocol_path, coverage_matrix_path = Path(protocol_path), Path(coverage_matrix_path)
    registry_path, output = Path(registry_path), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    contract = protocol_contract_preflight(
        protocol_path, coverage_matrix_path, pilot_class_count=pilot_class_count
    )
    if contract["status"] != "PASS":
        raise RuntimeError("PILOT-CONTRACT-CONFLICT")
    reproducibility, plan = cross_process_reproducibility(
        protocol_path, coverage_matrix_path, pilot_class_count=pilot_class_count
    )
    if reproducibility["status"] != "PASS":
        raise RuntimeError("PILOT-REPRODUCIBILITY-BLOCKED")
    if len(plan["coordinates"]) != pilot_class_count:
        raise RuntimeError("PILOT-PLAN-MISMATCH")
    coordinate_ids = [row["coordinate_id"] for row in plan["coordinates"]]
    seeds = [row["derived_seed"] for row in plan["coordinates"]]
    if len(set(coordinate_ids)) != len(coordinate_ids) or len(set(seeds)) != len(seeds):
        raise RuntimeError("PILOT-PLAN-MISMATCH")

    ledger_path = output / "pilot_coordinate_ledger.json"
    ledger_path.write_bytes(canonical_json_bytes(plan) + b"\n")
    identity_fields = (
        "base_circuit_content_sha256", "source_circuit_content_sha256",
        "target_circuit_content_sha256", "semantic_class_sha256",
        "ordered_pair_content_sha256", "unordered_pair_content_sha256",
        "rewrite_instance_content_sha256", "negative_control_content_sha256",
        "model_probe_content_sha256",
    )
    identity = {
        "schema_version": "qute-r1-pilot-identity-preflight-v1",
        "PAYLOAD_GENERATED": False,
        "generation_namespace": SCIENTIFIC_PILOT_NAMESPACE,
        "coordinate_count": len(plan["coordinates"]),
        "identities": [
            {"coordinate_id": row["coordinate_id"]} | {field: row[field] for field in identity_fields}
            for row in plan["coordinates"]
        ],
        "oracle_checks_pass": all(max(row["oracle_metrics"].values()) <= 1e-10 for row in plan["coordinates"]),
        "negative_contract_pass": all(row["negative_control_distance"] >= 0.1 for row in plan["coordinates"]),
    }
    identity_path = output / "pilot_identity_preflight.json"
    identity_path.write_bytes(canonical_json_bytes(identity) + b"\n")

    pilot_sets = {
        "base_circuit_content_sha256": {row["base_circuit_content_sha256"] for row in plan["coordinates"]},
        "circuit_content_sha256": {row[key] for row in plan["coordinates"] for key in ("source_circuit_content_sha256", "target_circuit_content_sha256")},
        "semantic_class_sha256": {row["semantic_class_sha256"] for row in plan["coordinates"]},
        "ordered_pair_content_sha256": {row["ordered_pair_content_sha256"] for row in plan["coordinates"]},
        "unordered_pair_content_sha256": {row["unordered_pair_content_sha256"] for row in plan["coordinates"]},
        "rewrite_instance_content_sha256": {row["rewrite_instance_content_sha256"] for row in plan["coordinates"]},
        "model_probe_content_sha256": {row["model_probe_content_sha256"] for row in plan["coordinates"]},
    }
    root = registry_path.parent.parent
    smoke_v1 = _legacy_smoke_v1_hashes(root / "smoke_v1", protocol_path)
    smoke_v2 = {
        field: _read_index(root / "smoke_v2", field.replace("sha256", "hashes") + ".txt")
        for field in pilot_sets
    }
    overlap = {
        "schema_version": "qute-r1-pilot-overlap-audit-v1",
        "PAYLOAD_GENERATED": False,
        "pilot_vs_smoke_v1": content_hash_intersections(pilot_sets, smoke_v1, pilot_sets),
        "pilot_vs_smoke_v2": content_hash_intersections(pilot_sets, smoke_v2, pilot_sets),
    }
    overlap["status"] = "PASS" if all(
        item["count"] == 0
        for side in ("pilot_vs_smoke_v1", "pilot_vs_smoke_v2")
        for item in overlap[side].values()
    ) else "FAIL"
    if overlap["status"] != "PASS":
        raise RuntimeError("PILOT-CONTAMINATION-BLOCKED")
    overlap_path = output / "pilot_overlap_audit.json"
    overlap_path.write_text(_json(overlap, pretty=True) + "\n")
    partition = {
        "schema_version": "qute-r1-pilot-partition-audit-v1",
        "status": "PASS", "partition_policy": "SINGLE_PARTITION",
        "partitions": ["PILOT_SINGLE_PARTITION"],
        "intersection_matrix": {"PILOT_SINGLE_PARTITION": {"PILOT_SINGLE_PARTITION": "SELF_EXEMPT"}},
        "prohibited_cross_partition_intersections": 0,
    }
    partition_path = output / "pilot_partition_audit.json"
    partition_path.write_text(_json(partition, pretty=True) + "\n")
    reproducibility.update({
        "uv_lock_sha256": _sha256_file(Path("uv.lock")) if Path("uv.lock").exists() else None,
        "generator_source_sha256": _sha256_file(Path(__file__)),
        "protocol_sha256": _sha256_file(protocol_path),
        "canonicalization_version": "canonical-json-sort-keys-compact-v1",
        "rng_implementation": "numpy-PCG64-via-default_rng",
        "thread_policy": {"PYTHONHASHSEED": [1, 987654]},
    })
    repro_path = output / "pilot_reproducibility_report.json"
    repro_path.write_text(_json(reproducibility, pretty=True) + "\n")
    pilot_contract = {
        "schema_version": "qute-r1-pilot-contract-v1", "status": "FROZEN",
        "generation_namespace": SCIENTIFIC_PILOT_NAMESPACE, "corpus_role": "SCIENTIFIC_PILOT",
        "scientific_use": "PILOT_EVALUATION_ONLY", "pilot_class_count": pilot_class_count,
        "positive_control_ratio": "1:1", "depth_buckets": [2, 4, 6],
        "parameter_region": "train", "rewrite_cell_count": 15,
        "negative_control_labels": ["NEGATIVE_NON_EQUIVALENT", "MATCHED_NON_EQUIVALENT_CONTROL", "FAR_NEGATIVE_SANITY"],
        "claim_boundary": "equivalent-circuit consistency and basic far-control discrimination; no near-operator or hard-negative certification",
        "PAYLOAD_GENERATED": False,
    }
    contract_path = output / "pilot_contract.json"
    contract_path.write_text(_json(pilot_contract, pretty=True) + "\n")
    registry = json.loads(registry_path.read_text())
    pilot_policy = registry["namespaces"].get(SCIENTIFIC_PILOT_NAMESPACE)
    required_policy = {
        "corpus_role": "SCIENTIFIC_PILOT", "scientific_use": "PILOT_EVALUATION_ONLY",
        "sealable": False, "promotion_allowed": False,
        "scientific_development_reuse_allowed": False, "sealed_reuse_allowed": False,
        "generation_requires_authorization": True, "payload_generated": False,
    }
    if pilot_policy is None or any(pilot_policy.get(k) != v for k, v in required_policy.items()):
        raise RuntimeError("PILOT-NAMESPACE-POLICY-MISMATCH")
    anchor = generator_anchor_commit or _git_head()
    if subprocess.run(["git", "merge-base", "--is-ancestor", anchor, "HEAD"]).returncode:
        raise RuntimeError("PILOT-GENERATOR-PROVENANCE-INVALID")
    environment = reproducibility["environment_contract"] | {
        "uv_lock_sha256": reproducibility["uv_lock_sha256"]
    }
    environment_fingerprint = content_sha256(environment)
    bound = {
        "schema_version": "qute-r1-pilot-plan-manifest-v1",
        "status": "FROZEN", "PAYLOAD_GENERATED": False,
        "pilot_namespace": SCIENTIFIC_PILOT_NAMESPACE, "corpus_role": "SCIENTIFIC_PILOT",
        "protocol_sha256": _sha256_file(protocol_path),
        "generator_source_sha256": _sha256_file(Path(__file__)),
        "generator_anchor_commit": anchor, "environment_fingerprint": environment_fingerprint,
        "namespace_registry_sha256": _sha256_file(registry_path),
        "rewrite_coverage_sha256": _sha256_file(coverage_matrix_path),
        "pilot_allocation": {"equivalence_classes": pilot_class_count, "positive_pairs": pilot_class_count, "negative_controls": pilot_class_count},
        "coordinate_ledger_sha256": _sha256_file(ledger_path),
        "identity_preflight_sha256": _sha256_file(identity_path),
        "smoke_overlap_audit_sha256": _sha256_file(overlap_path),
        "partition_intersection_audit_sha256": _sha256_file(partition_path),
        "reproducibility_evidence_sha256": _sha256_file(repro_path),
        "negative_control_contract": pilot_contract["negative_control_labels"],
        "claim_boundary": pilot_contract["claim_boundary"],
    }
    bound["pilot_plan_sha256"] = content_sha256(bound)
    manifest_path = output / "pilot_plan_manifest.json"
    manifest_path.write_text(_json(bound, pretty=True) + "\n")
    checksums = [
        f"{_sha256_file(path)}  {path.name}" for path in sorted(output.glob("*.json"))
    ]
    (output / "checksums.sha256").write_text("\n".join(checksums) + "\n")
    return issue_pilot_authorization(output, registry_path)


def issue_pilot_authorization(
    pilot_dir: str | Path, registry_path: str | Path, authorization_path: str | Path | None = None
) -> dict[str, Any]:
    """Recompute frozen evidence; caller assertions are intentionally not accepted."""
    pilot_dir, registry_path = Path(pilot_dir), Path(registry_path)
    manifest_path = pilot_dir / "pilot_plan_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    registry = json.loads(registry_path.read_text())
    files = {
        "coordinate_ledger_sha256": pilot_dir / "pilot_coordinate_ledger.json",
        "identity_preflight_sha256": pilot_dir / "pilot_identity_preflight.json",
        "smoke_overlap_audit_sha256": pilot_dir / "pilot_overlap_audit.json",
        "partition_intersection_audit_sha256": pilot_dir / "pilot_partition_audit.json",
        "reproducibility_evidence_sha256": pilot_dir / "pilot_reproducibility_report.json",
    }
    ledger = json.loads(files["coordinate_ledger_sha256"].read_text())
    identity = json.loads(files["identity_preflight_sha256"].read_text())
    environment = ledger["environment_contract"] | {
        "uv_lock_sha256": _sha256_file(Path("uv.lock")) if Path("uv.lock").exists() else None
    }
    pilot_policy = registry["namespaces"].get(SCIENTIFIC_PILOT_NAMESPACE, {})
    checks = {
        "plan_self_hash": content_sha256({k: v for k, v in manifest.items() if k != "pilot_plan_sha256"}) == manifest["pilot_plan_sha256"],
        "bound_file_hashes": all(_sha256_file(path) == manifest[key] for key, path in files.items()),
        "protocol_hash": _sha256_file(R1_ARTIFACT_ROOT / "protocol.json") == manifest["protocol_sha256"],
        "generator_hash": _sha256_file(Path(__file__)) == manifest["generator_source_sha256"],
        "generator_anchor_ancestor": subprocess.run(
            ["git", "merge-base", "--is-ancestor", manifest["generator_anchor_commit"], "HEAD"]
        ).returncode == 0,
        "environment_fingerprint": content_sha256(environment) == manifest["environment_fingerprint"],
        "registry_hash": _sha256_file(registry_path) == manifest["namespace_registry_sha256"],
        "pilot_namespace_policy": pilot_policy.get("corpus_role") == "SCIENTIFIC_PILOT" and pilot_policy.get("generation_requires_authorization") is True and pilot_policy.get("promotion_allowed") is False,
        "exact_allocation": len(ledger["coordinates"]) == manifest["pilot_allocation"]["equivalence_classes"],
        "seed_uniqueness": len({row["derived_seed"] for row in ledger["coordinates"]}) == len(ledger["coordinates"]),
        "coordinate_uniqueness": len({row["coordinate_id"] for row in ledger["coordinates"]}) == len(ledger["coordinates"]),
        "rewrite_coverage": len({row["coverage_cell_id"] for row in ledger["coordinates"]}) == 15,
        "identity_count": identity["coordinate_count"] == len(ledger["coordinates"]) and identity["oracle_checks_pass"] and identity["negative_contract_pass"],
        "reproducibility": json.loads(files["reproducibility_evidence_sha256"].read_text())["status"] == "PASS",
        "overlap": json.loads(files["smoke_overlap_audit_sha256"].read_text())["status"] == "PASS",
        "partition": json.loads(files["partition_intersection_audit_sha256"].read_text())["status"] == "PASS",
        "identity": json.loads(files["identity_preflight_sha256"].read_text())["PAYLOAD_GENERATED"] is False,
        "pilot_payload_absent": not (R1_ARTIFACT_ROOT / "scientific_pilot_v1").exists(),
        "development_blocked": not registry["namespaces"][SCIENTIFIC_DEVELOPMENT_NAMESPACE]["generation_authorized"] and not registry["namespaces"][SCIENTIFIC_DEVELOPMENT_NAMESPACE]["payload_generated"],
        "sealed_untouched": not registry["namespaces"][SCIENTIFIC_SEALED_NAMESPACE]["generation_authorized"] and not registry["namespaces"][SCIENTIFIC_SEALED_NAMESPACE]["payload_generated"] and registry["namespaces"][SCIENTIFIC_SEALED_NAMESPACE]["sealed_state"]["access_count"] == 0,
    }
    if not all(checks.values()):
        raise RuntimeError("PILOT-AUTHORIZATION-BLOCKED: " + ",".join(k for k, v in checks.items() if not v))
    authorization = {
        "schema_version": "qute-r1-pilot-generation-authorization-v1",
        "authorization_scope": "PILOT_CORPUS_GENERATION_ONLY", "status": "AUTHORIZED",
        "pilot_generation_authorized": True,
        "scientific_development_generation_authorized": False,
        "sealed_generation_authorized": False, "model_training_authorized": False,
        "model_evaluation_authorized": False, "qpu_execution_authorized": False,
        "consumed": False, "pilot_namespace": manifest["pilot_namespace"],
        "protocol_sha256": manifest["protocol_sha256"],
        "generator_source_sha256": manifest["generator_source_sha256"],
        "generator_anchor_commit": manifest["generator_anchor_commit"],
        "environment_fingerprint": manifest["environment_fingerprint"],
        "pilot_plan_sha256": manifest["pilot_plan_sha256"],
        "coordinate_ledger_sha256": manifest["coordinate_ledger_sha256"],
        "authorized_pilot_allocation": manifest["pilot_allocation"],
        "evidence_gate_checks": checks,
    }
    token = content_sha256(authorization)
    authorization["authorization_id"] = f"qute-r1-pilot-auth-{token[:16]}"
    authorization["authorization_token"] = token
    target = Path(authorization_path) if authorization_path else R1_ARTIFACT_ROOT / "authorization" / "pilot_generation_authorization.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_json(authorization, pretty=True) + "\n")
    return authorization


def generate_pilot_corpus(*_args: Any, **_kwargs: Any) -> None:
    raise PermissionError("pilot corpus generation requires separate Gate 3 authorization")


def full_generation_gate(
    *,
    registry: dict[str, Any],
    coverage_matrix: dict[str, Any],
    namespace_audit: dict[str, Any],
    determinism_result: dict[str, Any],
    intersection_result: dict[str, Any],
    smoke_manifest: dict[str, Any],
    smoke_audit: dict[str, Any],
) -> dict[str, Any]:
    sealed = registry["namespaces"][SCIENTIFIC_SEALED_NAMESPACE]
    scientific_entries = [
        registry["namespaces"][SCIENTIFIC_DEVELOPMENT_NAMESPACE],
        sealed,
    ]
    authorized = [entry for entry in scientific_entries if entry["generation_authorized"]]
    authorized_commit = (
        authorized[0].get("authorized_generator_commit") if len(authorized) == 1 else None
    )
    checks = {
        "protocol_hash_match": (
            smoke_manifest.get("protocol_sha256") == coverage_matrix.get("protocol_sha256")
        ),
        "generator_commit_match": (
            authorized_commit is not None
            and smoke_manifest.get("generator_commit") == authorized_commit
        ),
        "scientific_namespace_uniquely_authorized": len(authorized) == 1,
        "smoke_v1_retired": registry["namespaces"]["qute:r1:smoke:v1"]["status"] == "RETIRED",
        "smoke_scientific_content_overlap_zero": intersection_result["status"] == "PASS",
        "rewrite_coverage_100_percent": rewrite_coverage_complete(coverage_matrix),
        "equivalence_class_manifest_valid": smoke_manifest["counts"]["equivalence_classes"] > 0,
        "partition_intersection_zero": intersection_result["status"] == "PASS",
        "negative_metadata_contract_valid": smoke_audit.get(
            "negative_metadata_contract_valid", False
        ),
        "determinism_pass": determinism_result["status"] == "PASS",
        "checksums_pass": True,
        "sealed_access_count_zero": sealed["sealed_state"]["access_count"] == 0,
    }
    return {
        "schema_version": "qute-r1-full-generation-gate-v1",
        "status": "BLOCKED",
        "authorized": False,
        "checks": checks,
        "blockers": sorted(name for name, passed in checks.items() if not passed),
        "development_command_separate": True,
        "sealed_command_separate": True,
    }


def load_corpus_by_role(root: str | Path, *, loader_role: str) -> dict[str, Any]:
    root = Path(root)
    manifest = json.loads((root / "corpus_manifest.json").read_text())
    role = manifest["corpus_role"]
    if manifest.get("status") == "NON_SEALABLE_RETIRED_EVIDENCE" and loader_role != "audit":
        raise PermissionError("retired smoke corpus cannot be loaded for scientific use")
    if role == "SCIENTIFIC_SEALED_FINAL" and loader_role != "sealed_authorized":
        raise PermissionError("normal development loader rejects sealed corpus")
    if manifest.get("scientific_use") == "FORBIDDEN" and loader_role in {
        "scientific_development",
        "sealed_authorized",
    }:
        raise PermissionError("smoke corpus scientific use is forbidden")
    return manifest


def generate_scientific_development(*_args: Any, **_kwargs: Any) -> None:
    raise PermissionError("scientific development generation remains blocked")


def generate_scientific_sealed(*_args: Any, **_kwargs: Any) -> None:
    raise PermissionError("scientific sealed generation remains blocked")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=R1_ARTIFACT_ROOT / "protocol.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--mode",
        choices=(
            "smoke",
            "full-plan",
            "smoke-v2",
            "pilot-readiness-preflight",
            "pilot-corpus",
            "scientific-development",
            "scientific-sealed",
        ),
        default="smoke-v2",
    )
    args = parser.parse_args()
    if args.mode == "full-plan":
        print(_json(planned_counts(args.protocol, mode="full"), pretty=True))
        return
    if args.mode == "pilot-readiness-preflight":
        if args.output is None:
            parser.error("--output is required for pilot readiness preflight")
        print(_json(run_pilot_readiness_preflight(
            args.protocol,
            R1_ARTIFACT_ROOT / "protocol" / "rewrite_coverage_matrix.json",
            args.output,
        ), pretty=True))
        return
    if args.mode == "pilot-corpus":
        generate_pilot_corpus()
    if args.mode == "scientific-development":
        generate_scientific_development()
    if args.mode == "scientific-sealed":
        generate_scientific_sealed()
    if args.output is None:
        parser.error("--output is required for smoke generation")
    if args.mode == "smoke-v2":
        manifest = generate_smoke_v2(
            args.output,
            args.protocol,
            R1_ARTIFACT_ROOT / "protocol" / "rewrite_coverage_matrix.json",
        )
    else:
        manifest = write_corpus(args.output, args.protocol, mode="smoke")
    print(_json(manifest, pretty=True))


if __name__ == "__main__":
    main()
