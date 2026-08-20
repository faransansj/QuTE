# CC-NQE P4.6-A Report

- branch: `research/cc-nqe-p4-6-compositional-ood`
- HEAD: `e0dcd05ef7a96b94d2db83eec8d6e7fcd5e43647`
- G0: PASS
- G1: PASS
- G2: **G2-PASS**

## OOD-v2 exact split definitions

`train` and `iid_validation`: depths 1–6, product/random-local states, train parameter regions. `state_ood_validation`: the IID operators with fresh entangled/Haar-random states. `parameter_ood_validation`: depths 1–6 with separate interpolation and extrapolation regions. `composition_ood_validation` and `composition_ood_test_sealed`: controlled depths 4–6 with disjoint motif families. `depth_ood_validation`: depth 7. `depth_ood_test_sealed`: depths 8–10.

- State-OOD: {'state_ood_validation': ['entangled', 'Haar-random'], 'train_and_iid': ['product', 'random-local']}
- Parameter-OOD: {'parameter_ood_extrapolation': [[5.235987755982989, 6.283185307179586]], 'parameter_ood_interpolation': [[2.0943951023931953, 3.141592653589793]], 'train_composition_depth_state': [[0, 2.0943951023931953], [3.141592653589793, 5.235987755982989]]}
- Motif families: {'composition_ood_test_sealed': [8, 9], 'composition_ood_validation': [6, 7], 'train': [0, 1, 2, 3, 4, 5]}
- Motif k=2/3/4 coverage: see `datasets/audit.json` → `motif_coverage` (exact counts per split/family).
- Depth contract: {'depth_ood_test_sealed': [8, 9, 10], 'depth_ood_validation': [7], 'train': [1, 2, 3, 4, 5, 6]}

## Leakage audit

All hard checks: **PASS**

```json
{
  "composition_structural_leakage": true,
  "composition_val_test_disjoint": true,
  "depth_separation": true,
  "deterministic_regeneration": true,
  "exact_sample_duplicates": true,
  "hash_consistency": true,
  "manifest_consistency": true,
  "parameter_region_isolation": true,
  "sample_counts": true,
  "sealed_test_immutability": true,
  "state_family_isolation": true,
  "state_id_leakage": true,
  "state_ood_operator_matched": true,
  "state_target_normalization": true,
  "teacher_unitary_unitarity": true
}
```

Sealed-test access count: **0**.

## Diversity

Exact/structural/k2/k3/k4 counts, parameter bins, and interaction coverage are frozen per split in `datasets/audit.json` → `diversity`.

## Track-A resource estimates

Free disk: 78733922304 bytes. Maximum arm requirement: 432399999 bytes. Sufficient: **True**. Per-arm byte/RAM/time estimates: `datasets/resource_estimate.json`. Storage is sharded/memory-mapped; direct-state arms exclude unitary targets.

## Frozen Track-B protocol

{'B0': {'model': 'direct state', 'supervision': 'action-only'}, 'B1': {'model': 'unconstrained operator', 'supervision': 'action-only'}, 'B2': {'model': 'soft-unitarity operator', 'supervision': 'action-only + unitarity regularizer'}, 'B3': {'model': 'exact-unitarity exp((B-B†)/2)', 'supervision': 'action-only'}, 'B4': {'model': 'direct-U', 'supervision': 'PRIVILEGED OPERATOR SUPERVISION'}, 'B5': {'model': 'direct-U + independent operator composition outputs', 'supervision': 'PRIVILEGED OPERATOR SUPERVISION'}}

Normalized action fidelity: `|<target|M psi>|^2/(||target||^2 ||M psi||^2); raw ||M psi|| recorded`. Primary allocation: `{'name': 'A4', 'pairs': 1000008, 'probes_per_circuit': 17, 'unique_circuits': 58824}`.

## Frozen recurrent protocol

psi_hat[0]=input; psi_hat[i+1]=T_theta(G_i,psi_hat[i]). Prefix targets: exact intermediate states are targets only. `lambda_prefix=1.0`. Tautological C4 is excluded.

## Negative control

`test_composition_consistency_is_tautological_negative_control_for_shared_recurrence` proves the old shared-recurrence state-composition loss is zero by construction. It is not a Track-C intervention.

## Synthetic composition benchmark

Status: **COMPOSITION-SANITY-PASS**. Results: `synthetic/summary.json`. The deterministic noncommutative generator task covers ordered products, prefixes, length 7, and synthetic lengths 8–10 only.

## Stop decision

P4.6-B Track A may begin after human review: **YES**. Track A/B/C, seed confirmation, and sealed quantum tests were not started.
