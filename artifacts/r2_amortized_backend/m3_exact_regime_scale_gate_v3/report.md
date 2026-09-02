# M3 Exact-Regime Scale Gate Report

**Decision:** `M3_NEEDS_ITERATION`
**Training stage reached:** `full`
**Role:** exploratory, simulator-only; no QPU authorization

## Gate checks

| Gate | Result |
|---|---|
| `full_10000_circuit_corpus` | PASS |
| `energy_error_per_edge` | PASS |
| `marginal_mae` | PASS |
| `zz_mae` | PASS |
| `tvd_18q` | PASS |
| `latency` | PASS |
| `memory` | FAIL |
| `optimizer_steps_zero` | PASS |
| `exact_calls_zero` | PASS |
| `latency_width_trend` | PASS |
| `energy_width_trend` | PASS |
| `validation_boundary_24q` | PASS |
| `robust_median_energy_error_per_edge` | PASS |
| `robust_p90_energy_error_per_edge` | FAIL |
| `robust_median_marginal_mae` | PASS |
| `robust_p90_marginal_mae` | PASS |
| `robust_median_zz_mae` | FAIL |
| `robust_p90_zz_mae` | FAIL |

## Accuracy

```json
{
  "by_width": {
    "18": {
      "median_energy_error_per_edge": 0.0024784229462966323,
      "median_marginal_mae": 0.002536349813453853,
      "median_zz_mae": 0.0056446215603500605
    },
    "20": {
      "median_energy_error_per_edge": 0.0026270549278706312,
      "median_marginal_mae": 0.0024742125533521175,
      "median_zz_mae": 0.0060134888626635075
    },
    "22": {
      "median_energy_error_per_edge": 0.0023244222393259406,
      "median_marginal_mae": 0.0025440562749281526,
      "median_zz_mae": 0.0060665246564894915
    },
    "24": {
      "median_energy_error_per_edge": 0.002094480791129172,
      "median_marginal_mae": 0.0024601618060842156,
      "median_zz_mae": 0.005842844722792506
    }
  },
  "overall": {
    "median_energy_error_per_edge": 0.00232228368986398,
    "median_exact_tvd_18q": 0.01635744049668962,
    "median_marginal_mae": 0.0025278726825490594,
    "median_zz_mae": 0.0060015590861439705
  }
}
```

## Systems

- Worst warm 4,096-shot median latency: 51.925 ms.
- Maximum incremental peak RSS: 556.75 MiB.
- Per-circuit optimizer steps and neural-route exact calls remained zero.

## Boundary

Exact statevector teacher/validation completed through 24Q. M2 remains the classical boundary reference: statevector was resource-guarded at 26Q; MPS cycle reached 32Q, random 3-regular p=1 reached 28Q, and random 3-regular p=2/3 hit the 120-second guard at 20Q.

## Interpretation

A failed model gate means iterate inside M3; it does not justify QPU data. A pass means only that the exact-regime model is ready for the next simulator-side gate.
