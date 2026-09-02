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
| `memory` | PASS |
| `optimizer_steps_zero` | PASS |
| `exact_calls_zero` | PASS |
| `latency_width_trend` | PASS |
| `energy_width_trend` | PASS |
| `validation_boundary_24q` | PASS |
| `robust_median_energy_error_per_edge` | PASS |
| `robust_p90_energy_error_per_edge` | PASS |
| `robust_median_marginal_mae` | PASS |
| `robust_p90_marginal_mae` | PASS |
| `robust_median_zz_mae` | PASS |
| `robust_p90_zz_mae` | FAIL |

## Accuracy

```json
{
  "by_width": {
    "18": {
      "median_energy_error_per_edge": 0.0009084630873985589,
      "median_marginal_mae": 0.0029296875,
      "median_zz_mae": 0.0054880776442587376
    },
    "20": {
      "median_energy_error_per_edge": 0.0011990865459665656,
      "median_marginal_mae": 0.0030456542735919356,
      "median_zz_mae": 0.005604553036391735
    },
    "22": {
      "median_energy_error_per_edge": 0.0012932979734614491,
      "median_marginal_mae": 0.0031117525650188327,
      "median_zz_mae": 0.006138194585219026
    },
    "24": {
      "median_energy_error_per_edge": 0.0013557010679505765,
      "median_marginal_mae": 0.0030094782123342156,
      "median_zz_mae": 0.005557166179642081
    }
  },
  "overall": {
    "median_energy_error_per_edge": 0.0011892030015587807,
    "median_exact_tvd_18q": 0.02432726988881425,
    "median_marginal_mae": 0.0030659569893032312,
    "median_zz_mae": 0.0057017006911337376
  }
}
```

## Systems

- Worst warm 4,096-shot median latency: 23.598 ms.
- Maximum incremental peak RSS: 9.16 MiB.
- Per-circuit optimizer steps and neural-route exact calls remained zero.

## Boundary

Exact statevector teacher/validation completed through 24Q. M2 remains the classical boundary reference: statevector was resource-guarded at 26Q; MPS cycle reached 32Q, random 3-regular p=1 reached 28Q, and random 3-regular p=2/3 hit the 120-second guard at 20Q.

## Interpretation

A failed model gate means iterate inside M3; it does not justify QPU data. A pass means only that the exact-regime model is ready for the next simulator-side gate.
