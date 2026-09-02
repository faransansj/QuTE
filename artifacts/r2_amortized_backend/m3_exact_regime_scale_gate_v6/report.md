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
| `robust_p90_energy_error_per_edge` | FAIL |
| `robust_median_marginal_mae` | PASS |
| `robust_p90_marginal_mae` | PASS |
| `robust_median_zz_mae` | PASS |
| `robust_p90_zz_mae` | FAIL |

## Accuracy

```json
{
  "by_width": {
    "18": {
      "median_energy_error_per_edge": 0.0023247047793120146,
      "median_marginal_mae": 0.006793552078306675,
      "median_zz_mae": 0.006161724915727973
    },
    "20": {
      "median_energy_error_per_edge": 0.0024052937515079975,
      "median_marginal_mae": 0.006516647525131702,
      "median_zz_mae": 0.0062311808578670025
    },
    "22": {
      "median_energy_error_per_edge": 0.0023586389143019915,
      "median_marginal_mae": 0.006924022454768419,
      "median_zz_mae": 0.006622776621952653
    },
    "24": {
      "median_energy_error_per_edge": 0.002089818357490003,
      "median_marginal_mae": 0.00672149658203125,
      "median_zz_mae": 0.006070031085982919
    }
  },
  "overall": {
    "median_energy_error_per_edge": 0.0022745614405721426,
    "median_exact_tvd_18q": 0.030375317074771715,
    "median_marginal_mae": 0.006751124048605561,
    "median_zz_mae": 0.006275523919612169
  }
}
```

## Systems

- Worst warm 4,096-shot median latency: 45.852 ms.
- Maximum incremental peak RSS: 13.17 MiB.
- Per-circuit optimizer steps and neural-route exact calls remained zero.

## Boundary

Exact statevector teacher/validation completed through 24Q. M2 remains the classical boundary reference: statevector was resource-guarded at 26Q; MPS cycle reached 32Q, random 3-regular p=1 reached 28Q, and random 3-regular p=2/3 hit the 120-second guard at 20Q.

## Interpretation

A failed model gate means iterate inside M3; it does not justify QPU data. A pass means only that the exact-regime model is ready for the next simulator-side gate.
