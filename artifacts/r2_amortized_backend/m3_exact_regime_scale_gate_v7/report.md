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
      "median_energy_error_per_edge": 0.005283214384689927,
      "median_marginal_mae": 0.0026156108360737562,
      "median_zz_mae": 0.01079926686361432
    },
    "20": {
      "median_energy_error_per_edge": 0.005749511765316129,
      "median_marginal_mae": 0.0025482177734375,
      "median_zz_mae": 0.011622110847383738
    },
    "22": {
      "median_energy_error_per_edge": 0.005752794677391648,
      "median_marginal_mae": 0.0025839372538030148,
      "median_zz_mae": 0.012200558092445135
    },
    "24": {
      "median_energy_error_per_edge": 0.0066360896453261375,
      "median_marginal_mae": 0.0025746028404682875,
      "median_zz_mae": 0.013272179290652275
    }
  },
  "overall": {
    "median_energy_error_per_edge": 0.006230151979252696,
    "median_exact_tvd_18q": 0.09842062168676974,
    "median_marginal_mae": 0.0025872548576444387,
    "median_zz_mae": 0.012637837789952755
  }
}
```

## Systems

- Worst warm 4,096-shot median latency: 24.277 ms.
- Maximum incremental peak RSS: 29.72 MiB.
- Per-circuit optimizer steps and neural-route exact calls remained zero.

## Boundary

Exact statevector teacher/validation completed through 24Q. M2 remains the classical boundary reference: statevector was resource-guarded at 26Q; MPS cycle reached 32Q, random 3-regular p=1 reached 28Q, and random 3-regular p=2/3 hit the 120-second guard at 20Q.

## Interpretation

A failed model gate means iterate inside M3; it does not justify QPU data. A pass means only that the exact-regime model is ready for the next simulator-side gate.
