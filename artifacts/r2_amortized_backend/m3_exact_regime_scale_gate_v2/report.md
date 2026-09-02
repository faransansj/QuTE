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
| `robust_median_zz_mae` | FAIL |
| `robust_p90_zz_mae` | FAIL |

## Accuracy

```json
{
  "by_width": {
    "18": {
      "median_energy_error_per_edge": 0.013397216796875,
      "median_marginal_mae": 0.0038011339493095875,
      "median_zz_mae": 0.02679443359375
    },
    "20": {
      "median_energy_error_per_edge": 0.011313628870993853,
      "median_marginal_mae": 0.00435562152415514,
      "median_zz_mae": 0.022627257741987705
    },
    "22": {
      "median_energy_error_per_edge": 0.007828452391549945,
      "median_marginal_mae": 0.004398345947265625,
      "median_zz_mae": 0.01577203953638673
    },
    "24": {
      "median_energy_error_per_edge": 0.006927490234375,
      "median_marginal_mae": 0.0048106510657817125,
      "median_zz_mae": 0.01385498046875
    }
  },
  "overall": {
    "median_energy_error_per_edge": 0.009135041385889053,
    "median_exact_tvd_18q": 0.08173775113190243,
    "median_marginal_mae": 0.00435562152415514,
    "median_zz_mae": 0.018270082771778107
  }
}
```

## Systems

- Worst warm 4,096-shot median latency: 9.005 ms.
- Maximum incremental peak RSS: 4.28 MiB.
- Per-circuit optimizer steps and neural-route exact calls remained zero.

## Boundary

Exact statevector teacher/validation completed through 24Q. M2 remains the classical boundary reference: statevector was resource-guarded at 26Q; MPS cycle reached 32Q, random 3-regular p=1 reached 28Q, and random 3-regular p=2/3 hit the 120-second guard at 20Q.

## Interpretation

A failed model gate means iterate inside M3; it does not justify QPU data. A pass means only that the exact-regime model is ready for the next simulator-side gate.
