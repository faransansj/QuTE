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
      "median_energy_error_per_edge": 0.0010234691726509482,
      "median_marginal_mae": 0.0032513936748728156,
      "median_zz_mae": 0.005336620146408677
    },
    "20": {
      "median_energy_error_per_edge": 0.0010167440050281584,
      "median_marginal_mae": 0.0032356262672692537,
      "median_zz_mae": 0.005499776219949126
    },
    "22": {
      "median_energy_error_per_edge": 0.0013000026228837669,
      "median_marginal_mae": 0.003406524658203125,
      "median_zz_mae": 0.00586076220497489
    },
    "24": {
      "median_energy_error_per_edge": 0.0010121663799509406,
      "median_marginal_mae": 0.0031461715698242188,
      "median_zz_mae": 0.005385081050917506
    }
  },
  "overall": {
    "median_energy_error_per_edge": 0.001102177775464952,
    "median_exact_tvd_18q": 0.027149390149633836,
    "median_marginal_mae": 0.003273391746915877,
    "median_zz_mae": 0.005488981958478689
  }
}
```

## Systems

- Worst warm 4,096-shot median latency: 23.599 ms.
- Maximum incremental peak RSS: 9.17 MiB.
- Per-circuit optimizer steps and neural-route exact calls remained zero.

## Boundary

Exact statevector teacher/validation completed through 24Q. M2 remains the classical boundary reference: statevector was resource-guarded at 26Q; MPS cycle reached 32Q, random 3-regular p=1 reached 28Q, and random 3-regular p=2/3 hit the 120-second guard at 20Q.

## Interpretation

A failed model gate means iterate inside M3; it does not justify QPU data. A pass means only that the exact-regime model is ready for the next simulator-side gate.
