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
      "median_energy_error_per_edge": 0.0027177599258720875,
      "median_marginal_mae": 0.0029852124862372875,
      "median_zz_mae": 0.0067765978164970875
    },
    "20": {
      "median_energy_error_per_edge": 0.0027755737537518144,
      "median_marginal_mae": 0.0028980255592614412,
      "median_zz_mae": 0.006642659427598119
    },
    "22": {
      "median_energy_error_per_edge": 0.003252202761359513,
      "median_marginal_mae": 0.00272681494243443,
      "median_zz_mae": 0.007814349606633186
    },
    "24": {
      "median_energy_error_per_edge": 0.0028815799159929156,
      "median_marginal_mae": 0.0029112498741596937,
      "median_zz_mae": 0.007311079418286681
    }
  },
  "overall": {
    "median_energy_error_per_edge": 0.002960205078125,
    "median_exact_tvd_18q": 0.022894046223089486,
    "median_marginal_mae": 0.0028769811615347862,
    "median_zz_mae": 0.0074010908138006926
  }
}
```

## Systems

- Worst warm 4,096-shot median latency: 42.874 ms.
- Maximum incremental peak RSS: 38.47 MiB.
- Per-circuit optimizer steps and neural-route exact calls remained zero.

## Boundary

Exact statevector teacher/validation completed through 24Q. M2 remains the classical boundary reference: statevector was resource-guarded at 26Q; MPS cycle reached 32Q, random 3-regular p=1 reached 28Q, and random 3-regular p=2/3 hit the 120-second guard at 20Q.

## Interpretation

A failed model gate means iterate inside M3; it does not justify QPU data. A pass means only that the exact-regime model is ready for the next simulator-side gate.
