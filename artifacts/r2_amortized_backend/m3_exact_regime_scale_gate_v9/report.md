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
      "median_energy_error_per_edge": 0.0018061884911730886,
      "median_marginal_mae": 0.002819061279296875,
      "median_zz_mae": 0.005693789105862379
    },
    "20": {
      "median_energy_error_per_edge": 0.0018218993791379035,
      "median_marginal_mae": 0.002808380057103932,
      "median_zz_mae": 0.005696106003597379
    },
    "22": {
      "median_energy_error_per_edge": 0.0021221276838332415,
      "median_marginal_mae": 0.00283362646587193,
      "median_zz_mae": 0.0062223493587225676
    },
    "24": {
      "median_energy_error_per_edge": 0.0020533667411655188,
      "median_marginal_mae": 0.0029010772705078125,
      "median_zz_mae": 0.005806816974654794
    }
  },
  "overall": {
    "median_energy_error_per_edge": 0.0019297281978651881,
    "median_exact_tvd_18q": 0.019399236865158487,
    "median_marginal_mae": 0.0028369498904794455,
    "median_zz_mae": 0.0057922364212572575
  }
}
```

## Systems

- Worst warm 4,096-shot median latency: 23.738 ms.
- Maximum incremental peak RSS: 14.25 MiB.
- Per-circuit optimizer steps and neural-route exact calls remained zero.

## Boundary

Exact statevector teacher/validation completed through 24Q. M2 remains the classical boundary reference: statevector was resource-guarded at 26Q; MPS cycle reached 32Q, random 3-regular p=1 reached 28Q, and random 3-regular p=2/3 hit the 120-second guard at 20Q.

## Interpretation

A failed model gate means iterate inside M3; it does not justify QPU data. A pass means only that the exact-regime model is ready for the next simulator-side gate.
