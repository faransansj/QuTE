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
      "median_energy_error_per_edge": 0.003132855403237045,
      "median_marginal_mae": 0.0026427374687045813,
      "median_zz_mae": 0.007450245087966323
    },
    "20": {
      "median_energy_error_per_edge": 0.0027257284382358193,
      "median_marginal_mae": 0.0025955201126635075,
      "median_zz_mae": 0.007213846780359745
    },
    "22": {
      "median_energy_error_per_edge": 0.003195329220034182,
      "median_marginal_mae": 0.0027399930404499173,
      "median_zz_mae": 0.008266564458608627
    },
    "24": {
      "median_energy_error_per_edge": 0.002468109130859375,
      "median_marginal_mae": 0.0025990804424509406,
      "median_zz_mae": 0.007824367843568325
    }
  },
  "overall": {
    "median_energy_error_per_edge": 0.0029621830908581614,
    "median_exact_tvd_18q": 0.018082817747726382,
    "median_marginal_mae": 0.002687020692974329,
    "median_zz_mae": 0.007824367843568325
  }
}
```

## Systems

- Worst warm 4,096-shot median latency: 23.786 ms.
- Maximum incremental peak RSS: 14.39 MiB.
- Per-circuit optimizer steps and neural-route exact calls remained zero.

## Boundary

Exact statevector teacher/validation completed through 24Q. M2 remains the classical boundary reference: statevector was resource-guarded at 26Q; MPS cycle reached 32Q, random 3-regular p=1 reached 28Q, and random 3-regular p=2/3 hit the 120-second guard at 20Q.

## Interpretation

A failed model gate means iterate inside M3; it does not justify QPU data. A pass means only that the exact-regime model is ready for the next simulator-side gate.
