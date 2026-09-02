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
      "median_energy_error_per_edge": 0.002559803193435073,
      "median_marginal_mae": 0.0026329888496547937,
      "median_zz_mae": 0.00588424107991159
    },
    "20": {
      "median_energy_error_per_edge": 0.0021113078109920025,
      "median_marginal_mae": 0.002479934715665877,
      "median_zz_mae": 0.005933634238317609
    },
    "22": {
      "median_energy_error_per_edge": 0.0024964303011074662,
      "median_marginal_mae": 0.0025607022689655423,
      "median_zz_mae": 0.006291707279160619
    },
    "24": {
      "median_energy_error_per_edge": 0.0017411973676644266,
      "median_marginal_mae": 0.0025634765625,
      "median_zz_mae": 0.005611419677734375
    }
  },
  "overall": {
    "median_energy_error_per_edge": 0.0022879706230014563,
    "median_exact_tvd_18q": 0.016110357070768775,
    "median_marginal_mae": 0.0025586446281522512,
    "median_zz_mae": 0.00596053502522409
  }
}
```

## Systems

- Worst warm 4,096-shot median latency: 23.869 ms.
- Maximum incremental peak RSS: 11.23 MiB.
- Per-circuit optimizer steps and neural-route exact calls remained zero.

## Boundary

Exact statevector teacher/validation completed through 24Q. M2 remains the classical boundary reference: statevector was resource-guarded at 26Q; MPS cycle reached 32Q, random 3-regular p=1 reached 28Q, and random 3-regular p=2/3 hit the 120-second guard at 20Q.

## Interpretation

A failed model gate means iterate inside M3; it does not justify QPU data. A pass means only that the exact-regime model is ready for the next simulator-side gate.
