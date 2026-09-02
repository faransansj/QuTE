# M3 Exact-Regime Scale Gate Report

**Decision:** `M3_PASS_EXACT_REGIME`
**Training stage reached:** `full`
**Role:** exploratory, simulator-only; no QPU authorization

## Gate checks

| Gate | Result |
|---|---|
| `energy_error_per_edge` | PASS |
| `energy_width_trend` | PASS |
| `exact_calls_zero` | PASS |
| `full_10000_circuit_corpus` | PASS |
| `latency` | PASS |
| `latency_width_trend` | PASS |
| `marginal_mae` | PASS |
| `memory` | PASS |
| `optimizer_steps_zero` | PASS |
| `robust_median_energy_error_per_edge` | PASS |
| `robust_median_marginal_mae` | PASS |
| `robust_median_zz_mae` | PASS |
| `robust_p90_energy_error_per_edge` | PASS |
| `robust_p90_marginal_mae` | PASS |
| `robust_p90_zz_mae` | PASS |
| `tvd_18q` | PASS |
| `validation_boundary_24q` | PASS |
| `zz_mae` | PASS |

## Accuracy

```json
{
  "by_width": {
    "18": {
      "median_energy_error_per_edge": 0.0013580322265625,
      "median_marginal_mae": 0.0031424628105014563,
      "median_zz_mae": 0.005391438724473119
    },
    "20": {
      "median_energy_error_per_edge": 0.0012733459589071572,
      "median_marginal_mae": 0.0030574798583984375,
      "median_zz_mae": 0.005403137067332864
    },
    "22": {
      "median_energy_error_per_edge": 0.002022483153268695,
      "median_marginal_mae": 0.0030919855926185846,
      "median_zz_mae": 0.006370313232764602
    },
    "24": {
      "median_energy_error_per_edge": 0.0017490386962890625,
      "median_marginal_mae": 0.0028527576941996813,
      "median_zz_mae": 0.005341423908248544
    }
  },
  "overall": {
    "median_energy_error_per_edge": 0.0015089247026480734,
    "median_exact_tvd_18q": 0.022561834443535253,
    "median_marginal_mae": 0.0030641555786132812,
    "median_zz_mae": 0.005556177347898483
  }
}
```

## Confirmatory robustness

```json
{
  "by_width": {
    "18": {
      "median_energy_error_per_edge": 0.0042444863356649876,
      "median_marginal_mae": 0.0038214789237827063,
      "median_zz_mae": 0.013178507331758738
    },
    "20": {
      "median_energy_error_per_edge": 0.0040646870620548725,
      "median_marginal_mae": 0.0039272308349609375,
      "median_zz_mae": 0.011727396864444017
    },
    "22": {
      "median_energy_error_per_edge": 0.0045575229451060295,
      "median_marginal_mae": 0.0037089261459186673,
      "median_zz_mae": 0.011483394540846348
    },
    "24": {
      "median_energy_error_per_edge": 0.0039037069072946906,
      "median_marginal_mae": 0.003673553466796875,
      "median_zz_mae": 0.012063768226653337
    }
  },
  "overall": {
    "maximum_energy_error_per_edge": 0.027685094624757767,
    "maximum_marginal_mae": 0.024993896484375,
    "maximum_zz_mae": 0.07870596647262573,
    "median_energy_error_per_edge": 0.004246266558766365,
    "median_exact_tvd_18q": null,
    "median_marginal_mae": 0.003769075032323599,
    "median_zz_mae": 0.01196153461933136,
    "p90_energy_error_per_edge": 0.010936962999403478,
    "p90_marginal_mae": 0.011049970239400863,
    "p90_zz_mae": 0.03944939486682416
  }
}
```

## Systems

- Worst warm 4,096-shot median latency: 24.063 ms.
- Maximum incremental peak RSS: 9.22 MiB.
- Per-circuit optimizer steps and neural-route exact calls remained zero.

## Boundary

Exact statevector teacher/validation completed through 24Q. M2 remains the classical boundary reference: statevector was resource-guarded at 26Q; MPS cycle reached 32Q, random 3-regular p=1 reached 28Q, and random 3-regular p=2/3 hit the 120-second guard at 20Q.

## Interpretation

A failed model gate means iterate inside M3; it does not justify QPU data. A pass means only that the exact-regime model is ready for the next simulator-side gate.
