# M3 Exact-Regime Scale Gate Report

**Decision:** `M3_PASS_EXACT_REGIME`
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

## Accuracy

```json
{
  "by_width": {
    "18": {
      "median_energy_error_per_edge": 0.004174691624939442,
      "median_marginal_mae": 0.003106011194176972,
      "median_zz_mae": 0.013268365059047937
    },
    "20": {
      "median_energy_error_per_edge": 0.00459314975887537,
      "median_marginal_mae": 0.0030769348377361894,
      "median_zz_mae": 0.01399993896484375
    },
    "22": {
      "median_energy_error_per_edge": 0.004756349604576826,
      "median_marginal_mae": 0.0032078135991469026,
      "median_zz_mae": 0.012750798836350441
    },
    "24": {
      "median_energy_error_per_edge": 0.005412631668150425,
      "median_marginal_mae": 0.003296534181572497,
      "median_zz_mae": 0.013139936607331038
    }
  },
  "overall": {
    "median_energy_error_per_edge": 0.00463104248046875,
    "median_exact_tvd_18q": 0.07345481731230674,
    "median_marginal_mae": 0.0031581531511619687,
    "median_zz_mae": 0.013268365059047937
  }
}
```

## Systems

- Worst warm 4,096-shot median latency: 9.539 ms.
- Maximum incremental peak RSS: 4.03 MiB.
- Per-circuit optimizer steps and neural-route exact calls remained zero.

## Boundary

Exact statevector teacher/validation completed through 24Q. M2 remains the classical boundary reference: statevector was resource-guarded at 26Q; MPS cycle reached 32Q, random 3-regular p=1 reached 28Q, and random 3-regular p=2/3 hit the 120-second guard at 20Q.

## Interpretation

A failed model gate means iterate inside M3; it does not justify QPU data. A pass means only that the exact-regime model is ready for the next simulator-side gate.
