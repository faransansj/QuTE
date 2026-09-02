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
      "median_energy_error_per_edge": 0.002476444933563471,
      "median_marginal_mae": 0.003334469278343022,
      "median_zz_mae": 0.007089120568707585
    },
    "20": {
      "median_energy_error_per_edge": 0.002079264260828495,
      "median_marginal_mae": 0.0031921386253088713,
      "median_zz_mae": 0.006676737451925874
    },
    "22": {
      "median_energy_error_per_edge": 0.0026060162344947457,
      "median_marginal_mae": 0.0032750910613685846,
      "median_zz_mae": 0.007636330323293805
    },
    "24": {
      "median_energy_error_per_edge": 0.0021962058963254094,
      "median_marginal_mae": 0.0032033920288085938,
      "median_zz_mae": 0.0064684548415243626
    }
  },
  "overall": {
    "median_energy_error_per_edge": 0.002399472869001329,
    "median_exact_tvd_18q": 0.02972188114843574,
    "median_marginal_mae": 0.003252237569540739,
    "median_zz_mae": 0.006823984673246741
  }
}
```

## Systems

- Worst warm 4,096-shot median latency: 24.640 ms.
- Maximum incremental peak RSS: 9.39 MiB.
- Per-circuit optimizer steps and neural-route exact calls remained zero.

## Boundary

Exact statevector teacher/validation completed through 24Q. M2 remains the classical boundary reference: statevector was resource-guarded at 26Q; MPS cycle reached 32Q, random 3-regular p=1 reached 28Q, and random 3-regular p=2/3 hit the 120-second guard at 20Q.

## Interpretation

A failed model gate means iterate inside M3; it does not justify QPU data. A pass means only that the exact-regime model is ready for the next simulator-side gate.
