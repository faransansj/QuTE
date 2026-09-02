# M3 Exact-Regime Scale Gate Protocol v11

## Reason for v11

V10 passed every check except nonzero-angle p90 ZZ MAE, now `0.05375` against `0.05`. V11 performs one bounded continuation from v10: 10 epochs on the unchanged 12,000-circuit mixed development corpus, correlation-loss weight 60, and learning rate 0.0002. All data, architecture, thresholds, and decision rules remain fixed.

Every check remains mandatory. QPU work remains unauthorized.
