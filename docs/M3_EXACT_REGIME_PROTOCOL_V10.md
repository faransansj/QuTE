# M3 Exact-Regime Scale Gate Protocol v10

## Reason for v10

V9's edge-correlation loss improved robustness p90 ZZ MAE from `0.06860` to `0.06133` while retaining all other passes. V10 continues from the frozen v9 checkpoint on the same 12,000-circuit mixed development corpus, raises the correlation-loss weight from 10 to 30, and lowers learning rate to 0.0004 for 20 epochs. No data, split, architecture, threshold, or gate changes.

All checks remain mandatory. QPU work remains unauthorized.
