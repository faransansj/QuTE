# M3 Exact-Regime Scale Gate Protocol v12

## Reason for v12

V11's contextual decoder plateaued at robustness p90 ZZ MAE `0.05345`, with all other checks passing. V12 adds one direct learned prefix-pair term to the contextual logit. This is the smallest architectural change that exposes individual earlier-node interactions instead of only pooled prefix and neighbor summaries. Compatible context/encoder weights initialize from v11; the pair MLP is new. Training uses the unchanged 12,000-circuit mixed development corpus for 20 epochs, learning rate 0.0005, and correlation-loss weight 30.

No data, split, threshold, or decision-rule changes. Every check remains mandatory. QPU work remains unauthorized.
