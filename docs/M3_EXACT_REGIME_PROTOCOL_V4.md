# M3 Exact-Regime Scale Gate Protocol v4

## Reason for v4

V3 passed every main accuracy, latency, route, and scale check, but failed the frozen nonzero-angle robustness gates and exceeded the memory budget because sampling rebuilt all prefix embeddings at every token. V3 artifacts remain immutable.

V4 keeps the same contextual architecture and frozen data. It (1) replaces repeated prefix reconstruction with an incremental prefix state, (2) warm-starts from the v3 full checkpoint, and (3) continues full-corpus training for 40 epochs using all 256 retained teacher samples per circuit at learning rate 0.001. Thresholds are unchanged.

## Frozen gates

- Main median energy error/edge <=0.05, marginal MAE <=0.03, ZZ MAE <=0.05, 18Q TVD <=0.30.
- Nonzero-angle robustness median and p90 must each meet the same energy, marginal, and ZZ limits.
- Worst warm 4,096-shot median <=100 ms; maximum incremental peak RSS <=256 MiB; 24Q/20Q latency <=1.5.
- 24Q energy median <=2x 18Q median; exact validation through 24Q.
- Zero per-circuit optimizer steps, zero neural-route exact calls, and no explicit 2^n inference output.

`M3_PASS_EXACT_REGIME` requires every check. Otherwise the result is `M3_NEEDS_ITERATION`; unreliable infrastructure is `M3_BLOCKED`. QPU use remains unauthorized.
