# M3 Exact-Regime Scale Gate Protocol v5

## Reason for v5

V4 passed every gate except nonzero-angle robustness p90 ZZ MAE (`0.06349`, target `<=0.05`). The remaining failures concentrate in random 3-regular p=2/3 cells, where QAOA correlations propagate beyond direct neighbors. V4 artifacts remain immutable.

V5 retains every frozen data split and threshold. Its decoder receives separate signed prefix summaries for graph-distance shells 1, 2, and 3 in addition to the global prefix and target embedding. This matches the maximum QAOA depth without adding speculative generality. The graph encoder is initialized from v4 where tensor shapes match; the shell decoder is new. Full training uses all 256 retained teacher samples for 30 epochs.

## Gates

Unchanged from v4: main accuracy medians, nonzero-angle robustness medians and p90s, <=100 ms worst 4,096-shot latency, <=256 MiB incremental peak, width trends, exact validation through 24Q, zero optimizer steps/exact calls, and no explicit 2^n output must all pass.

Outcomes remain `M3_PASS_EXACT_REGIME`, `M3_NEEDS_ITERATION`, or `M3_BLOCKED`. QPU work remains unauthorized.
