# M3 Exact-Regime Scale Gate Protocol v6

## Reason for v6

V5's distance-shell decoder still failed the unchanged nonzero-angle p90 ZZ gate (`0.06808`). The remaining tail indicates that fixed summary statistics are insufficient for higher-order prefix dependence. V1–v5 artifacts remain immutable.

V6 uses a graph-conditioned GRU autoregressive decoder. The hidden state carries the complete generated prefix while each step receives the current graph embedding and previous sampled bit. This adds no width-specific output and keeps the inference loop bounded at 24 positions. Compatible graph-encoder weights initialize from v4; the GRU decoder is new. Training uses the same frozen 10,000 circuits, 128 retained samples per circuit, batch 16, and 30 epochs.

## Gates

No threshold or dataset changes: all main medians, nonzero-angle robustness medians and p90s, <=100 ms latency, <=256 MiB incremental peak, width trends, exact 24Q boundary, zero optimizer/exact calls, and no explicit 2^n output must pass.

QPU work remains unauthorized. Decisions remain `M3_PASS_EXACT_REGIME`, `M3_NEEDS_ITERATION`, or `M3_BLOCKED`.
