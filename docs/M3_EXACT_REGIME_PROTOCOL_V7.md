# M3 Exact-Regime Scale Gate Protocol v7

## Reason for v7

V4 is the strongest model so far but missed only the nonzero-angle p90 ZZ gate (`0.06349` vs `0.05`). V5 shell summaries and V6 GRU state did not improve the tail. The evidence points to insufficient teacher precision in hard random 3-regular p=2/3 cells, not missing generic architecture.

V7 therefore returns to the v4 contextual checkpoint and adds a preregistered 2,000-circuit development augmentation: random 3-regular only, n={20,22,24}, p={2,3}, graph seeds starting at 51000, and 1,024 exact teacher shots per circuit. These seeds are disjoint from main and robustness validation. Fine-tuning uses 512 retained samples for 30 epochs. The original 10,000-circuit corpus remains part of the trained checkpoint lineage.

## Gates

All v4 gates and the nonzero-angle robustness corpus remain unchanged. The augmentation does not include validation seeds or select examples from validation failures. Every main, robustness median/p90, latency, memory, width-trend, route, and boundary check must pass for `M3_PASS_EXACT_REGIME`.

QPU work remains unauthorized.
