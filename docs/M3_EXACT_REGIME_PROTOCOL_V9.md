# M3 Exact-Regime Scale Gate Protocol v9

## Reason for v9

V8 preserved overall accuracy but still missed only the nonzero-angle p90 ZZ gate. NLL gives each of 20–24 bits equal weight, so edge-correlation tails contribute too little to optimization. V9 keeps the v4 contextual checkpoint, frozen 12,000-circuit mixed development corpus, architecture, splits, and every threshold. It adds a teacher-forced edge-correlation auxiliary loss with frozen weight 10. For each edge, the loss compares the teacher ZZ correlation with the conditional model correlation estimated from the earlier teacher bit and later-bit conditional mean. No validation examples or metrics enter training.

Every main, robustness median/p90, systems, route, scale, and boundary check remains unchanged. QPU work remains unauthorized.
