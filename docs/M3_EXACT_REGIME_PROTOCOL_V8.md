# M3 Exact-Regime Scale Gate Protocol v8

## Reason for v8

V7 fine-tuned only on the 2,000 hard-cell augmentation and worsened tail robustness through distribution shift. V8 keeps the v4 contextual checkpoint and the same augmentation but mixes its 2,000 circuits with all 10,000 original circuits during fine-tuning. The 12,000-circuit mixture uses 256 samples per circuit, 20 epochs, and learning rate 0.0007. No validation data, threshold, or split changes.

All main, nonzero-angle median/p90, systems, route, scale, and exact-boundary gates remain identical. Every check must pass for `M3_PASS_EXACT_REGIME`; otherwise `M3_NEEDS_ITERATION`. QPU work remains unauthorized.
