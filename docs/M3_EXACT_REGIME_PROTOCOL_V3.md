# M3 Exact-Regime Scale Gate Protocol v3

## Reason for v3

V2 reduced nonzero-angle error but failed the frozen robustness ZZ gates (`median=0.0579`, `p90=0.1448`, target `<=0.05`). V2 artifacts remain immutable. V3 changes only model capacity and training duration; every v2 workload, split, teacher artifact, threshold, and decision rule is retained.

## Model iteration

V3 uses a graph-conditioned contextual autoregressive decoder. For each target qubit it combines the target graph embedding, the signed prefix embedding, and the signed already-generated neighbor embedding through a nonlinear decoder. Teacher forcing computes every token in parallel; inference retains one loop over at most 24 qubits and vectorizes 4,096 shots. Hidden width is 64, message-passing depth is 3, and full training is 30 epochs.

## Frozen data and gates

- Full teacher: 10,000 exact-regime circuits, n={20,22,24}, p={1,2,3}, random 3-regular plus cycle.
- Main validation: disjoint seeds 91000–91002, n={18,20,22,24}.
- Robustness validation: disjoint seeds {92000,92001}, nonzero Sobol indices {3,5,7,11}.
- Accuracy limits: energy error/edge <=0.05, marginal MAE <=0.03, ZZ MAE <=0.05, 18Q TVD <=0.30.
- Robustness requires both median and p90 to meet the same energy, marginal, and ZZ limits.
- Systems: worst 4,096-shot median <=100 ms; incremental peak <=256 MiB; 24Q/20Q latency <=1.5.
- Scale: 24Q energy median <=2x 18Q median; exact validation through 24Q.
- Route: zero optimizer steps, zero exact calls, no explicit 2^n inference output.

## Decisions

`M3_PASS_EXACT_REGIME` requires every main, robustness, systems, route, and scale check. Any failed check is `M3_NEEDS_ITERATION`; unreliable infrastructure is `M3_BLOCKED`. QPU work remains unauthorized regardless of outcome.
