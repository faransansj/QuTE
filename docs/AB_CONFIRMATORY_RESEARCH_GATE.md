# A/B Multi-Seed Confirmatory Gate

**Proposal / owner / date:** Minimal P4.6 A/B confirmation / QuTE / 2026-08-24

1. **Capability:** reduce selection risk before C1 is used in R1/M1.
2. **Restricted workload:** frozen four-qubit circuit distribution from P4.6.
3. **Contract:** no new runtime contract; validates the existing circuit/state-to-action research contract.
4. **Value:** avoids carrying an unstable data allocation or operator baseline into later simulator/QPU-call reduction work.
5. **Scale boundary:** four qubits only; no scale-out or production claim.
6. **Physical structure:** B3 exact-unitary Cayley output; A track uses the frozen direct-state comparator.
7. **OOD/fallback:** existing IID, State-, Parameter-, Composition-, and Depth-OOD validation splits; no deployment decision or fallback claim.
8. **Metrics:** balanced validation fidelity, paired per-seed deltas, training throughput, and runtime.
9. **Data:** existing frozen validation splits only. No sealed test is opened; no final-test claim is permitted.
10. **Class:** benchmark confirmation supporting future Core/Runtime work.

**Decision:** `ALIGNED_BENCHMARK`

## Frozen scope

- Reuse seed 2026 canonical results.
- Train seeds 2027 and 2028 only.
- Track A: A3 versus A4.
- Track B: B0, B1, and B3.
- Select best balanced-validation checkpoint exactly as in P4.6.
- No hyperparameter changes, post-result tuning, formal significance claim, or sealed-test access.

## Automatic drift warnings

- [ ] Larger model without workload/generalization hypothesis
- [ ] Arbitrary full-state/full-unitary scale-out
- [ ] Test-set-driven architecture change
- [ ] Neural-only deployment without fallback
- [ ] Fidelity-only study without systems metric
- [ ] LLM/language feature presented as execution core
- [ ] QPU replacement claim without domain/coverage limits
- [ ] Simulator implementation without learned-runtime purpose
- [ ] Primary output unrelated to quantum execution
