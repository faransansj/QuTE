# QuTE Research Proposal Gate

**Proposal / owner / date:**

Answer before execution:
1. Which QuTE product/runtime capability does this enable?
2. Which restricted workload is targeted?
3. What quantum-compatible input/output contract improves?
4. How does it reduce simulator/QPU calls, latency, or memory?
5. Does it avoid full state/full unitary output at scale?
6. Which physical structure is represented by construction?
7. What OOD/failure detector and fallback path exist?
8. Which task-level and systems metrics will be measured?
9. Which new untouched validation/test sets will be used?
10. Why is this Core, Runtime, Hardware Twin, Feynman, augmentation, or Benchmark work?

Choose **exactly one**: `ALIGNED_CORE` · `ALIGNED_RUNTIME` · `ALIGNED_HARDWARE_TWIN` · `ALIGNED_BENCHMARK` · `ADJACENT_SEPARATE_PROJECT` · `REJECTED_DIRECTION_DRIFT`

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

`REJECTED_DIRECTION_DRIFT` work must not start under QuTE.
