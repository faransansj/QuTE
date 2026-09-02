# M2 Next-Stage Recommendation

## Decision

`PROCEED_TO_M3` means there is evidence to train a small scale-capable QuTE model. It does not mean QuTE has achieved speedup.

## Exact M3 scope

```text
train widths:       {20, 22, 24}
validation widths:  {18, 20, 22, 24}; hold out graphs and parameter points
QAOA p:             {1, 2, 3}
graph families:     random 3-regular primary; cycle control
shots:              {1024, 4096, 65536}; optimize first for 4096
model output:        direct samples/counts; no explicit 2^n table
hardware:            simulator/CPU-GPU development only; no QPU jobs
```

Do not add Erdős–Rényi to the first M3 corpus. Random 3-regular already made MPS noncompetitive at `p=2/3`; another family is not needed yet.

## Engineering targets

- Warm 4,096-shot end-to-end latency: **<=100 ms per circuit**, batch size 1.
- Warm throughput: measure batches 1, 8, and 32; no minimum is claimed until the architecture exists.
- Peak process memory: **<=256 MiB incremental/model-attributable budget**, with process baseline reported separately.
- Model load: report separately; do not include it in repeated-evaluation latency.
- Per-circuit optimizer steps: **0**.
- Neural-route exact simulator calls: **0**.
- No explicit `2^n` inference output.

The 100 ms target is stricter than the 2× requirement for every primary measured candidate: random 3-regular 22Q baseline medians were 581.9–968.5 ms, and 24Q medians were 2,314.3–5,313.0 ms.

## Accuracy targets for preregistration

Freeze these before generating any confirmatory data:

- median cut-energy absolute error per edge: **<=0.05**;
- marginal-probability mean absolute error: **<=0.03**;
- ZZ-correlation mean absolute error: **<=0.05**;
- exact-distribution TVD: report only where a full exact distribution and statistically adequate sample count are tractable; exploratory target **<=0.30**;
- report optimal-cut probability error, Hellinger fidelity, and calibration/OOD behavior;
- require graph-seed and parameter-point holdouts at each validation width.

These are proposed M3 engineering gates, not reinterpretations of the frozen 6Q pilot thresholds.

## Teacher and training-data budget

Start with **10,000 teacher circuits total**, stratified across 20/22/24Q, p=1/2/3, and the two graph roles. Reserve at least 1,000 additional circuits for development validation with disjoint graph seeds. On the measured host, a 10,000-circuit random-3-regular corpus distributed uniformly across the nine width/depth cells is approximately **4.2 serial hours**, **0.52 idealized hours at 8 workers**, and **about 430 MiB** under the sparse-count upper-bound model.

Do not authorize 100,000 or 1,000,000 circuits until the 10,000-circuit study meets accuracy, latency, and memory gates. `teacher_cost_projection.json` contains the larger planning scenarios.

## Model work allowed

Replace the frozen per-shot Python feature-stack loop with a scale-capable batched representation. Preserve direct sample generation, zero per-circuit optimization, and zero exact-route calls. Do not build production `BackendV2`, fallback routing, or QPU integration in M3.

## QPU decision

**Do not use a QPU in M3.** QPU work becomes justifiable only after the exact-regime amortized model meets accuracy and systems gates, shows a width/depth scaling trend, reaches a documented simulator/MPS validation boundary, and leaves a specific gap that QPU data can resolve.
