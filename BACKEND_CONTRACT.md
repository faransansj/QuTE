# QuTE Backend Contract

**Contract:** `qute_backend_contract_v1`
**Qiskit reference checked:** 2.5 `BackendV2` API and custom-backend guide
**Pilot status:** backend-style synchronous adapter; full `BackendV2` packaging deferred

## 1. User-facing contract

```python
backend = QuTEBackend.from_pretrained("qute-qaoa-r1")
job = backend.run(
    circuits,
    shots=4096,
    seed_simulator=2026,
    fallback="aer",
)
result = job.result()
counts = result.get_counts()
```

`run` accepts one circuit or a list and returns a job object. Counts follow Qiskit bitstring order and sum to `shots` for every experiment.

The pilot accepts a compact `QAOACircuit` record and optionally a Qiskit `QuantumCircuit` carrying canonical `metadata["qute_qaoa"]`. Full semantic extraction from arbitrary transpiled circuits is not silently attempted.

## 2. Supported R1 input

All conditions must hold:

- ideal noiseless QAOA MaxCut;
- declared supported qubit and `p` range;
- simple undirected graph inside the frozen graph family/degree envelope;
- `|0>^n` followed by in-circuit state preparation;
- canonical QAOA cost and mixer parameterization;
- terminal computational-basis measurement only;
- requested shots and seed within implementation limits;
- model version has a matching support-envelope hash.

## 3. Unsupported input

R1 rejects or routes:

- arbitrary state initialization;
- gates or circuit motifs outside the canonical QAOA contract;
- mid-circuit measurement;
- reset;
- classical control flow or dynamic circuits;
- nonterminal measurements;
- noise models or hardware-calibration requests;
- unsupported width, depth, graph family, degree band, or parameter cell;
- malformed or absent semantic metadata when safe parsing is impossible.

Unsupported inputs never receive a silent neural answer.

## 4. Routing policy

`fallback` has three allowed values:

- `"reject"`: raise `QuTEUnsupportedCircuitError`;
- `"warn"`: require an explicit `allow_unsupported=True`, mark route `qute_unvalidated`, and emit a warning;
- `"aer"`: route to the version-frozen Aer backend and preserve both support and fallback metadata.

Default for production packaging: `"aer"`. Default for scientific pilot: `"reject"`, so unsupported execution cannot contaminate learned-route metrics.

The router runs before model inference. If estimated error exceeds the frozen threshold, the same policy applies even when syntax is supported.

## 5. Result schema

Each experiment result contains standard counts plus:

```json
{
  "qute": {
    "schema_version": "qute-result-v1",
    "model_name": "qute-qaoa-r1",
    "model_version": "...",
    "model_sha256": "...",
    "support_envelope_sha256": "...",
    "circuit_sha256": "...",
    "graph_sha256": "...",
    "num_qubits": 6,
    "qaoa_p": 1,
    "gate_set": ["h", "rzz", "rx", "measure"],
    "support_score": 0.0,
    "estimated_error": 0.0,
    "estimated_error_metric": "tvd",
    "calibration_version": "...",
    "execution_route": "qute",
    "abstained": false,
    "fallback_backend": null,
    "seed_simulator": 2026,
    "shots": 4096,
    "timing_ms": {
      "parse": 0.0,
      "support": 0.0,
      "encode": 0.0,
      "inference_and_sampling": 0.0,
      "packaging": 0.0,
      "neural_total": 0.0,
      "fallback": 0.0,
      "total": 0.0
    },
    "per_circuit_optimizer_steps": 0,
    "exact_simulator_calls_on_neural_route": 0
  }
}
```

Pilot fields that lack calibration are present with `null` and `calibration_status: "UNAVAILABLE_PILOT"`; they are not fabricated.

## 6. Samples and observables

The internal sampler generates bitstrings ancestrally from

```text
p_theta(x | C) = product_i p_theta(x_i | x_<i, C)
```

without materializing a `2^n` table. Counts are aggregated from samples. Any diagonal observable is computed from the same samples and carries shot uncertainty. R1 includes MaxCut energy, optimum probability, one-qubit `Z`, and pairwise `ZZ` utilities.

Amplitude queries, non-diagonal observables, and statevector access are absent from R1.

## 7. Determinism

For a fixed model hash, canonical circuit hash, shot count, and seed, the CPU reference route returns identical samples. Accelerator determinism is reported and tested for the frozen hardware; if unavailable, reproducibility is statistical and explicitly labeled.

## 8. Qiskit integration requirements

The packaged backend must follow the installed Qiskit API, currently `BackendV2` in Qiskit 2.5:

- expose a `Target`, `num_qubits`, `max_circuits`, and runtime `Options`;
- implement `run(run_input, **options)` returning a Qiskit `Job`;
- return standard Qiskit result/count structures;
- declare supported basis gates and connectivity;
- avoid relying on non-abstract options such as per-shot `memory` unless explicitly implemented;
- add provider/SamplerV2 integration only after the core backend contract passes.

The wrapper is integration work, not the research contribution.

## 9. Safety invariants

- no per-circuit fitting, gradient computation, or optimizer state in `run`;
- no exact simulator import or call on the neural route;
- no statevector/probability-table allocation proportional to `2^n` on the neural route;
- all unsupported or uncertain cases are visible in metadata;
- fallback latency and cost are included in end-to-end metrics;
- model and support-envelope hashes are mandatory for persisted results.

## 10. Versioning

A change to circuit semantics, bit order, support envelope, calibration, model architecture, result schema, or fallback policy increments the corresponding version and invalidates cached certification. Historical results retain their original hashes and interpretation.
