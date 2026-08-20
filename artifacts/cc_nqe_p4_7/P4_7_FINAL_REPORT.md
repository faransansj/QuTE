# CC-NQE P4.7 Final Report

## 1. Objective and frozen hypotheses
Test recurrent encoding, composition consistency, and privileged prefix supervision without sealed-test access.

## 2. P4.6 C0 anchor
C0 is the frozen P4.6 B3 monolithic action-only exact-unitary Cayley anchor.

## 3. P4.7 architecture variants
C1 is shared causal recurrence; C2 adds action-only composition self-consistency (no exact-U target); C3 adds privileged exact prefix-action targets and is not supervision-matched.

## 4. Screening seed-2026 results
Seed 2026 was screening evidence only; C0 came from P4.6 and C1–C3 from P4.7.

## 5. Confirmatory protocol
Seeds 2027/2028 used 10,000 updates and 10,240,000 exposures. Every comparison uses one frozen best-balanced-validation checkpoint, where S_balanced=(IID+Composition-OOD+Depth-OOD)/3. Per-metric maxima are excluded from primary comparisons.

## 6. Twelve-run integrity audit
PASS: 12/12 completed scientific cells; expected identities, supervision, dataset/config equivalence, checkpoints, finite metrics, exact unitarity (<1e-5), and zero sealed access verified.

## 7. Per-seed metric table
|Variant|Seed|Step|S_balanced|IID/action|State|Parameter|Composition|Depth|Process|Phase error|Unitarity|max raw norm|Runtime s|Samples/s|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|C0|2026|8500|0.590436|0.745513|0.651013|0.629877|0.494417|0.526930|0.623127|0.548612|2.701e-07|1.000000|326.628180|31350.632360|
|C0|2027|8500|0.542873|0.693059|0.607105|0.684706|0.418516|0.517044|0.567779|0.616826|2.693e-07|1.000000|328.965940|31127.842573|
|C0|2028|7000|0.534027|0.627145|0.570909|0.735261|0.384078|0.590858|0.564510|0.623870|2.752e-07|1.000000|325.813733|31429.000561|
|C1|2026|6000|0.609183|0.818279|0.724461|0.642386|0.481953|0.527317|0.683418|0.521548|2.423e-07|1.000000|720.229931|14217.681823|
|C1|2027|8000|0.638943|0.835206|0.747230|0.695181|0.488059|0.593564|0.698891|0.500957|2.613e-07|1.000000|701.996825|14586.960564|
|C1|2028|3000|0.607069|0.810717|0.722791|0.666932|0.463511|0.546979|0.670959|0.544010|2.395e-07|1.000000|700.622167|14615.580946|
|C2|2026|10000|0.607312|0.736452|0.640878|0.587508|0.485559|0.599925|0.610924|0.594298|2.446e-07|1.000000|2396.809295|4272.346582|
|C2|2027|9000|0.612535|0.758444|0.651910|0.627336|0.500569|0.578593|0.618471|0.588713|2.497e-07|1.000000|2319.412056|4414.911949|
|C2|2028|7000|0.597152|0.733561|0.634482|0.571772|0.488051|0.569844|0.605881|0.602834|2.649e-07|1.000000|2316.489540|4420.481863|
|C3|2026|8000|0.597892|0.788250|0.727650|0.682759|0.472691|0.532736|0.681210|0.510138|2.540e-07|1.000000|1579.938158|6481.266340|
|C3|2027|7500|0.622957|0.813818|0.744265|0.574223|0.469191|0.585862|0.700241|0.490683|2.405e-07|1.000000|1529.090694|6696.790478|
|C3|2028|9500|0.599745|0.778941|0.701306|0.662772|0.489927|0.530368|0.669841|0.522499|2.567e-07|1.000000|1526.607241|6707.684679|

## 8. Three-seed mean/std table
Values are mean ± sample SD; this is descriptive three-seed evidence, not large-sample statistical proof. Full individual/min/max values are in `confirmatory/summary.json`.

|Variant|S_balanced|IID|State|Parameter|Composition|Depth|Process fidelity|Runtime s|
|---|---|---|---|---|---|---|---|---|
|C0|0.555779 ± 0.030338|0.688572 ± 0.059311|0.609675 ± 0.040114|0.683281 ± 0.052707|0.432337 ± 0.056453|0.544944 ± 0.040069|0.585139 ± 0.032940|327.135951 ± 1.636299|
|C1|0.618398 ± 0.017824|0.821400 ± 0.012539|0.731494 ± 0.013653|0.668166 ± 0.026419|0.477841 ± 0.012780|0.555953 ± 0.034023|0.684423 ± 0.013993|707.616308 ± 10.945321|
|C2|0.605666 ± 0.007823|0.742819 ± 0.013608|0.642423 ± 0.008816|0.595539 ± 0.028640|0.491393 ± 0.008044|0.582787 ± 0.015473|0.611759 ± 0.006336|2344.236964 ± 45.552418|
|C3|0.606865 ± 0.013967|0.793670 ± 0.018059|0.724407 ± 0.021662|0.639918 ± 0.057765|0.477270 ± 0.011101|0.549655 ± 0.031379|0.683764 ± 0.015360|1545.212031 ± 30.099332|

## 9. C1-minus-C0 analysis
|Metric|2026|2027|2028|Mean|Sample SD|Signs|
|---|---:|---:|---:|---:|---:|---|
|S_balanced|0.018747|0.096070|0.073042|0.062620|0.039701|all_positive|
|IID|0.072766|0.142147|0.183572|0.132828|0.055988|all_positive|
|Composition_OOD|-0.012464|0.069543|0.079433|0.045504|0.050445|mixed|
|Depth_OOD|0.000387|0.076520|-0.043878|0.011010|0.060898|mixed|
|State_OOD|0.073448|0.140125|0.151882|0.121818|0.042300|all_positive|
|Parameter_OOD|0.012509|0.010475|-0.068330|-0.015115|0.046096|mixed|

The balanced effect is sign-consistent across all seeds; target OOD axes are mixed.

## 10. C2-minus-C1 analysis
|Metric|2026|2027|2028|Mean|Sample SD|Signs|
|---|---:|---:|---:|---:|---:|---|
|S_balanced|-0.001871|-0.026408|-0.009917|-0.012732|0.012508|all_negative|
|Composition_OOD|0.003605|0.012510|0.024540|0.013552|0.010506|all_positive|
|Depth_OOD|0.072609|-0.014971|0.022864|0.026834|0.043925|mixed|
|IID|-0.081827|-0.076762|-0.077156|-0.078582|0.002817|all_negative|

Composition-OOD improves in every seed while S_balanced falls in every seed.

## 11. C3-minus-C1 analysis
|Metric|2026|2027|2028|Mean|Sample SD|Signs|
|---|---:|---:|---:|---:|---:|---|
|S_balanced|-0.011290|-0.015986|-0.007324|-0.011533|0.004336|all_negative|
|Composition_OOD|-0.009262|-0.018868|0.026416|-0.000571|0.023860|mixed|
|Depth_OOD|0.005420|-0.007702|-0.016612|-0.006298|0.011083|mixed|
|IID|-0.030029|-0.021387|-0.031776|-0.027731|0.005563|all_negative|

C3's privileged supervision does not justify advancement: balanced effects are all negative and target-OOD effects are mixed.

## 12. Composition-OOD versus Depth-OOD interpretation
C2's Composition-OOD effect is sign-consistent. Its Depth-OOD signs are mixed; the large seed-2026 gain was not seed-robust. The supplied excerpt agrees with canonical values within rounding.

## 13. Mechanistic verdicts
- **RECURRENT-ENCODING-SUPPORTED**
- **COMPOSITION-SPECIFIC-GAIN-WITH-TRADEOFF**
- **PREFIX-SUPERVISION-NOT-SUPPORTED**

## 14. Overall P4.7 verdict
**P4.7-RECURSIVE-AND-COMPOSITION-SPECIFIC-BIAS-SUPPORTED**

## 15. Candidate selection
General-purpose: **C1**, mean S_balanced 0.618398, margin 0.012732; per-seed ranks {'2026': 1, '2027': 1, '2028': 1}. C2 is retained only as a composition specialist with its balanced trade-off explicit. C3 is archived as a negative privileged-supervision result.

## 16. Limitations
Three seeds, validation splits, four qubits, explicit U(16), and descriptive statistics only. No formal significance, arbitrary-circuit efficiency, large-qubit scaling, or universal OOD claim.

## 17. Sealed-test status
Not evaluated; access count **0**.

## 18. Recommended next phase
P4.8 is proposed but not executed: freeze C1 as primary, C2 as predeclared composition specialist, and C0 as anchor; prohibit tuning; open sealed Composition/Depth splits once and report without post-test selection.

## 19. Repository provenance
Implementation anchor `34263f65fc0fbd27c6049f5d2ec80c06019323a9`; canonical hashes are recorded in `artifact_hashes.json`. P4.6 historical artifacts were not rewritten.
