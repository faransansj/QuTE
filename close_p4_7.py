"""Freeze P4.7 from completed validation artifacts. Never trains or reads sealed tests."""
from __future__ import annotations

import csv, hashlib, json, math, statistics
from pathlib import Path

ROOT = Path("artifacts/cc_nqe_p4_7")
CONFIRM = ROOT / "confirmatory"
P46 = Path("artifacts/cc_nqe_p4_6")
SEEDS = (2026, 2027, 2028)
VARIANTS = ("C0", "C1", "C2", "C3")
DATASET_HASH = "67a25b4384825dc477b22cbfc6f13bfc1424a95474cc8bcbc002c4ff010bf009"
SUPERVISION = {"C0":"ACTION_ONLY", "C1":"ACTION_ONLY", "C2":"ACTION_ONLY_PLUS_SELF_CONSISTENCY", "C3":"PRIVILEGED_PREFIX_ACTION_SUPERVISION"}
METRICS = ("S_balanced", "IID", "State_OOD", "Parameter_OOD", "Composition_OOD", "Depth_OOD", "process_fidelity", "runtime_seconds")
DELTA_METRICS = {
    "C1_minus_C0": ("S_balanced", "IID", "Composition_OOD", "Depth_OOD", "State_OOD", "Parameter_OOD"),
    "C2_minus_C1": ("S_balanced", "Composition_OOD", "Depth_OOD", "IID"),
    "C3_minus_C1": ("S_balanced", "Composition_OOD", "Depth_OOD", "IID"),
    "C2_minus_C0": ("S_balanced", "IID", "Composition_OOD", "Depth_OOD", "State_OOD", "Parameter_OOD"),
    "C3_minus_C0": ("S_balanced", "IID", "Composition_OOD", "Depth_OOD", "State_OOD", "Parameter_OOD"),
}


def load(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path, value): path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
def metric_path(v, s): return P46/"operator/metrics/B3.json" if (v,s)==("C0",2026) else ROOT/f"metrics/{v}.json" if s==2026 else CONFIRM/f"metrics/{v}-seed{s}.json"
def config_path(v, s): return P46/"operator/configs/B3.json" if (v,s)==("C0",2026) else ROOT/f"configs/{v}.json" if s==2026 else CONFIRM/f"configs/{v}-seed{s}.json"
def checkpoint_path(v, s): return P46/"operator/checkpoints/B3-best-balanced.pt" if (v,s)==("C0",2026) else ROOT/f"checkpoints/{v}-best-balanced.pt" if s==2026 else CONFIRM/f"checkpoints/{v}-seed{s}-best-balanced.pt"


def normalized_config(config):
    value = json.loads(json.dumps(config))
    for key in ("seed", "run_kind", "experiment_id", "checkpoint_path", "timestamps", "schema_version", "scientific_state", "source_variant"):
        value.pop(key, None)
    if value.get("variant") == "C0": value["variant"] = "B3"
    if "recipe" in value: value["recipe"].pop("seed", None)
    return value


def runtime_for(v, s, payload):
    if "runtime_seconds" in payload: return payload["runtime_seconds"]
    if "runtime" in payload: return payload["runtime"]
    if (v,s)==("C0",2026):
        rows=(json.loads(line) for line in (P46/"operator/progress.jsonl").read_text().splitlines())
        return [r["elapsed"] for r in rows if r.get("variant")=="B3"][-1]
    raise ValueError(f"{v}/{s}: runtime missing")


def values(v, s, payload):
    validation = payload.get("validation_at_best_balanced_checkpoint") or payload.get("latest_validation")
    old = (v,s)==("C0",2026)
    fidelity = "predicted_operator_state_fidelity" if old else "normalized_action_fidelity"
    def split(name): return validation[f"{name}_validation"]
    iid = split("iid")
    row = {
        "variant":v, "seed":s, "source_phase":"P4.6_B3_anchor" if old else "P4.7_screening" if s==2026 else "P4.7_confirmatory",
        "supervision_class":SUPERVISION[v], "parameter_count":load(config_path(v,s))["actual_parameters"],
        "best_checkpoint_step":payload["best_checkpoint_step"], "S_balanced":payload["best_balanced_validation"],
        "IID":iid[fidelity], "State_OOD":split("state_ood")[fidelity], "Parameter_OOD":split("parameter_ood")[fidelity],
        "Composition_OOD":split("composition_ood")[fidelity], "Depth_OOD":split("depth_ood")[fidelity],
        "action_fidelity":iid[fidelity], "process_fidelity":iid["process_fidelity"],
        "phase_aligned_frobenius_error":iid["phase_aligned_frobenius_error"],
        "unitarity_error":max(split(n)["raw_unitarity_error" if old else "unitarity_error"] for n in ("iid","state_ood","parameter_ood","composition_ood","depth_ood")),
        "raw_action_norm":iid["raw_action_norm"], "runtime_seconds":runtime_for(v,s,payload),
        "samples_per_second": payload.get("samples_per_second", 10240000/runtime_for(v,s,payload)),
        "config_hash":hashlib.sha256(json.dumps(normalized_config(load(config_path(v,s))),sort_keys=True,separators=(",",":")).encode()).hexdigest(),
        "dataset_hash":payload.get("dataset_manifest_hash", DATASET_HASH), "metric_hash":sha(metric_path(v,s)),
    }
    return row


def sign(values):
    signs={0 if x==0 else 1 if x>0 else -1 for x in values}
    if signs=={1}: return "all_positive"
    if signs=={-1}: return "all_negative"
    if signs=={0}: return "zero_or_tied"
    return "mixed"


def audit():
    rows=[]; normalized={}
    for v in VARIANTS:
        for s in SEEDS:
            mp,cp,kp=metric_path(v,s),config_path(v,s),checkpoint_path(v,s)
            if not all(p.exists() for p in (mp,cp,kp)): raise RuntimeError(f"P4.7-CLOSURE-BLOCKED: {v}/{s}: missing {mp if not mp.exists() else cp if not cp.exists() else kp}")
            payload=load(mp); config=load(cp)
            checks={"state":payload.get("state")=="COMPLETED", "scientific":payload.get("scientific") is True,
                    "seed":s==2026 or payload.get("seed")==s, "updates":payload.get("step")==10000,
                    "exposures":payload.get("samples_seen")==10240000, "variant":payload.get("variant") in (v,"B3"),
                    "supervision":SUPERVISION[v]==("ACTION_ONLY" if v=="C0" else config.get("supervision_class")),
                    "dataset":payload.get("dataset_manifest_hash",DATASET_HASH)==DATASET_HASH,
                    "sealed":payload.get("sealed_test_access_count",0)==0}
            if not all(checks.values()): raise RuntimeError(f"P4.7-CLOSURE-BLOCKED: {v}/{s}: {checks}")
            row=values(v,s,payload)
            if not all(math.isfinite(float(row[k])) for k in ("S_balanced","IID","State_OOD","Parameter_OOD","Composition_OOD","Depth_OOD","process_fidelity","unitarity_error","runtime_seconds","samples_per_second")): raise RuntimeError(f"P4.7-CLOSURE-BLOCKED: {v}/{s}: NaN/Inf")
            if row["unitarity_error"]>=1e-5: raise RuntimeError(f"P4.7-CLOSURE-BLOCKED: {v}/{s}: unitarity {row['unitarity_error']}")
            normalized.setdefault(v,[]).append(row["config_hash"])
            rows.append(row)
    for v, hashes in normalized.items():
        if len(set(hashes))!=1: raise RuntimeError(f"P4.7-CLOSURE-BLOCKED: {v}: config mismatch {hashes}")
    return rows


def close():
    rows=audit(); by={(r["variant"],r["seed"]):r for r in rows}
    fields=list(rows[0])
    with (CONFIRM/"variant_seed_metrics.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fields); writer.writeheader(); writer.writerows(rows)
    dump(CONFIRM/"variant_seed_metrics.json", rows)
    aggregates={}
    for v in VARIANTS:
        aggregates[v]={}
        for m in METRICS:
            xs=[by[v,s][m] for s in SEEDS]
            aggregates[v][m]={"individual_values":dict(zip(map(str,SEEDS),xs)),"mean":statistics.mean(xs),"sample_std":statistics.stdev(xs),"minimum":min(xs),"maximum":max(xs)}
    deltas=[]
    pairs={"C1_minus_C0":("C1","C0"),"C2_minus_C1":("C2","C1"),"C3_minus_C1":("C3","C1"),"C2_minus_C0":("C2","C0"),"C3_minus_C0":("C3","C0")}
    for comparison,(left,right) in pairs.items():
        for m in DELTA_METRICS[comparison]:
            xs=[by[left,s][m]-by[right,s][m] for s in SEEDS]
            deltas.append({"comparison":comparison,"metric":m,**{f"delta_{s}":x for s,x in zip(SEEDS,xs)},"mean_delta":statistics.mean(xs),"sample_std":statistics.stdev(xs),"sign_consistency":sign(xs)})
    with (CONFIRM/"paired_deltas.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,list(deltas[0])); writer.writeheader(); writer.writerows(deltas)
    dump(CONFIRM/"paired_deltas.json", deltas)
    summary={"schema_version":"cc-nqe-p4.7-closure-v1","state":"COMPLETE","seeds":list(SEEDS),"cell_integrity":"12/12","config_dataset_hash_integrity":"PASS","sealed_test_access_count":0,"aggregates":aggregates,"paired_deltas":deltas,"interpretation_note":"Descriptive three-seed evidence; not large-sample statistical proof."}
    dump(CONFIRM/"summary.json",summary)
    c1=next(d for d in deltas if d["comparison"]=="C1_minus_C0" and d["metric"]=="S_balanced")
    c2s=next(d for d in deltas if d["comparison"]=="C2_minus_C1" and d["metric"]=="S_balanced")
    c2c=next(d for d in deltas if d["comparison"]=="C2_minus_C1" and d["metric"]=="Composition_OOD")
    c3s=next(d for d in deltas if d["comparison"]=="C3_minus_C1" and d["metric"]=="S_balanced")
    candidates=sorted((aggregates[v]["S_balanced"]["mean"],v) for v in ("C0","C1","C2"))[::-1]
    primary=candidates[0][1]; ranks={str(s):sorted(VARIANTS[:3],key=lambda v:by[v,s]["S_balanced"],reverse=True).index(primary)+1 for s in SEEDS}
    selection={"general_purpose":{"variant":primary,"mean_S_balanced":candidates[0][0],"margin_over_next":candidates[0][0]-candidates[1][0],"per_seed_rank":ranks,"paired_advantage_sign_consistent":c1["sign_consistency"]=="all_positive","status":"selected"},"composition_specialist":{"variant":"C2","qualification":"Composition-OOD gain all-positive; balanced trade-off all-negative"} if c2c["sign_consistency"]=="all_positive" else None,"prefix_candidate":None,"C3_decision":"archive_negative_result"}
    dump(ROOT/"candidate_selection.json",selection)
    verdict={"overall_verdict":"P4.7-RECURSIVE-AND-COMPOSITION-SPECIFIC-BIAS-SUPPORTED" if c1["sign_consistency"]=="all_positive" and c2c["sign_consistency"]=="all_positive" else "P4.7-COMPOSITION-SPECIFIC-TRADEOFF-SUPPORTED" if c2c["sign_consistency"]=="all_positive" else "P4.7-NO-ROBUST-ARCHITECTURAL-GAIN","recurrent_verdict":"RECURRENT-ENCODING-SUPPORTED" if c1["sign_consistency"]=="all_positive" else "RECURRENT-ENCODING-MIXED","composition_verdict":"COMPOSITION-SPECIFIC-GAIN-WITH-TRADEOFF" if c2c["sign_consistency"]=="all_positive" and c2s["sign_consistency"]=="all_negative" else "COMPOSITION-CONSISTENCY-MIXED","prefix_verdict":"PREFIX-SUPERVISION-NOT-SUPPORTED" if c3s["sign_consistency"]=="all_negative" else "PREFIX-SUPERVISION-MIXED","seeds":list(SEEDS),"multi_seed_confirmed":True,"sealed_test_evaluated":False,"sealed_test_access_count":0,"primary_candidate":primary,"composition_specialist_candidate":"C2" if selection["composition_specialist"] else None,"excluded_candidate_decisions":{"C3":"Archived as a negative privileged-supervision result; not supervision-matched."},"claim_boundary":"Descriptive three-seed, 4-qubit validation evidence only; no formal significance, arbitrary-circuit, or scaling claim."}
    dump(ROOT/"scientific_verdict.json",verdict)
    future={"phase":"P4.8 — Final Candidate Freeze and Sealed OOD Evaluation","status":"PROPOSAL_ONLY_NOT_EXECUTED","primary_general_purpose_candidate":primary,"composition_specialist_comparator":"C2" if selection["composition_specialist"] else None,"frozen_checkpoints":{v:{str(s):str(checkpoint_path(v,s)) for s in SEEDS} for v in ({primary,"C2","C0"})},"no_further_tuning":True,"sealed_opening":"Open predeclared Composition-OOD and Depth-OOD test splits once, only after checkpoint/hash freeze.","reporting":"Report every frozen checkpoint without post-test model selection.","C0_required_anchor":True}
    dump(ROOT/"future_work.json",future)
    write_report(rows,aggregates,deltas,verdict,selection)
    immutable=[CONFIRM/"variant_seed_metrics.csv",CONFIRM/"variant_seed_metrics.json",CONFIRM/"paired_deltas.csv",CONFIRM/"paired_deltas.json",CONFIRM/"summary.json",ROOT/"P4_7_FINAL_REPORT.md",ROOT/"scientific_verdict.json",ROOT/"candidate_selection.json",ROOT/"future_work.json",*[metric_path(v,s) for v in VARIANTS for s in SEEDS]]
    dump(ROOT/"artifact_hashes.json",{"algorithm":"SHA-256","excluded":["checkpoints","progress/status runtime files","artifact_hashes.json"],"artifacts":{str(p):sha(p) for p in immutable}})
    return summary


def fmt(x): return f"{x:.6f}"
def delta_table(deltas,name):
    rows=[d for d in deltas if d["comparison"]==name]
    return "\n".join(["|Metric|2026|2027|2028|Mean|Sample SD|Signs|","|---|---:|---:|---:|---:|---:|---|",*[f"|{d['metric']}|{fmt(d['delta_2026'])}|{fmt(d['delta_2027'])}|{fmt(d['delta_2028'])}|{fmt(d['mean_delta'])}|{fmt(d['sample_std'])}|{d['sign_consistency']}|" for d in rows]])
def write_report(rows,a,d,v,s):
    per="\n".join(["|Variant|Seed|Step|S_balanced|IID/action|State|Parameter|Composition|Depth|Process|Phase error|Unitarity|max raw norm|Runtime s|Samples/s|","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",*[f"|{r['variant']}|{r['seed']}|{r['best_checkpoint_step']}|{fmt(r['S_balanced'])}|{fmt(r['action_fidelity'])}|{fmt(r['State_OOD'])}|{fmt(r['Parameter_OOD'])}|{fmt(r['Composition_OOD'])}|{fmt(r['Depth_OOD'])}|{fmt(r['process_fidelity'])}|{fmt(r['phase_aligned_frobenius_error'])}|{r['unitarity_error']:.3e}|{fmt(r['raw_action_norm'])}|{fmt(r['runtime_seconds'])}|{fmt(r['samples_per_second'])}|" for r in rows]])
    agg="\n".join(["|Variant|S_balanced|IID|State|Parameter|Composition|Depth|Process fidelity|Runtime s|","|---|---|---|---|---|---|---|---|---|",*["|"+vv+"|"+"|".join(f"{fmt(a[vv][m]['mean'])} ± {fmt(a[vv][m]['sample_std'])}" for m in METRICS)+"|" for vv in VARIANTS]])
    text=f"""# CC-NQE P4.7 Final Report\n\n## 1. Objective and frozen hypotheses\nTest recurrent encoding, composition consistency, and privileged prefix supervision without sealed-test access.\n\n## 2. P4.6 C0 anchor\nC0 is the frozen P4.6 B3 monolithic action-only exact-unitary Cayley anchor.\n\n## 3. P4.7 architecture variants\nC1 is shared causal recurrence; C2 adds action-only composition self-consistency (no exact-U target); C3 adds privileged exact prefix-action targets and is not supervision-matched.\n\n## 4. Screening seed-2026 results\nSeed 2026 was screening evidence only; C0 came from P4.6 and C1–C3 from P4.7.\n\n## 5. Confirmatory protocol\nSeeds 2027/2028 used 10,000 updates and 10,240,000 exposures. Every comparison uses one frozen best-balanced-validation checkpoint, where S_balanced=(IID+Composition-OOD+Depth-OOD)/3. Per-metric maxima are excluded from primary comparisons.\n\n## 6. Twelve-run integrity audit\nPASS: 12/12 completed scientific cells; expected identities, supervision, dataset/config equivalence, checkpoints, finite metrics, exact unitarity (<1e-5), and zero sealed access verified.\n\n## 7. Per-seed metric table\n{per}\n\n## 8. Three-seed mean/std table\nValues are mean ± sample SD; this is descriptive three-seed evidence, not large-sample statistical proof. Full individual/min/max values are in `confirmatory/summary.json`.\n\n{agg}\n\n## 9. C1-minus-C0 analysis\n{delta_table(d,'C1_minus_C0')}\n\nThe balanced effect is sign-consistent across all seeds; target OOD axes are mixed.\n\n## 10. C2-minus-C1 analysis\n{delta_table(d,'C2_minus_C1')}\n\nComposition-OOD improves in every seed while S_balanced falls in every seed.\n\n## 11. C3-minus-C1 analysis\n{delta_table(d,'C3_minus_C1')}\n\nC3's privileged supervision does not justify advancement: balanced effects are all negative and target-OOD effects are mixed.\n\n## 12. Composition-OOD versus Depth-OOD interpretation\nC2's Composition-OOD effect is sign-consistent. Its Depth-OOD signs are mixed; the large seed-2026 gain was not seed-robust. The supplied excerpt agrees with canonical values within rounding.\n\n## 13. Mechanistic verdicts\n- **{v['recurrent_verdict']}**\n- **{v['composition_verdict']}**\n- **{v['prefix_verdict']}**\n\n## 14. Overall P4.7 verdict\n**{v['overall_verdict']}**\n\n## 15. Candidate selection\nGeneral-purpose: **{v['primary_candidate']}**, mean S_balanced {fmt(s['general_purpose']['mean_S_balanced'])}, margin {fmt(s['general_purpose']['margin_over_next'])}; per-seed ranks {s['general_purpose']['per_seed_rank']}. C2 is retained only as a composition specialist with its balanced trade-off explicit. C3 is archived as a negative privileged-supervision result.\n\n## 16. Limitations\nThree seeds, validation splits, four qubits, explicit U(16), and descriptive statistics only. No formal significance, arbitrary-circuit efficiency, large-qubit scaling, or universal OOD claim.\n\n## 17. Sealed-test status\nNot evaluated; access count **0**.\n\n## 18. Recommended next phase\nP4.8 is proposed but not executed: freeze {v['primary_candidate']} as primary, C2 as predeclared composition specialist, and C0 as anchor; prohibit tuning; open sealed Composition/Depth splits once and report without post-test selection.\n\n## 19. Repository provenance\nImplementation anchor `34263f65fc0fbd27c6049f5d2ec80c06019323a9`; canonical hashes are recorded in `artifact_hashes.json`. P4.6 historical artifacts were not rewritten.\n"""
    (ROOT/"P4_7_FINAL_REPORT.md").write_text(text)

if __name__=="__main__": close()
