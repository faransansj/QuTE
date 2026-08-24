"""Command-line orchestration for the gated CC-NQE P4.5 study."""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cc_nqe import Gate, git_sha, sha256
from cc_nqe_p4_5 import (BASELINE, CONFIRM, GRID, MODEL_SCALES, ROOT, SCHEMA, SPLITS,
                         CircuitDataset, ScaledCCNQE, ShardedDataset, Progress, append_jsonl,
                         artifact_hashes, atomic_json, audit_dataset, baseline_integrity,
                         config_hash, environment, generate_dataset, load_checkpoint,
                         metric_summary, operator_fidelity, parameter_count,
                         phase_aligned_matrix_error, save_checkpoint, state_fidelity,
                         status, unitarity_error, xpu_preflight, _tensorize_circuit)

RECIPE = {"optimizer": "AdamW", "learning_rate": 3e-4, "scheduler": "cosine", "effective_batch_size": 1024,
          "maximum_updates": 10_000, "validation_interval": 500, "stopping_rule": "fixed optimizer updates",
          "screening_seed": 2026, "confirmatory_seeds": [2026, 2027, 2028], "dtype": "float32"}
_INTERRUPTED = False


def _signal(_signum, _frame):
    global _INTERRUPTED
    _INTERRUPTED = True


def require_baseline() -> dict[str, Any]:
    result = baseline_integrity()
    if result["status"] != "PASS":
        raise SystemExit("BASELINE-INTEGRITY-BLOCKED")
    return result


def require_preflight() -> dict[str, Any]:
    require_baseline()
    result = xpu_preflight()
    if result["status"] != "PASS":
        blocked("G1", result.get("reason", "XPU preflight failed"))
        raise SystemExit("XPU-BLOCKED: full scaling was not started")
    return result


def require_dataset() -> dict[str, Any]:
    path = ROOT / "datasets/audit.json"
    result = json.loads(path.read_text()) if path.exists() else audit_dataset()
    if result["status"] != "PASS":
        blocked("G2", "dataset audit failed")
        raise SystemExit("DATASET-BLOCKED: training was not started")
    return result


def blocked(gate: str, reason: str) -> None:
    row = {key: None for key in Progress.FIELDS}
    row.update(timestamp=time.time(), experiment_id="cc-nqe-p4.5", phase=gate, device=str(ACCEL_DEVICE), dtype="float32", state="BLOCKED")
    append_jsonl(ROOT / "progress.jsonl", row); atomic_json(ROOT / "status.json", row)
    atomic_json(ROOT / "final_gate.json", {"schema_version": SCHEMA, "scientific_verdict": "INCONCLUSIVE", "infrastructure_status": "XPU-BLOCKED", "blocked_gate": gate, "reason": reason,
                                                  "resumable_command": "uv run python run_p4_5.py run-all"})


def _collate(batch):
    return tuple(torch.as_tensor(np.stack(x)) for x in zip(*batch))


def _evaluate(model, dataset, device, maximum=2048) -> float:
    loader=torch.utils.data.DataLoader(dataset,batch_size=256,shuffle=False,collate_fn=_collate)
    values=[]; model.eval()
    with torch.inference_mode():
        for batch in loader:
            gates,qubits,params,mask,state,target=(x.to(device) for x in batch)
            pred=model(gates,qubits,params,mask,state); values.extend(state_fidelity(pred,target).cpu().tolist())
            if len(values)>=maximum: break
    model.train(); return float(np.mean(values))


def train_one(dataset_scale: str, model_scale: str, seed: int, maximum_updates: int | None = None, resume_path: Path | None = None, run_kind: str = "screening", validation_interval: int | None = None) -> dict[str, Any]:
    require_preflight(); require_dataset()
    torch.manual_seed(seed); device=ACCEL_DEVICE; count={"10k":10_000,"100k":100_000,"1m":1_000_000}[dataset_scale]
    train=ShardedDataset(ROOT/"datasets","train",count); validation=ShardedDataset(ROOT/"datasets","validation")
    model=ScaledCCNQE(model_scale,"state").to(device); actual=parameter_count(model); optimizer=torch.optim.AdamW(model.parameters(),lr=RECIPE["learning_rate"])
    total=maximum_updates or RECIPE["maximum_updates"]; interval=validation_interval or min(RECIPE["validation_interval"], total); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,total); manifest_hash=sha256(ROOT/"datasets/master_manifest.json")
    config={"schema_version":SCHEMA,"task":"state","dataset_scale":dataset_scale,"model_scale":model_scale,"actual_parameters":actual,"seed":seed,"dtype":"float32","recipe":RECIPE,"run_kind":run_kind,"update_budget":total,"scheduler_horizon":total,"validation_interval":interval}
    suffix="" if run_kind=="screening" else f"-{run_kind}"; experiment_id=f"state-{dataset_scale}-{model_scale}-seed{seed}{suffix}"; checkpoint=ROOT/f"checkpoints/{experiment_id}.pt"; atomic_json(ROOT/f"configs/{experiment_id}.json",config); step=samples=0; best=0.0
    if resume_path:
        payload=load_checkpoint(resume_path,model,optimizer,scheduler,config,manifest_hash); step=payload["step"]; samples=payload["samples_seen"]; best=payload["best_metric"]
    loader=torch.utils.data.DataLoader(train,batch_size=min(RECIPE["effective_batch_size"],len(train)),shuffle=True,collate_fn=_collate,drop_last=False); iterator=iter(loader); progress=Progress(); started=time.monotonic(); recent=[]
    initial_validation=_evaluate(model,validation,device,512); initial_train=_evaluate(model,train,device,512); latest_validation=initial_validation; latest_comp=latest_depth=None; loss=torch.tensor(0.0); fidelity=torch.tensor([0.0]); rate=0.0
    curve=[{"step":0,"learning_rate":RECIPE["learning_rate"],"train_fidelity":initial_train,"validation_fidelity":initial_validation}]
    initial_parameter=next(model.parameters()).detach().clone(); finite_gradients=True; xpu_residency=True
    while step<total and not _INTERRUPTED:
        try: batch=next(iterator)
        except StopIteration: iterator=iter(loader); batch=next(iterator)
        gates,qubits,params,mask,state,target=(x.to(device) for x in batch); optimizer.zero_grad(set_to_none=True); pred=model(gates,qubits,params,mask,state); fidelity=state_fidelity(pred,target); loss=(1-fidelity).mean(); loss.backward()
        finite_gradients &= all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        xpu_residency &= all(x.device.type==ACCEL for x in (gates,qubits,params,mask,state,target,pred,loss))
        if not torch.isfinite(loss) or not finite_gradients: raise FloatingPointError("non-finite loss or gradients")
        optimizer.step(); scheduler.step(); accel_synchronize(); step+=1; samples+=len(state)
        recent.append(time.monotonic()); recent=recent[-20:]; rate=(len(state)*(len(recent)-1)/(recent[-1]-recent[0])) if len(recent)>1 else 0.; eta=(total-step)*len(state)/rate if rate else None
        if step%interval==0 or step==total:
            latest_train=_evaluate(model,train,device); latest_validation=_evaluate(model,validation,device); latest_comp=_evaluate(model,ShardedDataset(ROOT/"datasets","composition_ood"),device); latest_depth=_evaluate(model,ShardedDataset(ROOT/"datasets","depth_ood"),device); best=max(best,latest_validation)
            curve.append({"step":step,"learning_rate":optimizer.param_groups[0]["lr"],"train_fidelity":latest_train,"validation_fidelity":latest_validation,"composition_ood_fidelity":latest_comp,"depth_ood_fidelity":latest_depth})
            save_checkpoint(checkpoint,model,optimizer,scheduler,config,manifest_hash,step,samples,best)
        progress.update(experiment_id=experiment_id,phase="G3" if total<RECIPE["maximum_updates"] else "G4",task="state",dataset_scale=dataset_scale,model_scale=model_scale,actual_parameters=actual,seed=seed,device=str(ACCEL_DEVICE),dtype="float32",step=step,maximum_steps=total,samples_seen=samples,training_loss=float(loss.detach()),training_fidelity=float(fidelity.mean().detach()),validation_fidelity=latest_validation,composition_ood_fidelity=latest_comp,depth_ood_fidelity=latest_depth,learning_rate=optimizer.param_groups[0]["lr"],samples_per_second=rate,elapsed_seconds=time.monotonic()-started,eta_seconds=eta,best_metric=best,checkpoint=str(checkpoint),state="RUNNING")
    save_checkpoint(checkpoint,model,optimizer,scheduler,config,manifest_hash,step,samples,best)
    state="INTERRUPTED" if _INTERRUPTED else "COMPLETED"; progress.update(experiment_id=experiment_id,phase="G4",task="state",dataset_scale=dataset_scale,model_scale=model_scale,actual_parameters=actual,seed=seed,device=str(ACCEL_DEVICE),dtype="float32",step=step,maximum_steps=total,samples_seen=samples,training_loss=float(loss),training_fidelity=float(fidelity.mean()),validation_fidelity=latest_validation,composition_ood_fidelity=latest_comp,depth_ood_fidelity=latest_depth,learning_rate=optimizer.param_groups[0]["lr"],samples_per_second=rate,elapsed_seconds=time.monotonic()-started,eta_seconds=0,best_metric=best,checkpoint=str(checkpoint),state=state)
    result={"experiment_id":experiment_id,"state":state,"initial_train_fidelity":initial_train,"initial_validation_fidelity":initial_validation,"final_validation_fidelity":latest_validation,"best_validation_fidelity":best,"steps":step,"samples_seen":samples,"wall_seconds":time.monotonic()-started,"checkpoint":str(checkpoint),"config":config,"curve":curve,"finite_loss":bool(torch.isfinite(loss)),"finite_gradients":finite_gradients,"parameter_updated":not torch.equal(initial_parameter,next(model.parameters()).detach()),"xpu_residency":xpu_residency,"no_nan_inf":bool(torch.isfinite(loss) and torch.isfinite(fidelity).all()),"validation_pipeline":bool(np.isfinite(initial_validation) and np.isfinite(latest_validation))}
    atomic_json(ROOT/f"metrics/state/{experiment_id}.json",result); return result


def _checkpoint_roundtrip(result: dict[str, Any]) -> bool:
    config=result["config"]; model=ScaledCCNQE(config["model_scale"],"state"); optimizer=torch.optim.AdamW(model.parameters(),lr=RECIPE["learning_rate"]); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,config["scheduler_horizon"])
    payload=load_checkpoint(Path(result["checkpoint"]),model,optimizer,scheduler,config,sha256(ROOT/"datasets/master_manifest.json"))
    return payload["step"]==result["steps"] and payload["samples_seen"]==result["samples_seen"]


def _smoke_gate_checks(result: dict[str, Any], checkpoint_ok: bool) -> dict[str, bool]:
    checks={key:result[key] for key in ("finite_loss","finite_gradients","parameter_updated","xpu_residency","no_nan_inf","validation_pipeline")}; checks["checkpoint_save_resume"]=checkpoint_ok
    return checks


def smoke() -> dict[str, Any]:
    result=train_one("10k","60k",RECIPE["screening_seed"],maximum_updates=10,run_kind="smoke",validation_interval=10)
    checks=_smoke_gate_checks(result,_checkpoint_roundtrip(result))
    result["gate_checks"]=checks; result["gate_status"]="PASS" if all(checks.values()) else "TRAINING-BLOCKED"; atomic_json(ROOT/"metrics/state/smoke_summary.json",result)
    if result["gate_status"]!="PASS": raise SystemExit("TRAINING-BLOCKED: smoke correctness requirement failed")
    return result


def calibrate(updates: int = 750) -> dict[str, Any]:
    result=train_one("10k","60k",RECIPE["screening_seed"],maximum_updates=updates,run_kind="calibration",validation_interval=100)
    best_train=max(point["train_fidelity"] for point in result["curve"]); signal=best_train > result["initial_train_fidelity"] + 0.005
    summary={"experiment_id":result["experiment_id"],"random_state_reference":1/16,"scheduler_horizon":updates,"update_budget":updates,"initial_train_fidelity":result["initial_train_fidelity"],"best_train_fidelity":best_train,"initial_validation_fidelity":result["initial_validation_fidelity"],"best_validation_fidelity":result["best_validation_fidelity"],"learnability_signal":signal,"screening_ready":signal and all(result[k] for k in ("finite_loss","finite_gradients","parameter_updated","xpu_residency","no_nan_inf","validation_pipeline")),"curve":result["curve"],"checkpoint":result["checkpoint"]}
    atomic_json(ROOT/"metrics/state/calibration_summary.json",summary); return summary


def screen() -> None:
    calibration_path=ROOT/"metrics/state/calibration_summary.json"
    if not calibration_path.exists() or not json.loads(calibration_path.read_text()).get("screening_ready"): raise SystemExit("TRAINING-BLOCKED: successful learnability calibration required")
    for data,model in GRID: train_one(data,model,RECIPE["screening_seed"])


def confirm() -> None:
    for data,model in CONFIRM:
        for seed in RECIPE["confirmatory_seeds"]: train_one(data,model,seed)


def _evaluate_operator(model, dataset, device, maximum=2048) -> float:
    loader=torch.utils.data.DataLoader(dataset,batch_size=128,shuffle=False,collate_fn=_collate); values=[]; model.eval()
    with torch.inference_mode():
        for gates,qubits,params,mask,target in loader:
            gates,qubits,params,mask,target=(x.to(device) for x in (gates,qubits,params,mask,target)); exact=torch.complex(target[:,0],target[:,1]); values.extend(operator_fidelity(model(gates,qubits,params,mask),exact).cpu().tolist())
            if len(values)>=maximum: break
    model.train(); return float(np.mean(values))


def _operator_diagnostics(model, dataset, device, maximum=10_000) -> dict[str,Any]:
    loader=torch.utils.data.DataLoader(dataset,batch_size=128,shuffle=False,collate_fn=_collate); fidelities=[]; matrix_errors=[]; unity_errors=[]; model.eval()
    with torch.inference_mode():
        for gates,qubits,params,mask,target in loader:
            gates,qubits,params,mask,target=(x.to(device) for x in (gates,qubits,params,mask,target)); exact=torch.complex(target[:,0],target[:,1]); pred=model(gates,qubits,params,mask)
            fidelities.extend(operator_fidelity(pred,exact).cpu().tolist()); p,e=pred.cpu().numpy(),exact.cpu().numpy(); matrix_errors.extend(phase_aligned_matrix_error(p,e).tolist()); unity_errors.extend(unitarity_error(p).tolist())
            if len(fidelities)>=maximum: break
    model.train(); return {"operator_fidelity":metric_summary(fidelities),"normalized_matrix_error":metric_summary(matrix_errors),"raw_unitarity_error":metric_summary(unity_errors)}


def _composition_diagnostic(model, device, maximum=64) -> dict[str,Any]:
    rows=[json.loads(x) for x in (ROOT/"datasets/circuits.jsonl").read_text().splitlines() if json.loads(x)["split"]=="validation"]
    circuits=[[Gate.from_dict(g) for g in json.loads(row["serialized"])] for row in rows]
    values=[]; model.eval()
    with torch.inference_mode():
        for i in range(min(maximum,len(circuits)//2)):
            first,second=circuits[2*i],circuits[2*i+1]; combined=first+second
            if len(combined)>16: continue
            batch=[]
            for circuit in (first,second,combined): batch.append(tuple(torch.as_tensor(x)[None].to(device) for x in _tensorize_circuit(circuit)))
            u1=model(*batch[0]); u2=model(*batch[1]); u21=model(*batch[2]); values.extend(operator_fidelity(u21,u2@u1).cpu().tolist())
    model.train(); return metric_summary(values)


def train_operator(circuit_count: int, model_scale: str, seed: int, resume_path: Path | None = None) -> dict[str,Any]:
    require_preflight(); require_dataset(); device=ACCEL_DEVICE; torch.manual_seed(seed)
    train=CircuitDataset(ROOT/"datasets","train",circuit_count); validation=CircuitDataset(ROOT/"datasets","validation"); model=ScaledCCNQE(model_scale,"operator").to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=RECIPE["learning_rate"]); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,RECIPE["maximum_updates"]); manifest_hash=sha256(ROOT/"datasets/master_manifest.json")
    config={"schema_version":SCHEMA,"task":"operator","unique_circuits":len(train),"model_scale":model_scale,"actual_parameters":parameter_count(model),"seed":seed,"dtype":"float32","recipe":RECIPE}; experiment_id=f"operator-{len(train)}-{model_scale}-seed{seed}"; checkpoint=ROOT/f"checkpoints/{experiment_id}.pt"; atomic_json(ROOT/f"configs/{experiment_id}.json",config); loader=torch.utils.data.DataLoader(train,batch_size=min(128,len(train)),shuffle=True,collate_fn=_collate); iterator=iter(loader); initial=_evaluate_operator(model,validation,device); latest=initial; started=time.monotonic(); samples=0; start_step=0; progress=Progress()
    if resume_path:
        payload=load_checkpoint(resume_path,model,optimizer,scheduler,config,manifest_hash); start_step=payload["step"]; samples=payload["samples_seen"]; latest=payload["best_metric"]
    step=start_step
    for step in range(start_step+1,RECIPE["maximum_updates"]+1):
        if _INTERRUPTED: break
        try: batch=next(iterator)
        except StopIteration: iterator=iter(loader); batch=next(iterator)
        gates,qubits,params,mask,target=(x.to(device) for x in batch); exact=torch.complex(target[:,0],target[:,1]); optimizer.zero_grad(set_to_none=True); pred=model(gates,qubits,params,mask); fidelity=operator_fidelity(pred,exact); loss=(1-fidelity).mean(); loss.backward(); optimizer.step(); scheduler.step(); samples+=len(gates)
        if not torch.isfinite(loss): raise FloatingPointError("non-finite operator loss")
        if step%RECIPE["validation_interval"]==0 or step==RECIPE["maximum_updates"]:
            accel_synchronize(); latest=_evaluate_operator(model,validation,device); save_checkpoint(checkpoint,model,optimizer,scheduler,config,manifest_hash,step,samples,latest)
            progress.update(experiment_id=experiment_id,phase="G4",task="operator",dataset_scale=f"{len(train)} circuits",model_scale=model_scale,actual_parameters=parameter_count(model),seed=seed,device=str(ACCEL_DEVICE),dtype="float32",step=step,maximum_steps=RECIPE["maximum_updates"],samples_seen=samples,training_loss=float(loss),training_fidelity=float(fidelity.mean()),validation_fidelity=latest,composition_ood_fidelity=None,depth_ood_fidelity=None,learning_rate=optimizer.param_groups[0]["lr"],samples_per_second=samples/(time.monotonic()-started),elapsed_seconds=time.monotonic()-started,eta_seconds=(RECIPE["maximum_updates"]-step)*(time.monotonic()-started)/step,best_metric=latest,checkpoint=str(checkpoint),state="RUNNING")
    diagnostics={split:_operator_diagnostics(model,CircuitDataset(ROOT/"datasets",split),device) for split in ("iid","state_ood","parameter_interpolation","parameter_extrapolation","composition_ood","depth_ood")}; composition=_composition_diagnostic(model,device)
    result={"experiment_id":experiment_id,"state":"INTERRUPTED" if _INTERRUPTED else "COMPLETED","initial_operator_fidelity":initial,"final_operator_fidelity":latest,"unique_circuits":len(train),"model_scale":model_scale,"seed":seed,"checkpoint":str(checkpoint),"split_diagnostics":diagnostics,"composition_consistency":composition}; atomic_json(ROOT/f"metrics/operator/{experiment_id}.json",result); atomic_json(ROOT/f"metrics/composition/{experiment_id}.json",composition); return result


def operator() -> None:
    require_preflight(); require_dataset(); manifest=json.loads((ROOT/"datasets/master_manifest.json").read_text()); largest=manifest["train_unique_circuits"]
    results=[train_operator(10_000,"250k",RECIPE["screening_seed"]),train_operator(100_000,"1m",RECIPE["screening_seed"]),train_operator(largest,"5m",RECIPE["screening_seed"])]
    if results[-1]["final_operator_fidelity"] > results[-1]["initial_operator_fidelity"] + 0.01:
        for seed in RECIPE["confirmatory_seeds"][1:]: results.append(train_operator(largest,"5m",seed))
    atomic_json(ROOT/"metrics/operator/summary.json",results)


def resume(experiment_id: str) -> None:
    config_path=ROOT/f"configs/{experiment_id}.json"
    if not config_path.exists(): raise SystemExit(f"missing frozen config: {config_path}")
    config=json.loads(config_path.read_text()); checkpoint=ROOT/f"checkpoints/{experiment_id}.pt"
    if config["task"]=="state": train_one(config["dataset_scale"],config["model_scale"],config["seed"],maximum_updates=config.get("update_budget"),resume_path=checkpoint,run_kind=config.get("run_kind","screening"),validation_interval=config.get("validation_interval"))
    elif config["task"]=="operator": train_operator(config["unique_circuits"],config["model_scale"],config["seed"],resume_path=checkpoint)
    else: raise SystemExit(f"unsupported checkpoint task: {config['task']}")


def initialize_blocked_artifacts() -> None:
    for path in ("datasets","configs","checkpoints","metrics/state","metrics/operator","metrics/composition"):
        (ROOT/path).mkdir(parents=True,exist_ok=True)
    atomic_json(ROOT/"configs/study_protocol.json",{"schema_version":SCHEMA,"grid":GRID,"confirmatory":CONFIRM,"model_scales":MODEL_SCALES,"training_recipe":RECIPE,"git_sha":git_sha(),"baseline_manifest_sha256":sha256(BASELINE/"artifact_hashes.json"),"source_sha256":{name:sha256(Path(name)) for name in ("cc_nqe_p4_5.py","run_p4_5.py","tests/test_cc_nqe_p4_5.py")}})
    atomic_json(ROOT/"scaling_summary.json",{"schema_version":SCHEMA,"status":"NOT_RUN","reason":"XPU-BLOCKED","screening":[],"confirmatory":[],"operator":[]})
    for name in ("master_manifest.json","train_10k_manifest.json","train_100k_manifest.json","train_1m_manifest.json","evaluation_manifest.json"):
        atomic_json(ROOT/"datasets"/name,{"schema_version":SCHEMA,"status":"NOT_GENERATED","reason":"XPU-BLOCKED"})
    atomic_json(ROOT/"datasets/audit.json",{"schema_version":SCHEMA,"gate":"G2","status":"NOT_RUN","reason":"XPU-BLOCKED at G1","checks":{}})
    (ROOT/"datasets/hashes.sha256").write_text("")
    with (ROOT/"scaling_summary.csv").open("w",newline="") as handle: csv.writer(handle).writerow(("task","dataset_scale","model_scale","seed","status","mean_fidelity"))
    if not (ROOT/"status.json").exists(): blocked("G1","No native CUDA/XPU accelerator available")


def report() -> str:
    integrity=json.loads((ROOT/"baseline_integrity.json").read_text()) if (ROOT/"baseline_integrity.json").exists() else baseline_integrity(); env=environment(); pre=json.loads((ROOT/"xpu_preflight.json").read_text()) if (ROOT/"xpu_preflight.json").exists() else xpu_preflight(); final=json.loads((ROOT/"final_gate.json").read_text()) if (ROOT/"final_gate.json").exists() else {"scientific_verdict":"INCONCLUSIVE"}
    counts={s:parameter_count(ScaledCCNQE(s,"state")) for s in MODEL_SCALES}
    lines=["# CC-NQE P4.5 Scale-Up and Operator Learnability Report","",
    "## 1. Starting repository state",f"- Branch: `research/cc-nqe-p4-5-scaling`; starting HEAD: `{git_sha()}`; inherited P1–P4 working tree was uncommitted.",f"- Baseline integrity: **{integrity['status']}**, {integrity['verified']}/{integrity['entries']} hashes verified; frozen namespace `{BASELINE}` was not modified.","",
    "## 2. Environment and XPU validation",f"- CPU: {env['cpu']}; RAM: {env['ram_bytes']} bytes; OS: {env['os']}; Python: {env['python']}; PyTorch: {env['torch']}.",f"- OS-visible graphics: `{env['pci_graphics_devices']}`. XPU available/count: {env['xpu_available']}/{env['xpu_device_count']}; PyTorch device names: `{env['xpu_device_names']}`; FP32 preflight: **{pre['status']}**.","- Forward/backward/optimizer/parity/device-residency results could not run because native PyTorch exposes no XPU device. No CPU fallback was used.","",
    "## 3. Dataset construction","- Not run: G1 precedes large dataset generation in `run-all`; no scale data were generated after XPU-BLOCKED. The implemented generator uses deduplicated circuit/state tables, one unitary per circuit, memory-mapped sharded pairs/targets, fixed evaluation sets, and prefix-nested manifests.","",
    "## 4. Dataset audit","- Not run on a P4.5 master dataset. Audit code covers counts, normalization, unitary error, duplicates, circuit/structure/parameter/composition/depth leakage, nested subsets, and evaluation immutability.","",
    "## 5. Training protocol",f"- Frozen proposed recipe: `{json.dumps(RECIPE,sort_keys=True)}`. It was not executed.","",
    "## 6. Model configurations",*(f"- {scale}: {counts[scale]:,} state-model parameters; `{MODEL_SCALES[scale]}`; FP32 on XPU required." for scale in MODEL_SCALES),"",
    "## 7. Screening results","- All seven grid points: **NOT RUN — XPU-BLOCKED**.","",
    "## 8. Confirmatory results","- No confirmatory seeds ran.","",
    "## 9. State-transformation scaling","- No data-scaling or model-scaling inference is possible.","",
    "## 10. Operator learning","- No operator training ran. Shared circuit encoder, Hilbert–Schmidt fidelity, phase-aligned matrix error, and raw unitarity metrics are implemented and tested.","",
    "## 11. Failure localization","- Not measured; direct-state and predicted-operator application cannot be compared without valid trained checkpoints.","",
    "## 12. Composition results","- Not measured; phase-invariant `U(C2∘C1)` versus `U(C2)U(C1)` ordering diagnostic is implemented and tested.","",
    "## 13. Runtime and XPU utilization","- Dataset and training runtime: not measured. XPU memory and throughput: unavailable because no native XPU device was exposed.","",
    "## 14. Scaling conclusion","- More data: unknown. Larger model: unknown. Relative effect: unknown. Saturation: unknown. Bottleneck localization: unknown.","",
    "## 15. Final verdict",f"**{final['scientific_verdict']}**",f"- Infrastructure status: `{final.get('infrastructure_status','XPU-BLOCKED')}`. This is not a scientific NO-GO.","",
    "## 16. Minimal next experiment","- Install/use a native Intel-XPU-enabled PyTorch runtime, then run `uv run python run_p4_5.py run-all`. G1 will rerun all numerical and device-residency checks before dataset generation or training.","",
    "## 17. Repository changes","- Added the P4.5 module, CLI, focused tests, and blocked provenance artifacts under `artifacts/cc_nqe_p4_5/`. No commit or push was performed; P1–P4 artifacts remain frozen.",""]
    text="\n".join(lines); (ROOT/"REPORT.md").write_text(text); artifact_hashes(); return text


def run_all(args) -> None:
    integrity=require_baseline()
    pre=xpu_preflight()
    if pre["status"]!="PASS":
        initialize_blocked_artifacts(); report(); raise SystemExit("XPU-BLOCKED: resume with `uv run python run_p4_5.py run-all` after native XPU is available")
    generate_dataset(master_samples=args.master_samples,eval_per_split=args.eval_per_split); audit=audit_dataset()
    if audit["status"]!="PASS": raise SystemExit("DATASET-BLOCKED")
    smoke(); screen(); confirm(); operator(); report()


def main() -> None:
    signal.signal(signal.SIGINT,_signal); signal.signal(signal.SIGTERM,_signal)
    parser=argparse.ArgumentParser(description="CC-NQE P4.5 gated scaling study"); parser.add_argument("command",choices=("preflight","generate","audit","smoke","calibrate","screen","confirm","operator","status","resume","report","run-all")); parser.add_argument("--experiment-id"); parser.add_argument("--master-samples",type=int,default=1_000_000); parser.add_argument("--eval-per-split",type=int,default=10_000); args=parser.parse_args(); ROOT.mkdir(parents=True,exist_ok=True)
    if args.command != "status": require_baseline()
    if args.command=="preflight":
        print(json.dumps(xpu_preflight(),indent=2))
    elif args.command=="generate":
        require_preflight(); print(json.dumps(generate_dataset(master_samples=args.master_samples,eval_per_split=args.eval_per_split),indent=2))
    elif args.command=="audit":
        require_preflight(); print(json.dumps(audit_dataset(),indent=2))
    elif args.command=="smoke": print(json.dumps(smoke(),indent=2))
    elif args.command=="calibrate": print(json.dumps(calibrate(),indent=2))
    elif args.command=="screen": screen()
    elif args.command=="confirm": confirm()
    elif args.command=="operator": operator()
    elif args.command=="status": print(json.dumps(status(),indent=2))
    elif args.command=="resume":
        if not args.experiment_id: parser.error("resume requires --experiment-id")
        resume(args.experiment_id)
    elif args.command=="report": print(report())
    elif args.command=="run-all": run_all(args)

if __name__=="__main__": main()
