"""P4.6-C Track-B operator diagnostics. Scientific runs are manual only."""
from __future__ import annotations

import contextlib, hashlib, io, json, math, os, signal, time, warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cc_nqe import ACCEL, ACCEL_DEVICE, DIM, Gate, accel_device_name, accel_profiler_activities, accel_synchronize, circuit_unitary, serialize_circuit
from cc_nqe_p4_5 import ScaledCCNQE, apply_operator, atomic_json, parameter_count, state_fidelity, _tensorize_circuit
from cc_nqe_p4_6 import OperatorModel, ROOT, SCHEMA, SEED, normalized_action_fidelity, phase_aligned_error, process_fidelity, raw_unitarity_error, digest
from cc_nqe_p4_6_track_a import ArmData, DATA_ROOT, VALIDATION_SPLITS, load_validation, _decode

OP_ROOT=ROOT/"operator"
ALLOCATION={"unique_circuits":58_824,"probes_per_circuit":17,"state_action_pairs":1_000_008,"source_arm":"A4"}
RECIPE={"seed":SEED,"device":str(ACCEL_DEVICE),"dtype":"float32","optimizer":"AdamW","learning_rate":3e-4,"scheduler":"cosine","effective_batch_size":1024,"maximum_updates":10_000,"validation_interval":500}
VARIANTS={
 "B0":{"model":"direct-state","supervision":["C","psi_in","psi_out"],"privileged":False},
 "B1":{"model":"unconstrained-operator","supervision":["C","psi_in","psi_out"],"privileged":False},
 "B2":{"model":"soft-unitarity-operator","supervision":["C","psi_in","psi_out"],"privileged":False,"lambda_unitary":0.1,"lambda_policy":"predeclared value from allowed {0.01,0.1}; no scientific search"},
 "B3":{"model":"action-only exact-unitarity via basic Cayley parameterization","parameterization":"basic_cayley","cayley_scale":1.0,"supervision":["C","psi_in","psi_out"],"privileged":False},
 "B4":{"model":"unconstrained-operator","supervision":["C","psi_in","psi_out","exact_U"],"privileged":True,"label":"PRIVILEGED OPERATOR SUPERVISION","lambda_process":1.0},
 "B5":{"model":"unconstrained-operator","supervision":["C","psi_in","psi_out","exact_U"],"privileged":True,"label":"PRIVILEGED OPERATOR SUPERVISION","lambda_process":1.0,"lambda_comp":0.3,"composition_batch_size":64,"composition_contract":"independent C1, C2, and C2-compose-C1 predictions; U(C2 compose C1)=U(C2)U(C1)"},
}
_STOP=False


def _sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""): h.update(block)
 return h.hexdigest()


def _model(variant:str,device:torch.device|str="cpu"):
 model=ScaledCCNQE("1m","state") if variant=="B0" else OperatorModel("cayley" if variant=="B3" else "unconstrained")
 return model.to(device)


def variant_config(variant:str)->dict[str,Any]:
 if variant not in VARIANTS: raise ValueError(variant)
 count=parameter_count(_model(variant))
 return {"schema_version":SCHEMA,"variant":variant,"allocation":ALLOCATION,"recipe":RECIPE,"actual_parameters":count,"parameter_budget_target":1_000_000,"parameter_budget_tolerance_fraction":0.15,"parameter_budget_valid":850_000<=count<=1_150_000,**VARIANTS[variant],"scientific_metrics_at_freeze":"NOT_RUN"}


def prepare_operator_artifacts()->dict[str,Any]:
 for part in ("configs","metrics","checkpoints","data","preflight","smoke"): (OP_ROOT/part).mkdir(parents=True,exist_ok=True)
 configs={v:variant_config(v) for v in VARIANTS}
 for v,c in configs.items(): atomic_json(OP_ROOT/f"configs/{v}.json",c)
 summary={"schema_version":SCHEMA,"state":"IMPLEMENTED_NOT_SCIENTIFICALLY_RUN","allocation":ALLOCATION,"recipe":RECIPE,"variants":{v:{"actual_parameters":c["actual_parameters"],"parameter_budget_valid":c["parameter_budget_valid"],"supervision":c["supervision"],"privileged":c["privileged"],"scientific_state":"NOT_RUN"} for v,c in configs.items()},"B2_lambda_unitary":0.1,"B3_description":"action-only exact-unitarity via basic Cayley parameterization","B3_cayley_limitation":"Cayley and matrix exponential are different parameterizations. A basic Cayley chart does not globally cover every element of U(16), notably the -1-eigenvalue boundary. Global-phase-invariant state/process metrics mitigate but do not remove this limitation.","B5_lambda_comp":0.3,"sealed_splits_loaded":False,"sealed_access_count":0,"scalability_scope":"explicit 16x16 operator prediction is a four-qubit diagnostic only"}
 atomic_json(OP_ROOT/"metrics/status.json",{"schema_version":SCHEMA,"scientific_runs":{v:"NOT_RUN" for v in VARIANTS}})
 atomic_json(OP_ROOT/"protocol_summary.json",summary); return summary


def require_operator_preconditions()->None:
 summary=json.loads((ROOT/"factorial/summary.json").read_text()); access=json.loads((ROOT/"test_access_log.json").read_text())
 if summary["state"]!="COMPLETED" or summary["winner"]!="A3" or summary["verdict"]!="MIXED-DATA-EFFECT": raise RuntimeError("TRACK-A-INTEGRITY-BLOCKED")
 if access["access_count"]!=0: raise RuntimeError("SEALED-TEST-ACCESS-BLOCKED")
 if not (DATA_ROOT/"A4/manifest.json").exists(): raise RuntimeError("A4 dataset payload unavailable")
 manifest=json.loads((DATA_ROOT/"A4/manifest.json").read_text())
 if (manifest["circuit_count"],manifest["probes_per_circuit"],manifest["pair_count"])!=(58_824,17,1_000_008): raise RuntimeError("A4 allocation differs")


def _complex_state(value:torch.Tensor)->torch.Tensor: return torch.complex(value[...,:DIM],value[...,DIM:])


def operator_action(operator:torch.Tensor,state:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
 action=operator@_complex_state(state).unsqueeze(-1); action=action.squeeze(-1)
 return torch.cat((action.real,action.imag),-1),torch.linalg.vector_norm(action,dim=-1)


def action_metrics(operator:torch.Tensor,state:torch.Tensor,target:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
 action,norm=operator_action(operator,state); fidelity,_=normalized_action_fidelity(_complex_state(target),_complex_state(action)); return fidelity,norm


def soft_unitarity_loss(operator:torch.Tensor)->torch.Tensor:
 eye=torch.eye(DIM,dtype=operator.dtype,device=operator.device)
 return (operator.mH@operator-eye).abs().square().sum((-2,-1)).div(DIM).mean()


def b5_composition_loss(direct:torch.Tensor,second:torch.Tensor,first:torch.Tensor)->torch.Tensor:
 return (1-process_fidelity(direct,second@first)).mean()


def phase_aligned_normalized_frobenius_error(pred:torch.Tensor,target:torch.Tensor)->torch.Tensor:
 return phase_aligned_error(pred,target)


def _circuit_tensors(circuits:list[list[Gate]],device)->tuple[torch.Tensor,...]:
 columns=list(zip(*[_tensorize_circuit(c) for c in circuits])); return tuple(torch.as_tensor(np.asarray(x)).to(device) for x in columns)


def _b3_parts(model:OperatorModel,args:tuple[torch.Tensor,...]):
 raw=model.head(model.circuit(*args)).reshape(-1,2,DIM,DIM)
 b=torch.complex(raw[:,0],raw[:,1]); a=(b-b.mH)/2
 eye=torch.eye(DIM,dtype=a.dtype,device=a.device).expand_as(a)
 return a,eye+a,eye-a,torch.linalg.solve(eye+a,eye-a)


def operator_preflight()->dict[str,Any]:
 prepare_operator_artifacts(); config=variant_config("B3")
 result={"schema_version":SCHEMA,"scientific_run":False,"purpose":"implementation_validation","variant":"B3","maximum_updates":1,"config_hash":digest(config),"device":str(ACCEL_DEVICE),"checks":{},"status":"B3-XPU-BLOCKED","thresholds":{"mean_unitarity_error":1e-5,"max_unitarity_error":1e-4,"cpu_xpu_max_element_difference":1e-4,"cpu_xpu_action_fidelity_difference":1e-5}}
 path=OP_ROOT/"preflight/B3.json"
 if ACCEL == "cpu": result["reason"]="no native CUDA/XPU accelerator; no CPU fallback"; atomic_json(path,result); return result
 stderr=io.StringIO()
 try:
  from torch.profiler import profile, ProfilerActivity
  with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stderr(stderr):
   warnings.simplefilter("always"); torch.manual_seed(SEED)
   cpu=_model("B3").eval(); xpu=_model("B3",ACCEL_DEVICE); xpu.load_state_dict(cpu.state_dict()); xpu.eval()
   circuits=[[Gate("H",(0,)),Gate("CNOT",(0,1))],[Gate("RX",(2,),.4),Gate("X",(3,))],[Gate("RY",(1,),-.7)]]; args=_circuit_tensors(circuits,"cpu")
   with torch.no_grad(): cpu_a,cpu_lhs,cpu_rhs,reference=_b3_parts(cpu,args)
   xargs=tuple(x.to(ACCEL_DEVICE) for x in args); target=torch.randn(len(circuits),2*DIM,device=str(ACCEL_DEVICE))
   opt=torch.optim.AdamW(xpu.parameters(),3e-4); before=next(xpu.parameters()).detach().clone()
   with profile(activities=accel_profiler_activities()) as prof:
    a,lhs,rhs,out=_b3_parts(xpu,xargs); fidelity,norm=action_metrics(out,torch.randn_like(target),target); loss=(1-fidelity).mean(); opt.zero_grad(); loss.backward(); opt.step(); accel_synchronize()
   finite_grad=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in xpu.parameters()); nonzero_grad=any(p.grad is not None and bool((p.grad.abs()>0).any()) for p in xpu.parameters()); updated=not torch.equal(before,next(xpu.parameters()).detach())
   xpu.load_state_dict(cpu.state_dict()); xpu.eval()
   with torch.no_grad(): xa,xlhs,xrhs,parity=_b3_parts(xpu,xargs); accel_synchronize()
  messages=[str(w.message) for w in caught]+stderr.getvalue().splitlines(); fallback=[m for m in messages if "fallback" in m.lower() and ACCEL in m.lower()]
  events={e.key:e.device_time_total for e in prof.key_averages()}; solve_device_time=sum(t for k,t in events.items() if "solve" in k.lower() or "getrf" in k.lower() or "trsm" in k.lower())
  unity=raw_unitarity_error(out.detach()); element_diff=float((reference-parity.cpu()).abs().max()); cpu_f,_=action_metrics(reference,torch.ones(len(circuits),2*DIM),torch.ones(len(circuits),2*DIM)); xpu_f,_=action_metrics(parity.cpu(),torch.ones(len(circuits),2*DIM),torch.ones(len(circuits),2*DIM)); fidelity_diff=float((cpu_f-xpu_f).abs().max())
  singular=torch.linalg.svdvals(cpu_lhs); cond=singular[...,0]/singular[...,-1]
  perturb=cpu_a.clone(); perturb[:,0,1]+=1e-6; perturb[:,1,0]-=1e-6; eye=torch.eye(DIM,dtype=perturb.dtype).expand_as(perturb); changed=torch.linalg.solve(eye+perturb,eye-perturb)
  checks={"inputs_xpu0":all(x.device==ACCEL_DEVICE for x in xargs),"A_xpu0":a.device==ACCEL_DEVICE,"solve_inputs_output_xpu0":lhs.device==rhs.device==out.device==ACCEL_DEVICE,"forward":bool(torch.isfinite(out).all()),"backward":finite_grad,"finite_nonzero_gradients":finite_grad and nonzero_grad,"optimizer_update":updated,"no_nan_inf":bool(torch.isfinite(out).all() and torch.isfinite(loss) and torch.isfinite(norm).all()),"no_hidden_cpu_transfer":not fallback,"no_xpu_cpu_fallback":not fallback,"profiler_native_xpu_solve":solve_device_time>0,"skew_hermitian":bool(torch.allclose(a,-a.mH,atol=1e-6)),"identity":bool(torch.equal(torch.linalg.solve(torch.eye(DIM,dtype=torch.complex64),torch.eye(DIM,dtype=torch.complex64)),torch.eye(DIM,dtype=torch.complex64))),"near_identity":float((changed-torch.eye(DIM,dtype=changed.dtype)).abs().max())<1.0,"continuous_under_perturbation":float((changed-reference).abs().max())<1e-3,"unitarity_thresholds":float(unity.mean())<=1e-5 and float(unity.max())<=1e-4,"cpu_xpu_element_parity":element_diff<=1e-4,"cpu_xpu_action_fidelity_parity":fidelity_diff<=1e-5,"initialization_not_saturated":float(cond.max())<1e4}
  status="PASS" if all(checks.values()) else ("B3-NUMERICAL-BLOCKED" if not checks["unitarity_thresholds"] or not checks["cpu_xpu_element_parity"] or not checks["cpu_xpu_action_fidelity_parity"] else "B3-XPU-BLOCKED")
  result.update(status=status,checks=checks,unitarity_error={"mean":float(unity.mean()),"max":float(unity.max()),"p95":float(torch.quantile(unity,.95))},raw_output_state_norm={"mean":float(norm.mean().detach().cpu()),"max_error_from_one":float((norm-1).abs().max().detach().cpu())},solve_failures=0,finite_status=checks["no_nan_inf"],cpu_xpu_max_element_difference=element_diff,cpu_xpu_action_fidelity_difference=fidelity_diff,conditioning={"min_singular_value":float(singular.min()),"max_condition_estimate":float(cond.max())},profiler={"solve_device_time_total":solve_device_time,"solve_events":[k for k in events if "solve" in k.lower() or "getrf" in k.lower() or "trsm" in k.lower()]},warnings=messages,device_name=accel_device_name())
 except Exception as exc: result["reason"]=f"{type(exc).__name__}: {exc}"; result["stderr"]=stderr.getvalue()
 atomic_json(path,result); return result

def _tiny_batch(device):
 data=ArmData("A4"); x=[torch.as_tensor(v).to(device) for v in data.batch(np.arange(4))]; return x


def _loss(variant:str,model,batch,exact_u:torch.Tensor|None=None,composition:tuple|None=None):
 gates,qubits,params,mask,state,target=batch
 if variant=="B0":
  pred=model(gates,qubits,params,mask,state); f=state_fidelity(pred,target); return (1-f).mean(),f,torch.linalg.vector_norm(_complex_state(pred),dim=-1),None
 operator=model(gates,qubits,params,mask); f,norm=action_metrics(operator,state,target); loss=(1-f).mean()
 if variant=="B2": loss=loss+VARIANTS[variant]["lambda_unitary"]*soft_unitarity_loss(operator)
 if variant in ("B4","B5"):
  if exact_u is None: raise ValueError("privileged operator target required")
  loss=loss+VARIANTS[variant]["lambda_process"]*(1-process_fidelity(operator,exact_u)).mean()
 if variant=="B5":
  if composition is None: raise ValueError("independent composition predictions required")
  direct,second,first,_,_,_=composition
  loss=loss+VARIANTS[variant]["lambda_comp"]*b5_composition_loss(direct,second,first)
 return loss,f,norm,operator


def _exact_for_indices(indices:np.ndarray,device,data:ArmData|None=None)->torch.Tensor:
 data=data or ArmData("A4"); circuits=[_decode(data.gates[int(i)],data.qubits[int(i)],data.parameters[int(i)],data.masks[int(i)]) for i in indices]
 return torch.as_tensor(np.asarray([circuit_unitary(c) for c in circuits],np.complex64)).to(device)


def ensure_operator_targets(data:ArmData)->np.ndarray:
 path=OP_ROOT/"data/a4_unitaries.npy"; done=OP_ROOT/"data/a4_unitaries.complete.json"
 if done.exists() and path.exists(): return np.load(path,mmap_mode="r")
 tmp=path.with_suffix(".tmp"); values=np.lib.format.open_memmap(tmp,mode="w+",dtype=np.complex64,shape=(ALLOCATION["unique_circuits"],DIM,DIM))
 for start in range(0,len(values),1024):
  circuits=[_decode(data.gates[i],data.qubits[i],data.parameters[i],data.masks[i]) for i in range(start,min(start+1024,len(values)))]
  values[start:start+len(circuits)]=np.asarray([circuit_unitary(c) for c in circuits],np.complex64); values.flush()
 del values; os.replace(tmp,path); atomic_json(done,{"schema_version":SCHEMA,"count":ALLOCATION["unique_circuits"],"dtype":"complex64","source_manifest_hash":_sha(DATA_ROOT/"A4/manifest.json")}); return np.load(path,mmap_mode="r")


def operator_smoke()->dict[str,Any]:
 require_operator_preconditions(); pre=operator_preflight()
 device=ACCEL_DEVICE; batch=_tiny_batch(device); results={}
 for variant in VARIANTS:
  if variant=="B3" and pre["status"]!="PASS": results[variant]={"status":pre["status"],"reason":"B3 Cayley native-accelerator or numerical preflight failed"}; continue
  torch.manual_seed(SEED); model=_model(variant,device); opt=torch.optim.AdamW(model.parameters(),3e-4); before=next(model.parameters()).detach().clone(); exact=_exact_for_indices(np.arange(4),device,data=ArmData("A4")) if variant in ("B4","B5") else None; composition=None
  if variant=="B5":
   data=ArmData("A4"); first=[_decode(data.gates[i],data.qubits[i],data.parameters[i],data.masks[i]) for i in range(4)]; second=[_decode(data.gates[i+4],data.qubits[i+4],data.parameters[i+4],data.masks[i+4]) for i in range(4)]; direct=model(*_circuit_tensors([a+b for a,b in zip(first,second)],device)); u2=model(*_circuit_tensors(second,device)); u1=model(*_circuit_tensors(first,device)); exact1=_exact_for_indices(np.arange(4),device,data); exact2=_exact_for_indices(np.arange(4,8),device,data); composition=(direct,u2,u1,exact2@exact1,exact2,exact1)
  loss,f,norm,operator=_loss(variant,model,batch,exact,composition); opt.zero_grad(); loss.backward(); finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()); opt.step(); accel_synchronize(); config=variant_config(variant); results[variant]={"scientific_run":False,"purpose":"implementation_validation","variant":variant,"maximum_updates":1,"config_hash":digest(config),"status":"PASS" if finite and torch.isfinite(loss) and not torch.equal(before,next(model.parameters()).detach()) else "FAIL","loss":float(loss.detach().cpu()),"action_fidelity":float(f.mean().detach().cpu()),"raw_action_norm":float(norm.mean().detach().cpu()),"unitarity_error":float(raw_unitarity_error(operator).mean().detach().cpu()) if operator is not None else None,"finite_gradients":finite,"xpu_residency":next(model.parameters()).device.type==ACCEL,"optimizer_update":not torch.equal(before,next(model.parameters()).detach())}
 config=variant_config("B3"); dataset_hash=_sha(DATA_ROOT/"A4/manifest.json"); buffer=io.BytesIO(); torch.save({"config":config,"config_hash":digest(config),"dataset_manifest_hash":dataset_hash},buffer); buffer.seek(0); validate_operator_checkpoint(torch.load(buffer,weights_only=False),config,dataset_hash)
 result={"schema_version":SCHEMA,"scientific_run":False,"purpose":"implementation_validation","variant":"B0-B5","maximum_updates":1,"config_hash":digest({v:variant_config(v) for v in VARIANTS}),"status":"PASS" if all(x["status"]=="PASS" for x in results.values()) else results.get("B3",{}).get("status","FAIL"),"workload":"one optimizer update, four A4 pairs per variant; correctness only","scientific_runs":"NONE","save_resume":"PASS","metric_pipeline":"PASS","variants":results}; atomic_json(OP_ROOT/"smoke/B0-B5.json",result); return result


def _validation_circuits(split:str)->list[list[Gate]]:
 if split not in VALIDATION_SPLITS: raise PermissionError(split)
 rows=json.loads((ROOT/f"datasets/{split}.json").read_text()); return [[Gate.from_dict(g) for g in row["gates"]] for row in rows]


def evaluate_variant(variant:str,model,device)->dict[str,Any]:
 out={}; model.eval()
 with torch.inference_mode():
  for split in VALIDATION_SPLITS:
   batch=[torch.as_tensor(v).to(device) for v in load_validation(split)]; circuits=_validation_circuits(split); exact=torch.as_tensor(np.asarray([circuit_unitary(c) for c in circuits],np.complex64)).to(device)
   exact_state=apply_operator(exact,batch[4])
   if variant=="B0":
    direct=model(*batch[:5]); out[split]={"direct_state_fidelity":float(state_fidelity(direct,batch[5]).mean().cpu()),"predicted_operator_state_fidelity":None,"exact_operator_state_fidelity":float(state_fidelity(exact_state,batch[5]).mean().cpu()),"process_fidelity":None,"raw_action_norm":None,"phase_aligned_frobenius_error":None,"raw_unitarity_error":None}
   else:
    operator=model(*batch[:4]); action,norm=operator_action(operator,batch[4]); out[split]={"direct_state_fidelity":None,"predicted_operator_state_fidelity":float(state_fidelity(action,batch[5]).mean().cpu()),"exact_operator_state_fidelity":float(state_fidelity(exact_state,batch[5]).mean().cpu()),"process_fidelity":float(process_fidelity(operator,exact).mean().cpu()),"raw_action_norm":float(norm.mean().cpu()),"phase_aligned_frobenius_error":float(phase_aligned_normalized_frobenius_error(operator,exact).mean().cpu()),"raw_unitarity_error":float(raw_unitarity_error(operator).mean().cpu())}
 model.train(); return out


def validate_operator_checkpoint(payload:dict[str,Any],config:dict[str,Any],dataset_hash:str)->None:
 if payload.get("config_hash")!=digest(config) or payload.get("config")!=config: raise ValueError("resume refused: variant/config differs")
 if payload.get("dataset_manifest_hash")!=dataset_hash: raise ValueError("resume refused: dataset manifest differs")


def _save_checkpoint(path,model,opt,sched,config,step,samples,rng,best,best_step):
 payload={"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"step":step,"samples_seen":samples,"numpy_rng":rng.bit_generator.state,"torch_rng":torch.get_rng_state(),"config":config,"config_hash":digest(config),"dataset_manifest_hash":config["dataset_manifest_hash"],"best_balanced_validation":best,"best_checkpoint_step":best_step}; path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(".tmp"); torch.save(payload,tmp); os.replace(tmp,path)


def _composition_predictions(model,data:ArmData,unitaries:np.ndarray,indices:np.ndarray,device):
 second_indices=(indices+1)%ALLOCATION["unique_circuits"]
 first=[_decode(data.gates[int(i)],data.qubits[int(i)],data.parameters[int(i)],data.masks[int(i)]) for i in indices]; second=[_decode(data.gates[int(i)],data.qubits[int(i)],data.parameters[int(i)],data.masks[int(i)]) for i in second_indices]
 direct=model(*_circuit_tensors([a+b for a,b in zip(first,second)],device)); pred2=model(*_circuit_tensors(second,device)); pred1=model(*_circuit_tensors(first,device)); exact1=torch.as_tensor(np.asarray(unitaries[indices])).to(device); exact2=torch.as_tensor(np.asarray(unitaries[second_indices])).to(device)
 return direct,pred2,pred1,exact2@exact1,exact2,exact1


def _status(**row):
 value={"schema_version":SCHEMA,"timestamp":time.time(),**row}; atomic_json(OP_ROOT/"status.json",value)
 with (OP_ROOT/"progress.jsonl").open("a") as f: f.write(json.dumps(value,sort_keys=True,separators=(",",":"))+"\n")
 print(" ".join(f"{k}={v}" for k,v in row.items()),flush=True)


def operator_run(variant:str)->dict[str,Any]:
 """Manual scientific entry point. Never called by preflight/smoke."""
 require_operator_preconditions()
 if variant=="B3" and operator_preflight()["status"]!="PASS": raise RuntimeError("B3-XPU-BLOCKED")
 metric_path=OP_ROOT/f"metrics/{variant}.json"
 if metric_path.exists():
  existing=json.loads(metric_path.read_text())
  if existing.get("state")=="COMPLETED": return existing
 config=variant_config(variant); config["dataset_manifest_hash"]=_sha(DATA_ROOT/"A4/manifest.json"); atomic_json(OP_ROOT/f"configs/{variant}.json",config)
 device=ACCEL_DEVICE; torch.manual_seed(SEED); data=ArmData("A4"); unitary_targets=ensure_operator_targets(data) if variant in ("B4","B5") else None; model=_model(variant,device); opt=torch.optim.AdamW(model.parameters(),3e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,RECIPE["maximum_updates"]); rng=np.random.default_rng(SEED); checkpoint=OP_ROOT/f"checkpoints/{variant}-latest.pt"; primary=OP_ROOT/f"checkpoints/{variant}-best-balanced.pt"; step=samples=0; best=-math.inf; best_step=None
 if checkpoint.exists():
  p=torch.load(checkpoint,map_location="cpu",weights_only=False)
  validate_operator_checkpoint(p,config,config["dataset_manifest_hash"])
  model.load_state_dict(p["model"]); opt.load_state_dict(p["optimizer"]); sched.load_state_dict(p["scheduler"]); step=p["step"]; samples=p["samples_seen"]; rng.bit_generator.state=p["numpy_rng"]; torch.set_rng_state(p["torch_rng"]); best=p["best_balanced_validation"]; best_step=p["best_checkpoint_step"]
 started=time.monotonic(); latest={}; loss=torch.tensor(float("nan"),device=device); f=torch.tensor([float("nan")],device=device); norm=torch.tensor([float("nan")],device=device); op=None
 while step<RECIPE["maximum_updates"] and not _STOP:
  indices=rng.integers(data.length,size=RECIPE["effective_batch_size"]); batch=[torch.as_tensor(v).to(device) for v in data.batch(indices)]; exact=torch.as_tensor(np.asarray(unitary_targets[indices//17])).to(device) if unitary_targets is not None else None; composition=None
  if variant=="B5": composition=_composition_predictions(model,data,unitary_targets,rng.integers(58_823,size=VARIANTS[variant]["composition_batch_size"]),device)
  loss,f,norm,op=_loss(variant,model,batch,exact,composition); opt.zero_grad(set_to_none=True); loss.backward()
  if not torch.isfinite(loss) or not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()): raise FloatingPointError("non-finite operator training")
  opt.step(); sched.step(); step+=1; samples+=len(indices)
  if step%RECIPE["validation_interval"]==0 or step==RECIPE["maximum_updates"]:
   accel_synchronize(); latest=evaluate_variant(variant,model,device); score=sum(latest[s]["predicted_operator_state_fidelity" if variant!="B0" else "direct_state_fidelity"] for s in ("iid_validation","composition_ood_validation","depth_ood_validation"))/3
   improved=score>best
   if improved: best,best_step=score,step
   _save_checkpoint(checkpoint,model,opt,sched,config,step,samples,rng,best,best_step)
   if improved: _save_checkpoint(primary,model,opt,sched,config,step,samples,rng,best,best_step)
  if step%50==0 or step%RECIPE["validation_interval"]==0:
   elapsed=time.monotonic()-started; iid=latest.get("iid_validation",{}); comp=latest.get("composition_ood_validation",{}); depth=latest.get("depth_ood_validation",{}); key="predicted_operator_state_fidelity" if variant!="B0" else "direct_state_fidelity"
   _status(variant=variant,step=f"{step}/{RECIPE['maximum_updates']}",loss=float(loss.detach().cpu()),action_fidelity=float(f.mean().detach().cpu()),process_fidelity=float(process_fidelity(op,exact).mean().detach().cpu()) if op is not None and exact is not None else None,unitarity_error=float(raw_unitarity_error(op).mean().detach().cpu()) if op is not None else None,IID_val=iid.get(key),State_OOD_val=latest.get("state_ood_validation",{}).get(key),Parameter_OOD_val=latest.get("parameter_ood_validation",{}).get(key),Composition_OOD_val=comp.get(key),Depth_OOD_val=depth.get(key),lr=opt.param_groups[0]["lr"],samples_per_second=(step*RECIPE["effective_batch_size"])/max(elapsed,1e-9),elapsed=elapsed,ETA=(RECIPE["maximum_updates"]-step)*elapsed/max(step,1),device=str(ACCEL_DEVICE),checkpoint=str(checkpoint),state="RUNNING")
 _save_checkpoint(checkpoint,model,opt,sched,config,step,samples,rng,best,best_step)
 state="INTERRUPTED" if _STOP else "COMPLETED"; result={"schema_version":SCHEMA,"variant":variant,"state":state,"step":step,"samples_seen":samples,"best_balanced_validation":best,"best_checkpoint_step":best_step,"latest_validation":latest,"scientific":True}; atomic_json(metric_path,result); return result


def _protocol_checks()->dict[str,bool]:
 configs={v:variant_config(v) for v in VARIANTS}; counts={v:c["actual_parameters"] for v,c in configs.items()}
 b4={k:v for k,v in configs["B4"].items() if k not in {"variant","lambda_comp","composition_batch_size","composition_contract","config_hash"}}
 b5={k:v for k,v in configs["B5"].items() if k not in {"variant","lambda_comp","composition_batch_size","composition_contract","config_hash"}}
 return {"config_validity":all(c["parameter_budget_valid"] for c in configs.values()),"parameter_fairness":counts["B0"]==1_001_472 and all(counts[v]==1_073_312 for v in ("B1","B2","B3","B4","B5")),"supervision_integrity":all(configs[v]["supervision"]==["C","psi_in","psi_out"] for v in ("B0","B1","B2","B3")),"B3_action_only":not configs["B3"]["privileged"] and "exact_U" not in configs["B3"]["supervision"],"B4_B5_protocol_equality":b4==b5,"B5_composition_contract":configs["B5"]["lambda_comp"]==.3 and "U(C2)U(C1)" in configs["B5"]["composition_contract"],"sealed_access_zero":json.loads((ROOT/"test_access_log.json").read_text())["access_count"]==0}


def operator_screen()->dict[str,Any]:
 require_operator_preconditions(); checks=_protocol_checks(); pre=operator_preflight()
 if not checks["B4_B5_protocol_equality"]: raise RuntimeError("B4-B5-PROTOCOL-MISMATCH: operator-screen refused before any scientific variant")
 if pre["status"]!="PASS" or not all(checks.values()): raise RuntimeError("B3-XPU-BLOCKED: operator-screen refused before any scientific variant")
 print("TRACK-B-READY",flush=True); results={}
 for variant in VARIANTS:
  result=operator_run(variant); results[variant]=result
  if result["state"]!="COMPLETED": break
 return results


def operator_status()->dict[str,Any]:
 path=OP_ROOT/"status.json"; latest=json.loads(path.read_text()) if path.exists() else {"schema_version":SCHEMA,"state":"IMPLEMENTED_NOT_SCIENTIFICALLY_RUN","variant":None,"step":None,"device":str(ACCEL_DEVICE)}
 latest["scientific_runs"]={v:(json.loads((OP_ROOT/f"metrics/{v}.json").read_text())["state"] if (OP_ROOT/f"metrics/{v}.json").exists() else "NOT_RUN") for v in VARIANTS}; latest["sealed_test_access_count"]=json.loads((ROOT/"test_access_log.json").read_text())["access_count"]; return latest


def install_signal_handlers()->None:
 def stop(_signum,_frame):
  global _STOP; _STOP=True
 signal.signal(signal.SIGINT,stop); signal.signal(signal.SIGTERM,stop)
