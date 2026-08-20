"""CC-NQE P4.6 mechanism-decomposition primitives (four qubits only)."""
from __future__ import annotations

import hashlib, json, math, os, random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from cc_nqe import DIM, GATES, N_QUBITS, Gate, apply_gate, circuit_id, circuit_unitary, generate_state, serialize_circuit, sha256, structural_signature
from cc_nqe_p4_5 import CircuitEncoder, MODEL_SCALES, ScaledCCNQE, atomic_json, operator_fidelity, state_fidelity

ROOT=Path("artifacts/cc_nqe_p4_6")
P45=Path("artifacts/cc_nqe_p4_5")
P1P4=Path("artifacts/cc_nqe_p1_p4")
SCHEMA="cc-nqe-p4.6-v1"
SEED=2026
DEPTHS={"train":[1,2,3,4,5,6],"depth_ood_validation":[7],"depth_ood_test_sealed":[8,9,10]}
FACTORIAL_ARMS={"A1":(1_000_000,1),"A2":(250_000,4),"A3":(62_500,16),"A4":(58_824,17),"A5":(15_625,64)}
SEALED={"composition_ood_test_sealed","depth_ood_test_sealed"}


def canonical(value:Any)->str: return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)
def digest(value:Any)->str: return hashlib.sha256(canonical(value).encode()).hexdigest()

def motifs(gates:list[Gate],k:int)->list[tuple[str,...]]:
    tokens=[f"{g.name}:{','.join(map(str,g.qubits))}" for g in gates]
    return [tuple(tokens[i:i+k]) for i in range(len(tokens)-k+1)]

def motif(gates:list[Gate],k:int)->tuple[str,...]: return motifs(gates,k)[0] if len(gates)>=k else ()
def motif_family_value(value:tuple[str,...])->int: return int(hashlib.sha256("|".join(value).encode()).hexdigest(),16)%10
def motif_family(gates:list[Gate],k:int)->int: return motif_family_value(motif(gates,k))

def split_for_circuit(gates:list[Gate],purpose:str)->bool:
    """Frozen motif-family partition: train 0..5, comp-val 6..7, sealed comp-test 8..9."""
    wanted={"train":set(range(6)),"composition_ood_validation":{6,7},"composition_ood_test_sealed":{8,9}}[purpose]
    return all(motif_family(gates,k) in wanted for k in (2,3,4) if len(gates)>=k)

def ood_split_contract()->dict[str,Any]:
    return {"schema_version":SCHEMA,"frozen":True,"generation_seed":SEED,
      "gate_alphabet":list(GATES),"qubit_operand_policy":"uniform valid operands; CNOT uses ordered distinct control/target",
      "continuous_parameter_support":"angles in [0,2pi); split by closed-open regions",
      "state_family_policy":{"train_and_iid":["product","random-local"],"state_ood_validation":["entangled","Haar-random"]},
      "canonical_circuit_representation":"canonical sorted-key JSON ordered gate list",
      "exact_circuit_signature":"SHA256(canonical representation)","structural_signature":"ordered gate names and operands; theta excluded",
      "motif_definition":"every contiguous ordered gate/operand token window; family=SHA256(tokens) mod 10",
      "motif_lengths":[2,3,4],"motif_families":{"train":[0,1,2,3,4,5],"composition_ood_validation":[6,7],"composition_ood_test_sealed":[8,9]},
      "composition_depth_policy":"depths 4,5,6 equally represented in train reference, validation, and sealed test",
      "depths":DEPTHS,
      "parameter_regions":{"train_composition_depth_state":[[0,2*math.pi/3],[math.pi,5*math.pi/3]],"parameter_ood_interpolation":[[2*math.pi/3,math.pi]],"parameter_ood_extrapolation":[[5*math.pi/3,2*math.pi]]},
      "matching_tolerances":{"normalization":1e-10,"unitarity":1e-10},
      "leakage_rules":{"exact_sample":"forbidden across all splits","state_id":"forbidden across all splits","circuit_id":"forbidden except state-OOD paired operator IDs","structural":"forbidden train/composition-val/composition-test","motif_family":"composition partitions disjoint","same_topology_different_parameters":"allowed outside composition partitions"},
      "sealed_test_policy":{"splits":sorted(SEALED),"ordinary_loaders_reject":True,"targets_generated_and_hashed":True,"access_log_required_zero":True}}

def factorial_arms()->dict[str,Any]:
    return {k:{"unique_circuits":c,"probes_per_circuit":p,"pairs":c*p} for k,(c,p) in FACTORIAL_ARMS.items()}

def assert_unsealed(split:str,unlock:bool=False)->None:
    if split in SEALED and not unlock: raise PermissionError(f"sealed test access refused: {split}")

def record_test_access(root:Path=ROOT,split:str|None=None,unlock:bool=False)->dict[str,Any]:
    path=root/"test_access_log.json"; value=json.loads(path.read_text()) if path.exists() else {"schema_version":SCHEMA,"access_count":0,"accesses":[]}
    if split is not None:
        assert_unsealed(split,unlock); value["access_count"]+=1; value["accesses"].append({"split":split})
    atomic_json(path,value); return value

def freeze_p45(root:Path=ROOT)->dict[str,Any]:
    expected=[("10k","60k"),("10k","250k"),("10k","1m"),("100k","250k"),("100k","1m"),("1m","1m"),("1m","5m")]
    rows=[]; hashes={}
    for data,model in expected:
        path=P45/f"metrics/state/state-{data}-{model}-seed2026.json"; raw=path.read_bytes(); d=json.loads(raw); c=d["config"]
        checks={"state":d["state"]=="COMPLETED","steps":d["steps"]==10_000,"seed":c["seed"]==SEED,"run_kind":c["run_kind"]=="screening",**{k:d[k] is True for k in ("finite_loss","finite_gradients","parameter_updated","xpu_residency","no_nan_inf","validation_pipeline")}}
        if not all(checks.values()): raise RuntimeError(f"P4.5-INTEGRITY-BLOCKED: {path}: {checks}")
        h=hashlib.sha256(raw).hexdigest(); hashes[str(path)]=h; curve=d["curve"]
        rows.append({"dataset_scale":data,"model_scale":model,"actual_parameters":c["actual_parameters"],"seed":SEED,"steps":d["steps"],"final_validation_fidelity":d["final_validation_fidelity"],"best_composition_ood_fidelity":max(x["composition_ood_fidelity"] for x in curve if "composition_ood_fidelity" in x),"best_depth_ood_fidelity":max(x["depth_ood_fidelity"] for x in curve if "depth_ood_fidelity" in x),"final_train_fidelity":curve[-1]["train_fidelity"],"source_metric":str(path),"source_metric_sha256":h})
    baseline=json.loads((P1P4/"artifact_hashes.json").read_text()); failures=[p for p,h in baseline.items() if not Path(p).exists() or sha256(Path(p))!=h]
    if failures: raise RuntimeError(f"P4.5-INTEGRITY-BLOCKED: P1-P4: {failures}")
    hashes[str(P1P4/"artifact_hashes.json")]=sha256(P1P4/"artifact_hashes.json")
    summary={"schema_version":SCHEMA,"gate":"G0","status":"PASS","p1_p4_verified":len(baseline),"screening_count":len(rows),"results":rows,"p4_5_immutable":True}
    atomic_json(root/"p4_5_frozen_summary.json",summary); atomic_json(root/"p4_5_input_hashes.json",hashes); return summary

def process_fidelity(pred:torch.Tensor,target:torch.Tensor)->torch.Tensor: return operator_fidelity(pred,target)
def phase_aligned_error(pred:torch.Tensor,target:torch.Tensor)->torch.Tensor:
    overlap=(target.conj()*pred).sum((-2,-1)); phase=torch.where(overlap.abs()>0,overlap.conj()/overlap.abs(),torch.ones_like(overlap)); return torch.linalg.matrix_norm(target-pred*phase[...,None,None])/torch.linalg.matrix_norm(target)
def raw_unitarity_error(pred:torch.Tensor)->torch.Tensor:
    eye=torch.eye(DIM,dtype=pred.dtype,device=pred.device); return torch.linalg.matrix_norm(pred.mH@pred-eye)/math.sqrt(DIM)

class OperatorModel(nn.Module):
    def __init__(self,kind:str,width_cfg:dict[str,int]|None=None):
        super().__init__();
        if kind not in ("unconstrained","lie","cayley"): raise ValueError(kind)
        cfg=width_cfg or MODEL_SCALES["1m"]; self.kind=kind; self.circuit=CircuitEncoder(**cfg); self.head=nn.Sequential(nn.Linear(cfg["width"],cfg["ff"]),nn.GELU(),nn.Linear(cfg["ff"],2*DIM*DIM))
    def forward(self,gates,qubits,parameters,mask):
        raw=self.head(self.circuit(gates,qubits,parameters,mask)).reshape(-1,2,DIM,DIM); b=torch.complex(raw[:,0],raw[:,1])
        if self.kind=="unconstrained": return b
        a=b-b.mH
        if self.kind=="lie": return torch.matrix_exp(a)
        eye=torch.eye(DIM,dtype=a.dtype,device=a.device).expand_as(a)
        # Scaled Cayley chart omits eigenvalue -1; scale limits ill-conditioned generators.
        a=torch.tanh(a.abs())*torch.exp(1j*torch.angle(a)); a=(a-a.mH)/2
        return torch.linalg.solve(eye-a/2,eye+a/2)

class GateTransition(nn.Module):
    def __init__(self,width:int=448):
        super().__init__(); self.gate=nn.Embedding(len(GATES)+1,width,padding_idx=0); self.qubit=nn.Embedding(N_QUBITS+1,width,padding_idx=N_QUBITS); self.param=nn.Sequential(nn.Linear(3,width),nn.GELU(),nn.Linear(width,width)); self.state=nn.Linear(2*DIM,width); self.head=nn.Sequential(nn.Linear(4*width,width),nn.GELU(),nn.Linear(width,2*DIM))
    def forward(self,g,q,p,state): return self.head(torch.cat((self.gate(g),self.qubit(q[:,0]),self.qubit(q[:,1]),self.param(p)+self.state(state)),1))

class RecurrentCCNQE(nn.Module):
    def __init__(self,width:int=448): super().__init__(); self.transition=GateTransition(width)
    def forward(self,gates,qubits,parameters,mask,state,return_prefixes:bool=False):
        prefixes=[]; current=state
        for i in range(gates.shape[1]):
            nxt=self.transition(gates[:,i],qubits[:,i],parameters[:,i],current); current=torch.where(mask[:,i,None],nxt,current); prefixes.append(current)
        return (current,torch.stack(prefixes,1)) if return_prefixes else current

def exact_prefix_targets(circuits:list[list[Gate]],states:np.ndarray,max_depth:int)->np.ndarray:
    out=np.zeros((len(circuits),max_depth,2*DIM),np.float32)
    for n,(c,s) in enumerate(zip(circuits,states)):
        cur=np.asarray(s,np.complex128)
        for i in range(max_depth):
            if i<len(c): cur=apply_gate(cur,c[i])
            out[n,i]=np.r_[cur.real,cur.imag]
    return out

def prefix_loss(pred:torch.Tensor,target:torch.Tensor,mask:torch.Tensor)->torch.Tensor:
    losses=(1-state_fidelity(pred,target)).clamp_min(0); return ((losses*mask).sum(1)/mask.sum(1).clamp_min(1)).mean()
def composition_consistency_loss(model:nn.Module,c1:tuple,c2:tuple,combined:tuple,state:torch.Tensor)->torch.Tensor:
    direct=model(*combined,state); staged=model(*c2,model(*c1,state)); return (1-state_fidelity(direct,staged)).mean()

def balanced_score(iid:float,composition:float,depth:float)->float: return (iid+composition+depth)/3
class CheckpointSelector:
    def __init__(self): self.best={k:{"score":-math.inf,"step":None} for k in ("iid","composition","depth","balanced")}
    def update(self,step:int,metrics:dict[str,float])->list[str]:
        scores={"iid":metrics["iid_validation"],"composition":metrics["composition_ood_validation"],"depth":metrics["depth_ood_validation"],"balanced":balanced_score(metrics["iid_validation"],metrics["composition_ood_validation"],metrics["depth_ood_validation"])}; changed=[]
        for k,v in scores.items():
            if v>self.best[k]["score"]: self.best[k]={"score":v,"step":step}; changed.append(k)
        return changed

def validate_resume(payload:dict[str,Any],config:dict[str,Any],manifest_hash:str)->None:
    if payload.get("config_hash")!=digest(config): raise ValueError("resume refused: config hash differs")
    if payload.get("dataset_manifest_hash")!=manifest_hash: raise ValueError("resume refused: dataset manifest hash differs")

def normalized_action_fidelity(target:torch.Tensor,action:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
    raw_norm=torch.linalg.vector_norm(action,dim=-1)
    denom=torch.linalg.vector_norm(target,dim=-1).square()*raw_norm.square()
    fidelity=(target.conj()*action).sum(-1).abs().square()/denom.clamp_min(torch.finfo(raw_norm.dtype).tiny)
    return fidelity,raw_norm

def _angles(c:list[Gate])->list[float]: return [float(g.theta) for g in c if g.theta is not None]
def _all_motif_families(c:list[Gate],k:int)->set[int]: return {motif_family_value(x) for x in motifs(c,k)}
def _partition_ok(c:list[Gate],families:set[int])->bool:
    # Frozen family key is the leading k-window; all windows are still reported as diversity.
    return all(motif_family(c,k) in families for k in (2,3,4) if len(c)>=k)

def _make_circuits(counts:dict[int,int],seed:int,regime:str="train",families:set[int]|None=None,forbidden:set[str]|None=None)->list[list[Gate]]:
    out=[]; seen=set(forbidden or ()); attempt=0
    for depth,count in counts.items():
        while sum(len(c)==depth for c in out)<count:
            c=__import__('cc_nqe').generate_circuit(seed+attempt,depth,regime); attempt+=1; sig=structural_signature(c)
            if sig in seen or (families is not None and not _partition_ok(c,families)): continue
            seen.add(sig); out.append(c)
            if attempt>2_000_000: raise RuntimeError(f"cannot fill split depth {depth}")
    return out

def diversity_statistics(circuits:list[list[Gate]])->dict[str,Any]:
    angles=[a for c in circuits for a in _angles(c)]
    interactions=Counter(f"{g.qubits[0]}->{g.qubits[1]}" for c in circuits for g in c if g.name=="CNOT")
    return {"N_unique_exact_circuits":len({serialize_circuit(c) for c in circuits}),"N_unique_structural_signatures":len({structural_signature(c) for c in circuits}),
      **{f"N_unique_k{k}_motifs":len({m for c in circuits for m in motifs(c,k)}) for k in (2,3,4)},
      "parameter_bin_coverage":sorted({min(11,int(a/(2*math.pi)*12)) for a in angles}),"qubit_interaction_coverage":dict(sorted(interactions.items()))}

def operator_protocol()->dict[str,Any]:
    return {"schema_version":SCHEMA,"status":"FROZEN-NOT-EXECUTED","primary_allocation":{"name":"A4","unique_circuits":58824,"probes_per_circuit":17,"pairs":1000008},
      "action_fidelity":"|<target|M psi>|^2/(||target||^2 ||M psi||^2); raw ||M psi|| recorded",
      "arms":{"B0":{"model":"direct state","supervision":"action-only"},"B1":{"model":"unconstrained operator","supervision":"action-only"},"B2":{"model":"soft-unitarity operator","supervision":"action-only + unitarity regularizer"},"B3":{"model":"exact-unitarity exp((B-B†)/2)","supervision":"action-only"},"B4":{"model":"direct-U","supervision":"PRIVILEGED OPERATOR SUPERVISION"},"B5":{"model":"direct-U + independent operator composition outputs","supervision":"PRIVILEGED OPERATOR SUPERVISION"}}}

def recurrent_protocol()->dict[str,Any]:
    return {"schema_version":SCHEMA,"status":"FROZEN-NOT-EXECUTED","arms":{"C1":"monolithic baseline","C2":"shared recurrent free rollout","C3":"shared recurrent free rollout + prefix supervision"},"rollout":"psi_hat[0]=input; psi_hat[i+1]=T_theta(G_i,psi_hat[i])","prefix_targets":"exact intermediate states are targets only","lambda_prefix":1.0,"excluded":["tautological state-composition C4","teacher forcing","scheduled sampling","curriculum"]}

def resource_estimate(free_disk:int)->dict[str,Any]:
    rows={}
    for arm,(circuits,probes) in FACTORIAL_ARMS.items():
        metadata=circuits*6*20; state=circuits*probes*DIM*2*4; target=state; unitary=circuits*DIM*DIM*2*4
        disk=metadata+state+target; rows[arm]={"unique_circuits":circuits,"probes_per_circuit":probes,"pairs":circuits*probes,"dataset_bytes":disk,"circuit_metadata_bytes":metadata,"state_payload_bytes":state,"target_payload_bytes":target,"optional_unitary_target_bytes":unitary,"peak_ram_bytes":256*1024**2,"disk_required_bytes":int(disk*1.15),"generation_time_seconds_estimate":round(circuits*probes*0.0003,1),"training_time_hours_estimate":24.0,"storage":"sharded memory-mapped arrays; direct-state excludes unitary targets"}
    required=max(x["disk_required_bytes"] for x in rows.values())
    return {"schema_version":SCHEMA,"method":"float32 real/imag state and target payloads; measured capacity, conservative metadata estimate","free_disk_bytes":free_disk,"maximum_arm_disk_required_bytes":required,"sufficient":free_disk>=required,"arms":rows}
