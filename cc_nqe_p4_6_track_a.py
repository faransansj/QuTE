"""P4.6-B Track-A factorial screen (validation only; sealed splits are never loaded)."""
from __future__ import annotations

import hashlib, json, math, os, shutil, signal, time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cc_nqe import ACCEL_DEVICE, DIM, GATES, GATE_TO_ID, N_QUBITS, Gate, accel_synchronize, generate_circuit, generate_state, serialize_circuit, simulate
from cc_nqe_p4_5 import MODEL_SCALES, ScaledCCNQE, atomic_json, parameter_count, state_fidelity, _tensorize_circuit
from cc_nqe_p4_6 import FACTORIAL_ARMS, ROOT, SCHEMA, SEED, balanced_score, digest, motif_family, motifs, resource_estimate

TRACK_ROOT = ROOT / "track_a"
DATA_ROOT = TRACK_ROOT / "datasets"
MASTER_COUNT = 1_000_000
SHARD_SIZE = 50_000
MAX_UPDATES = 10_000
BATCH_SIZE = 1024
VALIDATION_INTERVAL = 500
VALIDATION_SPLITS = ("iid_validation", "state_ood_validation", "parameter_ood_validation", "composition_ood_validation", "depth_ood_validation")
FROZEN_HASHES = {
    "ood_split_contract.json": "591fa9fcf9ab09f55a7b857b07be9e0a767424a31847e90cca8eafc057c02b20",
    "audit.json": "81089e5fab2887e7e210232c1449266928175b94ec3f156b9147b32939a27fe1",
}
RECIPE = {"model": "direct state", "model_scale": "1m", "seed": SEED, "dtype": "float32", "device": str(ACCEL_DEVICE), "optimizer": "AdamW", "learning_rate": 3e-4, "scheduler": "cosine", "effective_batch_size": BATCH_SIZE, "optimizer_updates": MAX_UPDATES, "validation_interval": VALIDATION_INTERVAL}
VERDICT_RULE = "Pre-specified heuristic (not a statistical significance claim): balanced-score range < 0.01 => NO-CLEAR-DATA-EFFECT; otherwise A1/A2 winner => CIRCUIT-COVERAGE-DOMINANT, A3 => MIXED-DATA-EFFECT, A4/A5 => PROBE-COVERAGE-DOMINANT."
_STOP = False


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def require_track_a_preconditions() -> None:
    final = json.loads((ROOT / "final_gate.json").read_text())
    if not (final.get("G0") == "PASS" and final.get("G1") == "PASS" and final.get("G2") == "G2-PASS" and final.get("synthetic") == "COMPOSITION-SANITY-PASS"):
        raise RuntimeError(f"P4.6-B-BLOCKED: gates differ: {final}")
    for name, expected in FROZEN_HASHES.items():
        path = ROOT / "datasets" / name
        if _sha(path) != expected: raise RuntimeError(f"P4.6-B-BLOCKED: frozen {name} differs")
    access = json.loads((ROOT / "test_access_log.json").read_text())
    if access.get("access_count") != 0: raise RuntimeError("P4.6-B-BLOCKED: sealed-test access is nonzero")
    estimate = resource_estimate(shutil.disk_usage(".").free)
    master_bytes=MASTER_COUNT*(16+16*2+16*3*4+16+32)
    pair_bytes=sum(c*p for c,p in FACTORIAL_ARMS.values())*(2*DIM*2*4+32)
    estimate.update(track_a_master_bytes=master_bytes,track_a_all_arm_pair_bytes=pair_bytes,track_a_cumulative_required_bytes=int((master_bytes+pair_bytes)*1.15))
    estimate["sufficient"]=estimate["free_disk_bytes"]>=estimate["track_a_cumulative_required_bytes"]
    atomic_json(TRACK_ROOT / "resource_recheck.json", estimate)
    if not estimate["sufficient"]: raise RuntimeError("P4.6-B-BLOCKED: insufficient free disk")


def _open(path: Path, shape: tuple[int, ...], dtype: Any, resume: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="r+" if resume and path.exists() else "w+", dtype=dtype, shape=shape)


def _status(phase: str, **values: Any) -> None:
    row = {"schema_version": SCHEMA, "timestamp": time.time(), "phase": phase, **values}
    atomic_json(TRACK_ROOT / "status.json", row)
    with (TRACK_ROOT / "progress.jsonl").open("a") as f: f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    print(" ".join(f"{k}={v}" for k, v in row.items() if k not in ("schema_version", "timestamp")), flush=True)


def _decode(gates: np.ndarray, qubits: np.ndarray, parameters: np.ndarray, mask: np.ndarray) -> list[Gate]:
    names = (None,) + GATES
    out = []
    for g, q, p, keep in zip(gates, qubits, parameters, mask):
        if not keep: break
        name = names[int(g)]; qs = (int(q[0]), int(q[1])) if name == "CNOT" else (int(q[0]),)
        theta = float(math.atan2(float(p[0]), float(p[1])) % (2 * math.pi)) if name in ("RX", "RY", "RZ") else None
        out.append(Gate(name, qs, theta))
    return out


def generate_master_pool(count: int = MASTER_COUNT) -> dict[str, Any]:
    """Create one exact-circuit-deduplicated pool; every arm is a prefix."""
    root = DATA_ROOT / "master"; root.mkdir(parents=True, exist_ok=True)
    progress_path = root / "generation_progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"next_index": 0, "attempt": 0}
    resume = progress["next_index"] > 0
    arrays = {
        "gates": _open(root / "gates.npy", (count, 16), np.uint8, resume),
        "qubits": _open(root / "qubits.npy", (count, 16, 2), np.uint8, resume),
        "parameters": _open(root / "parameters.npy", (count, 16, 3), np.float32, resume),
        "masks": _open(root / "masks.npy", (count, 16), np.bool_, resume),
        "exact_sha256": _open(root / "exact_sha256.npy", (count, 32), np.uint8, resume),
    }
    next_index, attempt = int(progress["next_index"]), int(progress["attempt"])
    seen = {bytes(x) for x in arrays["exact_sha256"][:next_index]}
    started = time.monotonic()
    while next_index < count:
        depth = next_index % 6 + 1
        circuit = generate_circuit(SEED + attempt, depth, "train"); attempt += 1
        if not all(motif_family(circuit, k) in set(range(6)) for k in (2, 3, 4) if depth >= k): continue
        raw = serialize_circuit(circuit).encode(); exact = hashlib.sha256(raw).digest()
        if exact in seen: continue
        seen.add(exact); g, q, p, m = _tensorize_circuit(circuit)
        arrays["gates"][next_index] = g; arrays["qubits"][next_index] = q; arrays["parameters"][next_index] = p; arrays["masks"][next_index] = m; arrays["exact_sha256"][next_index] = np.frombuffer(exact, np.uint8)
        next_index += 1
        if next_index % 10_000 == 0 or next_index == count:
            for value in arrays.values(): value.flush()
            atomic_json(progress_path, {"next_index": next_index, "attempt": attempt})
            _status("generation-master", circuits=next_index, total=count, attempts=attempt, circuits_per_second=next_index/max(time.monotonic()-started, 1e-9), resumable_command="uv run python run_p4_6.py factorial-screen")
    manifest = {"schema_version": SCHEMA, "state": "COMPLETED", "seed": SEED, "exact_circuit_count": count, "subset_policy": "A1-A5 are prefixes of this ordered pool", "depth_policy": "index mod 6 + 1", "motif_families": [0,1,2,3,4,5], "arrays": {k: str(root/f"{k}.npy") for k in arrays}}
    atomic_json(root / "manifest.json", manifest)
    return manifest


def probe_seed(exact_digest: bytes, probe_index: int) -> int:
    return int.from_bytes(hashlib.sha256(b"p4.6-probe-v1" + exact_digest + probe_index.to_bytes(2, "big")).digest()[:8], "big") % (2**32)


def probe_family(circuit_index: int, probe_index: int) -> str:
    return ("product", "random-local")[(circuit_index + probe_index) % 2]


def ordered_probes(exact_digest: bytes, circuit_index: int, count: int) -> list[np.ndarray]:
    return [generate_state(probe_seed(exact_digest, j), probe_family(circuit_index, j)) for j in range(count)]


def _projective_fingerprint(state: np.ndarray) -> np.bytes_:
    pivot = int(np.argmax(np.abs(state))); z = state[pivot]
    aligned = state * (z.conjugate()/abs(z) if abs(z) else 1)
    raw = np.round(np.r_[aligned.real, aligned.imag], 12).astype("<f8").tobytes()
    return np.bytes_(hashlib.sha256(raw).digest())


def generate_arm_pairs(arm: str, circuit_count: int, probes: int) -> dict[str, Any]:
    root = DATA_ROOT / arm; root.mkdir(parents=True, exist_ok=True); total = circuit_count * probes
    master = DATA_ROOT / "master"; gates=np.load(master/"gates.npy",mmap_mode="r"); qubits=np.load(master/"qubits.npy",mmap_mode="r"); parameters=np.load(master/"parameters.npy",mmap_mode="r"); masks=np.load(master/"masks.npy",mmap_mode="r"); ids=np.load(master/"exact_sha256.npy",mmap_mode="r")
    shards=[]; projective_sample=[]; family_counts=Counter(); started=time.monotonic()
    for offset in range(0,total,SHARD_SIZE):
        n=min(SHARD_SIZE,total-offset); stem=f"{offset//SHARD_SIZE:05d}"; state_path=root/f"states_{stem}.npy"; target_path=root/f"targets_{stem}.npy"; hash_path=root/f"probe_hashes_{stem}.npy"
        if not (state_path.exists() and target_path.exists() and hash_path.exists()):
            temp_paths=[x.with_name(x.name+'.tmp') for x in (state_path,target_path,hash_path)]
            states=_open(temp_paths[0],(n,2*DIM),np.float32); targets=_open(temp_paths[1],(n,2*DIM),np.float32); hashes=_open(temp_paths[2],(n,),"S32")
            for local,pair_index in enumerate(range(offset,offset+n)):
                ci,j=divmod(pair_index,probes); circuit=_decode(gates[ci],qubits[ci],parameters[ci],masks[ci]); state=generate_state(probe_seed(bytes(ids[ci]),j),probe_family(ci,j)); target=simulate(circuit,state); states[local]=np.r_[state.real,state.imag]; targets[local]=np.r_[target.real,target.imag]; hashes[local]=_projective_fingerprint(state)
            states.flush(); targets.flush(); hashes.flush(); del states,targets,hashes
            for temp,final in zip(temp_paths,(state_path,target_path,hash_path)): os.replace(temp,final)
        hashes=np.load(hash_path,mmap_mode="r")
        # Samples are used only for bounded projective-overlap diagnostics.
        if len(projective_sample)<4096:
            x=np.load(state_path,mmap_mode="r"); projective_sample.extend(np.asarray(x[:min(len(x),4096-len(projective_sample))]))
        for ci in range(offset//probes,(offset+n-1)//probes+1):
            lo=max(offset,ci*probes)-ci*probes; hi=min(offset+n,(ci+1)*probes)-ci*probes
            for j in range(lo,hi): family_counts[probe_family(ci,j)]+=1
        shards.append({"offset":offset,"count":n,"states":state_path.name,"targets":target_path.name,"probe_hashes":hash_path.name})
        _status("generation-pairs", arm=arm, pairs=min(offset+n,total), total=total, pairs_per_second=min(offset+n,total)/max(time.monotonic()-started,1e-9), resumable_command="uv run python run_p4_6.py factorial-screen")
    all_hashes=np.concatenate([np.load(root/x["probe_hashes"],mmap_mode="r") for x in shards]); unique, multiplicity=np.unique(all_hashes,return_counts=True)
    sample=np.asarray(projective_sample,np.float32); complex_sample=sample[:,:DIM]+1j*sample[:,DIM:]; overlaps=np.abs(np.sum(complex_sample[:-1].conj()*complex_sample[1:],axis=1))**2 if len(sample)>1 else np.array([])
    probe_stats={"unique_probe_states":int(len(unique)),"probe_reuse_multiplicity":{"minimum":int(multiplicity.min()),"mean":float(multiplicity.mean()),"maximum":int(multiplicity.max()),"reused_state_count":int((multiplicity>1).sum())},"state_family_counts":dict(family_counts),"projective_probe_diversity":{"definition":"adjacent deterministic sample squared overlaps |<psi_i|psi_j>|^2; lower means more diverse","sample_pairs":int(len(overlaps)),"mean":float(overlaps.mean()) if len(overlaps) else None,"median":float(np.median(overlaps)) if len(overlaps) else None,"p95":float(np.percentile(overlaps,95)) if len(overlaps) else None,"maximum":float(overlaps.max()) if len(overlaps) else None}}
    manifest={"schema_version":SCHEMA,"state":"COMPLETED","arm":arm,"circuit_count":circuit_count,"probes_per_circuit":probes,"pair_count":total,"master_prefix":[0,circuit_count],"probe_policy":"SHA256(p4.6-probe-v1 || exact circuit SHA256 || ordered probe index); family=(circuit_index+probe_index) mod 2","shard_size":SHARD_SIZE,"shards":shards,"probe_statistics":probe_stats}
    atomic_json(root/"manifest.json",manifest); return manifest


class ArmData:
    def __init__(self, arm: str):
        self.root=DATA_ROOT/arm; self.manifest=json.loads((self.root/"manifest.json").read_text()); self.probes=self.manifest["probes_per_circuit"]; self.length=self.manifest["pair_count"]
        master=DATA_ROOT/"master"; self.gates=np.load(master/"gates.npy",mmap_mode="r"); self.qubits=np.load(master/"qubits.npy",mmap_mode="r"); self.parameters=np.load(master/"parameters.npy",mmap_mode="r"); self.masks=np.load(master/"masks.npy",mmap_mode="r")
        self.states=[np.load(self.root/x["states"],mmap_mode="r") for x in self.manifest["shards"]]; self.targets=[np.load(self.root/x["targets"],mmap_mode="r") for x in self.manifest["shards"]]
    def batch(self, indices: np.ndarray):
        ci=indices//self.probes; states=np.empty((len(indices),2*DIM),np.float32); targets=np.empty_like(states)
        for shard in np.unique(indices//SHARD_SIZE):
            where=np.flatnonzero(indices//SHARD_SIZE==shard); local=indices[where]%SHARD_SIZE; states[where]=self.states[int(shard)][local]; targets[where]=self.targets[int(shard)][local]
        return self.gates[ci].astype(np.int64),self.qubits[ci].astype(np.int64),self.parameters[ci],self.masks[ci],states,targets


def load_validation(split: str):
    if split not in VALIDATION_SPLITS: raise PermissionError(f"non-validation split refused: {split}")
    root=ROOT/"datasets"; rows=json.loads((root/f"{split}.json").read_text()); payload=np.load(root/f"{split}.npz")
    circuits=[[Gate.from_dict(x) for x in row["gates"]] for row in rows]; tensors=list(zip(*[_tensorize_circuit(c) for c in circuits])); states=payload["inputs"]; targets=payload["targets"]
    return tuple(np.asarray(x) for x in tensors)+(np.c_[states.real,states.imag].astype(np.float32),np.c_[targets.real,targets.imag].astype(np.float32))


def evaluate(model: torch.nn.Module, split: str, device: torch.device) -> float:
    values=[]; batch=load_validation(split); model.eval()
    with torch.inference_mode():
        for start in range(0,len(batch[0]),256):
            x=[torch.as_tensor(v[start:start+256]).to(device) for v in batch]; values.extend(state_fidelity(model(*x[:5]),x[5]).cpu().tolist())
    model.train(); return float(np.mean(values))


def _train_fidelity(model, data: ArmData, device: torch.device, maximum: int = 2048) -> float:
    batch=data.batch(np.arange(min(maximum,data.length))); x=[torch.as_tensor(v).to(device) for v in batch]; model.eval()
    with torch.inference_mode(): value=float(state_fidelity(model(*x[:5]),x[5]).mean().cpu())
    model.train(); return value


def _checkpoint(path: Path, model, optimizer, scheduler, config, step, samples, rng, seen, circuit_exposures, curve) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); payload={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"config":config,"config_hash":digest(config),"step":step,"samples_seen":samples,"numpy_rng":rng.bit_generator.state,"torch_rng":torch.get_rng_state(),"seen_pairs":seen,"circuit_exposures":circuit_exposures,"curve":curve}; tmp=path.with_suffix(".tmp"); torch.save(payload,tmp); os.replace(tmp,path)


def train_arm(arm: str) -> dict[str, Any]:
    metric_path=TRACK_ROOT/f"metrics/{arm}.json"
    if metric_path.exists() and json.loads(metric_path.read_text()).get("state")=="COMPLETED": return json.loads(metric_path.read_text())
    device=ACCEL_DEVICE; torch.manual_seed(SEED); data=ArmData(arm); model=ScaledCCNQE("1m","state").to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,MAX_UPDATES)
    config={"schema_version":SCHEMA,"arm":arm,"unique_circuits":data.manifest["circuit_count"],"probes_per_circuit":data.probes,"pair_count":data.length,"actual_parameters":parameter_count(model),"recipe":RECIPE,"dataset_manifest_hash":_sha(data.root/"manifest.json")}; atomic_json(TRACK_ROOT/f"configs/{arm}.json",config)
    checkpoint=TRACK_ROOT/f"checkpoints/{arm}-latest.pt"; primary=TRACK_ROOT/f"checkpoints/{arm}-best-balanced.pt"; step=samples=0; rng=np.random.default_rng(SEED); seen=np.zeros(data.length,np.bool_); circuit_exposures=np.zeros(config["unique_circuits"],np.int64); curve=[]; best=-math.inf
    if checkpoint.exists():
        p=torch.load(checkpoint,map_location="cpu",weights_only=False)
        if p["config_hash"]!=digest(config): raise ValueError(f"resume refused for {arm}: config differs")
        model.load_state_dict(p["model"]); optimizer.load_state_dict(p["optimizer"]); scheduler.load_state_dict(p["scheduler"]); step=p["step"]; samples=p["samples_seen"]; rng.bit_generator.state=p["numpy_rng"]; torch.set_rng_state(p["torch_rng"]); seen=p["seen_pairs"]; circuit_exposures=p["circuit_exposures"]; curve=p["curve"]; best=max((x["balanced_validation"] for x in curve),default=-math.inf)
    started=time.monotonic(); last=time.monotonic(); loss=torch.tensor(float("nan")); fidelity=torch.tensor([float("nan")])
    while step<MAX_UPDATES and not _STOP:
        indices=rng.integers(data.length,size=BATCH_SIZE); batch=data.batch(indices); x=[torch.as_tensor(v).to(device) for v in batch]; optimizer.zero_grad(set_to_none=True); pred=model(*x[:5]); fidelity=state_fidelity(pred,x[5]); loss=(1-fidelity).mean(); loss.backward()
        if not torch.isfinite(loss) or not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()): raise FloatingPointError(f"{arm}: non-finite training")
        optimizer.step(); scheduler.step(); step+=1; samples+=BATCH_SIZE; seen[indices]=True; np.add.at(circuit_exposures,indices//data.probes,1)
        if step%VALIDATION_INTERVAL==0 or step==MAX_UPDATES:
            accel_synchronize(); metrics={split:evaluate(model,split,device) for split in VALIDATION_SPLITS}; metrics.update(step=step,train_fidelity=_train_fidelity(model,data,device),balanced_validation=balanced_score(metrics["iid_validation"],metrics["composition_ood_validation"],metrics["depth_ood_validation"])); curve.append(metrics)
            _checkpoint(checkpoint,model,optimizer,scheduler,config,step,samples,rng,seen,circuit_exposures,curve)
            if metrics["balanced_validation"]>best: best=metrics["balanced_validation"]; _checkpoint(primary,model,optimizer,scheduler,config,step,samples,rng,seen,circuit_exposures,curve)
        if time.monotonic()-last>=30 or step%VALIDATION_INTERVAL==0:
            elapsed=time.monotonic()-started; _status("training",arm=arm,step=step,total=MAX_UPDATES,samples_seen=samples,loss=float(loss.detach().cpu()),training_fidelity=float(fidelity.mean().detach().cpu()),samples_per_second=(samples-(samples-BATCH_SIZE*step))/max(elapsed,1e-9),eta_seconds=(MAX_UPDATES-step)*elapsed/max(step,1),checkpoint=str(checkpoint),resumable_command="uv run python run_p4_6.py factorial-screen"); last=time.monotonic()
    _checkpoint(checkpoint,model,optimizer,scheduler,config,step,samples,rng,seen,circuit_exposures,curve)
    if _STOP: return {"arm":arm,"state":"INTERRUPTED","step":step}
    final=curve[-1]; p=torch.load(primary,map_location="cpu",weights_only=False); model.load_state_dict(p["model"]); primary_metrics={split:evaluate(model,split,device) for split in VALIDATION_SPLITS}; primary_metrics["balanced_validation"]=balanced_score(primary_metrics["iid_validation"],primary_metrics["composition_ood_validation"],primary_metrics["depth_ood_validation"]); primary_metrics["step"]=p["step"]
    result={"schema_version":SCHEMA,"arm":arm,"state":"COMPLETED","config":config,"optimizer_updates":step,"pair_exposures":samples,"unique_pair_coverage":int(seen.sum()),"unique_pair_coverage_fraction":float(seen.mean()),"effective_pair_epochs":samples/data.length,"circuit_exposures":int(circuit_exposures.sum()),"mean_exposures_per_unique_circuit":float(circuit_exposures.mean()),"probe_exposures":samples,"mean_exposures_per_probe":samples/data.manifest["probe_statistics"]["unique_probe_states"],"samples_seen":samples,"wall_seconds":time.monotonic()-started,"samples_per_second":samples/max(time.monotonic()-started,1e-9),"final_train_fidelity":final["train_fidelity"],"train_validation_gap":final["train_fidelity"]-final["iid_validation"],"primary_checkpoint":{"path":str(primary),"selection":"best balanced validation","metrics":primary_metrics},"best_validation":{split:max(({"value":x[split],"step":x["step"]} for x in curve),key=lambda value:value["value"]) for split in VALIDATION_SPLITS},"final_validation":{split:final[split] for split in VALIDATION_SPLITS},"curve":curve,"probe_statistics":data.manifest["probe_statistics"]}
    atomic_json(metric_path,result); return result


def circuit_statistics(limit: int) -> dict[str, Any]:
    root=DATA_ROOT/"master"; gates=np.load(root/"gates.npy",mmap_mode="r"); qubits=np.load(root/"qubits.npy",mmap_mode="r"); parameters=np.load(root/"parameters.npy",mmap_mode="r"); masks=np.load(root/"masks.npy",mmap_mode="r")
    structures=set(); motif_sets={k:set() for k in (2,3,4)}; gate_counts=Counter(); depths=Counter(); bins=Counter(); interactions=Counter()
    for i in range(limit):
        c=_decode(gates[i],qubits[i],parameters[i],masks[i]); structures.add("|".join(f"{g.name}:{','.join(map(str,g.qubits))}" for g in c)); depths[len(c)]+=1
        for g in c:
            gate_counts[g.name]+=1
            if g.theta is not None: bins[min(11,int(g.theta/(2*math.pi)*12))]+=1
            if g.name=="CNOT": interactions[f"{g.qubits[0]}->{g.qubits[1]}"]+=1
        for k in motif_sets: motif_sets[k].update(motifs(c,k))
    return {"exact_circuit_count":limit,"structural_signature_count":len(structures),"unique_motifs":{f"k{k}":len(v) for k,v in motif_sets.items()},"parameter_bin_coverage":sorted(bins),"distributions":{"primitive_gate":dict(gate_counts),"depth":dict(depths),"parameter_bin":dict(bins),"qubit_interaction":dict(interactions)}}


def audit_factorial(manifests: dict[str,dict[str,Any]]) -> dict[str,Any]:
    stats={arm:circuit_statistics(FACTORIAL_ARMS[arm][0]) for arm in FACTORIAL_ARMS}
    circuit_nesting=all(manifests[a]["master_prefix"][0]==0 for a in manifests) and all(FACTORIAL_ARMS[a][0]>=FACTORIAL_ARMS[b][0] for a,b in zip(FACTORIAL_ARMS,list(FACTORIAL_ARMS)[1:]))
    # Ordered prefix construction proves every probe nesting relation; check deterministic values across a bounded sample too.
    ids=np.load(DATA_ROOT/"master/exact_sha256.npy",mmap_mode="r"); probe_check=all(np.array_equal(ordered_probes(bytes(ids[i]),i,1)[0],ordered_probes(bytes(ids[i]),i,64)[0]) for i in range(min(1000,len(ids))))
    def fractions(counter):
        total=sum(counter.values()); return {str(k):v/total for k,v in counter.items()}
    controls={}
    reference=stats["A1"]["distributions"]
    for arm in FACTORIAL_ARMS:
        deltas={name:max((abs(fractions(stats[arm]["distributions"][name]).get(k,0)-fractions(reference[name]).get(k,0)) for k in set(fractions(stats[arm]["distributions"][name]))|set(fractions(reference[name]))),default=0) for name in reference}
        family=manifests[arm]["probe_statistics"]["state_family_counts"]; deltas["state_family"]=abs(family.get("product",0)/sum(family.values())-.5)
        controls[arm]={"maximum_fraction_differences_from_A1":deltas,"matched_within_0.02":all(x<=.02 for x in deltas.values())}
    audit={"schema_version":SCHEMA,"status":"PASS" if circuit_nesting and probe_check and all(x["matched_within_0.02"] for x in controls.values()) else "FACTORIAL-AUDIT-BLOCKED","circuit_nesting":{"passed":circuit_nesting,"relation":"A5 subset A4 subset A3 subset A2 subset A1","method":"ordered master-pool prefixes"},"probe_nesting":{"passed":probe_check,"relation":"P1(C) subset P4(C) subset P16(C) subset P17(C) subset P64(C)","method":"ordered deterministic probe prefixes; 1000-circuit regeneration check"},"distribution_control":controls,"arms":{a:{**stats[a],**manifests[a]["probe_statistics"],"probes_per_circuit":FACTORIAL_ARMS[a][1]} for a in FACTORIAL_ARMS},"sealed_test_access_count":json.loads((ROOT/"test_access_log.json").read_text())["access_count"]}
    atomic_json(TRACK_ROOT/"factorial_audit.json",audit); return audit


def track_a_verdict(results: dict[str,dict[str,Any]]) -> str:
    scores={a:r["primary_checkpoint"]["metrics"]["balanced_validation"] for a,r in results.items()}; winner=max(scores,key=scores.get)
    if max(scores.values())-min(scores.values())<.01: return "NO-CLEAR-DATA-EFFECT"
    if winner in ("A1","A2"): return "CIRCUIT-COVERAGE-DOMINANT"
    if winner in ("A4","A5"): return "PROBE-COVERAGE-DOMINANT"
    return "MIXED-DATA-EFFECT"


def report_track_a(results: dict[str,dict[str,Any]], audit: dict[str,Any]) -> str:
    winner=max(results,key=lambda a:results[a]["primary_checkpoint"]["metrics"]["balanced_validation"]); verdict=track_a_verdict(results)
    lines=["# CC-NQE P4.6-B Track-A Report","",f"Validation-only winner: **{winner}**",f"Track-A verdict: **{verdict}**","",VERDICT_RULE,"","All primary comparisons use each arm's single best-balanced checkpoint. Per-metric best values are secondary diagnostics. No sealed split was loaded.",""]
    for arm,r in results.items():
        a=audit["arms"][arm]; lines += [f"## {arm}","",f"- exact circuits / structural signatures: {a['exact_circuit_count']:,} / {a['structural_signature_count']:,}",f"- unique k2/k3/k4 motifs: {a['unique_motifs']}",f"- parameter bins: {a['parameter_bin_coverage']}; interactions: {a['distributions']['qubit_interaction']}",f"- probes/circuit: {a['probes_per_circuit']}; probe statistics: {json.dumps(r['probe_statistics'],sort_keys=True)}",f"- primary checkpoint: {json.dumps(r['primary_checkpoint'],sort_keys=True)}",f"- final train / gap: {r['final_train_fidelity']:.8f} / {r['train_validation_gap']:.8f}",f"- best validation (secondary): {json.dumps(r['best_validation'],sort_keys=True)}",f"- final validation: {json.dumps(r['final_validation'],sort_keys=True)}",f"- exposure: updates={r['optimizer_updates']}, pairs={r['pair_exposures']}, unique-pair={r['unique_pair_coverage']} ({r['unique_pair_coverage_fraction']:.6f}), pair-epochs={r['effective_pair_epochs']:.4f}, circuit={r['circuit_exposures']}, mean/circuit={r['mean_exposures_per_unique_circuit']:.4f}, probe={r['probe_exposures']}, mean/probe={r['mean_exposures_per_probe']:.4f}",f"- samples/wall/rate: {r['samples_seen']} / {r['wall_seconds']:.2f}s / {r['samples_per_second']:.2f} samples/s",""]
    lines += ["## Audits","",f"- circuit nesting: {audit['circuit_nesting']}",f"- probe nesting: {audit['probe_nesting']}",f"- distribution control: {json.dumps(audit['distribution_control'],sort_keys=True)}",f"- sealed-test access count: {audit['sealed_test_access_count']}","","## STOP","","Track A complete. Track B/C and seeds 2027/2028 were not run."]
    text="\n".join(lines)+"\n"; (ROOT/"P4_6B_REPORT.md").write_text(text); atomic_json(TRACK_ROOT/"summary.json",{"schema_version":SCHEMA,"state":"COMPLETED","winner":winner,"verdict":verdict,"sealed_test_access_count":audit["sealed_test_access_count"],"track_b_executed":False,"track_c_executed":False}); return text


def factorial_screen() -> dict[str,Any]:
    require_track_a_preconditions(); TRACK_ROOT.mkdir(parents=True,exist_ok=True)
    atomic_json(TRACK_ROOT/"protocol.json",{"schema_version":SCHEMA,"state":"FROZEN-BEFORE-EXECUTION","recipe":RECIPE,"factorial_arms":{arm:{"circuits":c,"probes_per_circuit":p} for arm,(c,p) in FACTORIAL_ARMS.items()},"circuit_nesting":"A5 subset A4 subset A3 subset A2 subset A1 via ordered master prefixes","probe_nesting":"P1(C) subset P4(C) subset P16(C) subset P17(C) subset P64(C) via ordered deterministic probe prefixes","primary_checkpoint":"maximum balanced validation score","balanced_score":"(IID-validation + Composition-OOD-validation + Depth-OOD-validation) / 3","verdict_rule":VERDICT_RULE,"sealed_splits_loaded":False})
    generate_master_pool()
    manifests={arm:generate_arm_pairs(arm,*values) for arm,values in FACTORIAL_ARMS.items()}; audit=audit_factorial(manifests)
    if audit["status"]!="PASS": raise RuntimeError(audit["status"])
    results={}
    for arm in FACTORIAL_ARMS:
        result=train_arm(arm)
        if result.get("state")!="COMPLETED": return result
        results[arm]=result
    if json.loads((ROOT/"test_access_log.json").read_text())["access_count"]!=0: raise RuntimeError("sealed-test access changed")
    report_track_a(results,audit); return json.loads((TRACK_ROOT/"summary.json").read_text())


def install_signal_handlers() -> None:
    def stop(_signum,_frame):
        global _STOP; _STOP=True
    signal.signal(signal.SIGINT,stop); signal.signal(signal.SIGTERM,stop)
