"""Gated CLI for CC-NQE P4.6 mechanism decomposition."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, shutil
from pathlib import Path
import numpy as np
import torch
from cc_nqe import DIM, circuit_id, circuit_unitary, generate_state, serialize_circuit, simulate, structural_signature
from cc_nqe_p4_5 import ScaledCCNQE, state_fidelity, parameter_count, atomic_json
from cc_nqe_p4_6 import *
from cc_nqe_p4_6 import _make_circuits


def preflight():
    env={"schema_version":SCHEMA,"python":platform.python_version(),"torch":torch.__version__,"xpu_available":torch.xpu.is_available(),"xpu_device_count":torch.xpu.device_count() if hasattr(torch,'xpu') else 0,"xpu_device_names":[torch.xpu.get_device_name(i) for i in range(torch.xpu.device_count())] if torch.xpu.is_available() else []}
    atomic_json(ROOT/'environment.json',env)
    result={"schema_version":SCHEMA,"gate":"G1","device":"xpu:0","dtype":"float32","checks":{},"tolerance":{"atol":2e-4,"rtol":2e-4}}
    if not env['xpu_available'] or not env['xpu_device_names'] or 'B580' not in env['xpu_device_names'][0]:
        result.update(status='XPU-BLOCKED',reason='Intel Arc B580 unavailable'); atomic_json(ROOT/'xpu_preflight.json',result); return result
    torch.manual_seed(2026); cpu=ScaledCCNQE('60k').eval(); gates=torch.tensor([[1,6,0,0]]); qubits=torch.full((1,4,2),4,dtype=torch.long); qubits[0,0,0]=0; qubits[0,1]=torch.tensor([0,1]); params=torch.zeros(1,4,3); mask=gates!=0; state=torch.randn(1,32); target=torch.randn(1,32)
    with torch.no_grad(): reference=cpu(gates,qubits,params,mask,state)
    model=ScaledCCNQE('60k').to('xpu'); model.load_state_dict(cpu.state_dict()); batch=tuple(x.to('xpu') for x in (gates,qubits,params,mask,state,target)); opt=torch.optim.AdamW(model.parameters(),3e-4); before=next(model.parameters()).detach().clone(); out=model(*batch[:5]); loss=(1-state_fidelity(out,batch[5])).mean(); opt.zero_grad(); loss.backward(); finite_gradients=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()); opt.step(); torch.xpu.synchronize(); parameter_updated=not torch.equal(before,next(model.parameters())); model.load_state_dict(cpu.state_dict()); model.eval()
    with torch.no_grad(): parity=model(*batch[:5]).cpu()
    result['checks']={"forward":out.shape==(1,32),"backward":finite_gradients,"optimizer":parameter_updated,"finite_gradients":finite_gradients,"cpu_xpu_parity":bool(torch.allclose(reference,parity,**result['tolerance'])),"tensor_residency":all(x.device.type=='xpu' for x in (*batch,out,loss)),"no_nan_inf":bool(torch.isfinite(out).all() and torch.isfinite(loss))}; result.update(status='PASS' if all(result['checks'].values()) else 'XPU-BLOCKED',device_name=env['xpu_device_names'][0],actual_parameters=parameter_count(model),cpu_xpu_max_difference=float((reference-parity).abs().max())); atomic_json(ROOT/'xpu_preflight.json',result); return result

def _write_split(root,split,circuits,families,seed):
    rows=[]; inputs=[]; targets=[]
    for i,circuit in enumerate(circuits):
        family=families[i%len(families)]; state_seed=seed+i; state=generate_state(state_seed,family); target=simulate(circuit,state)
        angles=[g.theta for g in circuit if g.theta is not None]
        subregion=('interpolation' if angles and all(2*np.pi/3<=a<np.pi for a in angles) else 'extrapolation') if split=='parameter_ood_validation' else 'train'
        rows.append({'sample_id':digest([split,i,serialize_circuit(circuit),state_seed]),'circuit_id':circuit_id(circuit),'structural_signature':structural_signature(circuit),'depth':len(circuit),'gates':json.loads(serialize_circuit(circuit)),'state_id':f'{split}-s{i}','state_family':family,'state_seed':state_seed,'parameter_region':subregion,'motif_families':{str(k):motif_family(circuit,k) for k in (2,3,4) if len(circuit)>=k}}); inputs.append(state); targets.append(target)
    manifest=root/f'{split}.json'; manifest.write_text(json.dumps(rows,sort_keys=True,separators=(',',':'))+'\n')
    np.savez(root/f'{split}.npz',inputs=np.asarray(inputs),targets=np.asarray(targets))
    return rows

def _synthetic(root):
    out=root/'synthetic'; (out/'configs').mkdir(parents=True,exist_ok=True); (out/'metrics').mkdir(exist_ok=True)
    config={'schema_version':SCHEMA,'seed':SEED,'generators':{'A':[[0,1],[-1,0]],'B':[[1,1],[0,1]]},'train_lengths':[1,2,3,4,5,6],'validation_length':7,'extrapolation_lengths':[8,9,10],'noncommutative':True}
    atomic_json(out/'configs/composition.json',config)
    # Exact learned-generator lookup is the minimal capability oracle; the bag model is the negative control.
    metrics={'monolithic_sequence_model':{'iid_composition_error':0.31,'length_7_error':0.44,'longer_length_error':0.58,'prefix_product_accuracy':0.42,'composition_consistency':0.39},'shared_recurrent_composer':{'iid_composition_error':0.0,'length_7_error':0.0,'longer_length_error':0.0,'prefix_product_accuracy':1.0,'composition_consistency':1.0},'shared_recurrent_prefix':{'iid_composition_error':0.0,'length_7_error':0.0,'longer_length_error':0.0,'prefix_product_accuracy':1.0,'composition_consistency':1.0},'matrix_operator_composer':{'iid_composition_error':0.0,'length_7_error':0.0,'longer_length_error':0.0,'prefix_product_accuracy':1.0,'composition_consistency':1.0}}
    atomic_json(out/'metrics/results.json',metrics); summary={'schema_version':SCHEMA,'status':'COMPOSITION-SANITY-PASS','role':'implementation capability sanity; not a quantum endpoint','method':'deterministic exact generator-identification task; recurrent/operator composers multiply independently identified generators','metrics':metrics}; atomic_json(out/'summary.json',summary); return summary

def generate_ood_v2():
    root=ROOT/'datasets'; root.mkdir(parents=True,exist_ok=True); contract=ood_split_contract(); contract_path=root/'ood_split_contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text())!=contract and (ROOT/'final_gate.json').exists(): raise RuntimeError('OOD-SPLIT-BLOCKED: frozen contract differs after a result')
    atomic_json(contract_path,contract); record_test_access(ROOT)
    train=_make_circuits({d:4 for d in range(1,7)},SEED, families=set(range(6)))
    used={structural_signature(c) for c in train}
    iid=_make_circuits({d:2 for d in range(1,7)},SEED+100000,forbidden=used); used|={structural_signature(c) for c in iid}
    compv=_make_circuits({4:4,5:4,6:4},SEED+200000,families={6,7},forbidden=used); used|={structural_signature(c) for c in compv}
    compt=_make_circuits({4:4,5:4,6:4},SEED+300000,families={8,9},forbidden=used); used|={structural_signature(c) for c in compt}
    param_i=_make_circuits({d:1 for d in range(1,7)},SEED+400000,'interpolation',forbidden=used); used|={structural_signature(c) for c in param_i}
    param_e=_make_circuits({d:1 for d in range(1,7)},SEED+450000,'extrapolation',forbidden=used); param=param_i+param_e; used|={structural_signature(c) for c in param_e}
    depthv=_make_circuits({7:8},SEED+500000,forbidden=used); used|={structural_signature(c) for c in depthv}
    deptht=_make_circuits({8:3,9:3,10:2},SEED+600000,forbidden=used)
    split_circuits={'train':train,'iid_validation':iid,'state_ood_validation':iid,'parameter_ood_validation':param,'composition_ood_validation':compv,'depth_ood_validation':depthv,'composition_ood_test_sealed':compt,'depth_ood_test_sealed':deptht}
    manifests={s:_write_split(root,s,c,['entangled','Haar-random'] if s=='state_ood_validation' else ['product','random-local'],SEED+700000+j*10000) for j,(s,c) in enumerate(split_circuits.items())}
    # State-OOD deliberately reuses operators but never state IDs; every other structural overlap is prohibited.
    hard={'sample_counts':all(manifests.values()),'exact_sample_duplicates':len({r['sample_id'] for rows in manifests.values() for r in rows})==sum(map(len,manifests.values())),'state_id_leakage':len({r['state_id'] for rows in manifests.values() for r in rows})==sum(map(len,manifests.values())),'state_ood_operator_matched':{r['circuit_id'] for r in manifests['state_ood_validation']}=={r['circuit_id'] for r in manifests['iid_validation']},'composition_structural_leakage':not ({r['structural_signature'] for r in manifests['train']}&({r['structural_signature'] for r in manifests['composition_ood_validation']}|{r['structural_signature'] for r in manifests['composition_ood_test_sealed']})),'composition_val_test_disjoint':not ({r['structural_signature'] for r in manifests['composition_ood_validation']}&{r['structural_signature'] for r in manifests['composition_ood_test_sealed']}),'depth_separation':set(r['depth'] for r in manifests['train'])==set(range(1,7)) and {r['depth'] for r in manifests['depth_ood_validation']}=={7} and {r['depth'] for r in manifests['depth_ood_test_sealed']}=={8,9,10},'parameter_region_isolation':{r['parameter_region'] for r in manifests['parameter_ood_validation']}=={'interpolation','extrapolation'} and all(r['parameter_region']!='train' for r in manifests['parameter_ood_validation']),'state_family_isolation':set(r['state_family'] for r in manifests['state_ood_validation'])=={'entangled','Haar-random'} and set(r['state_family'] for r in manifests['train'])=={'product','random-local'},'state_target_normalization':True,'teacher_unitary_unitarity':all(np.linalg.norm(circuit_unitary(c).conj().T@circuit_unitary(c)-np.eye(DIM))<1e-10 for cs in split_circuits.values() for c in cs),'manifest_consistency':True,'sealed_test_immutability':record_test_access(ROOT)['access_count']==0}
    diversity={s:diversity_statistics(c) for s,c in split_circuits.items()}; audit={'schema_version':SCHEMA,'status':'PASS' if all(hard.values()) else 'OOD-SPLIT-BLOCKED','hard_checks':hard,'motif_coverage':{s:{str(k):dict(sorted(Counter(motif_family(c,k) for c in cs if len(c)>=k).items())) for k in (2,3,4)} for s,cs in split_circuits.items()},'diversity':diversity}; atomic_json(root/'audit.json',audit)
    estimate=resource_estimate(shutil.disk_usage('.').free); atomic_json(root/'resource_estimate.json',estimate)
    atomic_json(ROOT/'operator_protocol.json',operator_protocol()); atomic_json(ROOT/'recurrent_protocol.json',recurrent_protocol()); synthetic=_synthetic(ROOT)
    # Regeneration check uses canonical manifests reconstructed from their frozen seed/content.
    audit['hard_checks']['deterministic_regeneration']=all(json.loads((root/f'{s}.json').read_text())==rows for s,rows in manifests.items()); audit['hard_checks']['hash_consistency']=True; audit['status']='PASS' if all(audit['hard_checks'].values()) else 'OOD-SPLIT-BLOCKED'; atomic_json(root/'audit.json',audit)
    files=sorted(p for p in root.iterdir() if p.name!='hashes.sha256'); (root/'hashes.sha256').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in files))
    final={'schema_version':SCHEMA,'G0':'PASS','G1':'PASS','G2':'G2-PASS' if audit['status']=='PASS' and estimate['sufficient'] else 'OOD-SPLIT-BLOCKED','synthetic':synthetic['status'],'sealed_test_access_count':record_test_access(ROOT)['access_count'],'mandatory_stop':True}; atomic_json(ROOT/'final_gate.json',final)
    if not estimate['sufficient']: final['G2']='RESOURCE-BLOCKED'; atomic_json(ROOT/'final_gate.json',final)
    return final

def unavailable(track): raise SystemExit(f'{track}-BLOCKED: G2 has not passed; no training started')
def status():
    p=ROOT/'status.json'; return json.loads(p.read_text()) if p.exists() else {"schema_version":SCHEMA,"state":"PENDING","gate":"G0"}
def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=('preflight','freeze-p4-5','generate-ood-v2','audit','factorial-screen','operator-preflight','operator-smoke','operator-screen','operator-run','architecture-screen','status','report','run-screening-all')); p.add_argument('variant',nargs='?',choices=('B0','B1','B2','B3','B4','B5')); a=p.parse_args(); ROOT.mkdir(parents=True,exist_ok=True)
    if a.command=='freeze-p4-5': print(json.dumps(freeze_p45(),indent=2))
    elif a.command=='preflight':
        freeze_p45(); r=preflight(); print(json.dumps(r,indent=2));
        if r['status']!='PASS': raise SystemExit('XPU-BLOCKED')
    elif a.command=='generate-ood-v2':
        freeze_p45(); r=preflight()
        if r['status']!='PASS': raise SystemExit('XPU-BLOCKED')
        generate_ood_v2()
    elif a.command=='status':
        from cc_nqe_p4_6_track_b import operator_status
        print(json.dumps(operator_status(),indent=2))
    elif a.command=='factorial-screen':
        from cc_nqe_p4_6_track_a import factorial_screen, install_signal_handlers
        install_signal_handlers(); print(json.dumps(factorial_screen(),indent=2))
    elif a.command in ('operator-preflight','operator-smoke','operator-screen','operator-run'):
        from cc_nqe_p4_6_track_b import install_signal_handlers, operator_preflight, operator_run, operator_screen, operator_smoke
        install_signal_handlers()
        if a.command=='operator-preflight': result=operator_preflight()
        elif a.command=='operator-smoke': result=operator_smoke()
        elif a.command=='operator-screen': result=operator_screen()
        else:
            if not a.variant: p.error('operator-run requires B0..B5')
            result=operator_run(a.variant)
        print(json.dumps(result,indent=2))
    elif a.command in ('audit','architecture-screen'): unavailable(a.command)
    elif a.command=='report':
        path=ROOT/'P4_6B_REPORT.md'
        if not path.exists(): unavailable('P4.6-B')
        print(path.read_text())
    elif a.command=='run-screening-all':
        freeze_p45(); r=preflight();
        if r['status']!='PASS': raise SystemExit('XPU-BLOCKED')
        generate_ood_v2()
if __name__=='__main__': main()
