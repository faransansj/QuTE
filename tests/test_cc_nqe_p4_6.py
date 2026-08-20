import json
from pathlib import Path
import numpy as np
import pytest
import torch
from cc_nqe import DIM, Gate, circuit_unitary, generate_state
from cc_nqe_p4_5 import parameter_count, state_fidelity
from cc_nqe_p4_6 import *
from cc_nqe_p4_6_track_a import generate_master_pool, load_validation, ordered_probes, probe_family, probe_seed, track_a_verdict
from cc_nqe_p4_6_track_b import (ALLOCATION, VARIANTS, _model, _protocol_checks, action_metrics, b5_composition_loss,
    operator_action, operator_status, phase_aligned_normalized_frobenius_error,
    soft_unitarity_loss, validate_operator_checkpoint, variant_config)


def _circuit_batch(circuit,max_depth=4):
    from cc_nqe_p4_5 import _tensorize_circuit
    return tuple(torch.as_tensor(x)[None] for x in _tensorize_circuit(circuit,max_depth))

def test_frozen_p45_reconstruction_and_source_hashes(tmp_path):
    out=freeze_p45(tmp_path); assert out['status']=='PASS' and out['screening_count']==7 and out['p1_p4_verified']==30
    hashes=json.loads((tmp_path/'p4_5_input_hashes.json').read_text()); assert len([x for x in hashes if '/metrics/state/' in x])==7

def test_previous_artifacts_are_not_written_by_freeze(tmp_path):
    before=(P45/'screening_results.json').read_bytes(); freeze_p45(tmp_path); assert (P45/'screening_results.json').read_bytes()==before

def test_depth_and_factorial_contracts():
    assert DEPTHS=={'train':[1,2,3,4,5,6],'depth_ood_validation':[7],'depth_ood_test_sealed':[8,9,10]}
    arms=factorial_arms(); assert [arms[x]['pairs'] for x in arms]==[1_000_000,1_000_000,1_000_000,1_000_008,1_000_000]
    assert [arms[x]['probes_per_circuit'] for x in arms]==[1,4,16,17,64]

def test_motif_families_are_disjoint_and_deterministic():
    contract=ood_split_contract(); f=contract['motif_families']; assert not set(f['train'])&set(f['composition_ood_validation']); assert not set(f['composition_ood_validation'])&set(f['composition_ood_test_sealed'])
    c=[Gate('H',(0,)),Gate('CNOT',(0,1)),Gate('RX',(2,),.4),Gate('X',(3,))]
    assert [motif_family(c,k) for k in (2,3,4)]==[motif_family(c,k) for k in (2,3,4)]

def test_sealed_test_prevention_and_access_log(tmp_path):
    assert record_test_access(tmp_path)['access_count']==0
    with pytest.raises(PermissionError): record_test_access(tmp_path,'depth_ood_test_sealed')
    assert json.loads((tmp_path/'test_access_log.json').read_text())['access_count']==0

def test_operator_metrics_phase_invariant_application_and_ordering():
    c1=[Gate('H',(0,))]; c2=[Gate('CNOT',(0,1))]; u1=torch.tensor(circuit_unitary(c1)); u2=torch.tensor(circuit_unitary(c2)); exact=u2@u1
    assert torch.allclose(process_fidelity(exact*torch.exp(torch.tensor(.7j)),exact),torch.tensor(1.,dtype=torch.float64),atol=1e-10)
    assert raw_unitarity_error(exact)<1e-10
    psi=torch.tensor(generate_state(1,'Haar-random')); assert torch.allclose(exact@psi,u2@(u1@psi))
    assert not torch.allclose(u1@u2,u2@u1)

def test_lie_and_basic_cayley_are_unitary():
    args=_circuit_batch([Gate('H',(0,)),Gate('CNOT',(0,1))])
    for kind in ('lie','cayley'):
        u=OperatorModel(kind,{'width':16,'ff':32,'layers':1,'heads':4})(*args)
        assert raw_unitarity_error(u).max()<2e-4

def test_recurrent_weight_sharing_prefix_targets_and_depth_normalization():
    model=RecurrentCCNQE(16); assert len({id(model.transition) for _ in range(7)})==1
    circuits=[[Gate('H',(0,)),Gate('X',(1,))],[Gate('X',(0,))]]; states=np.stack([generate_state(2,'product'),generate_state(3,'product')]); targets=exact_prefix_targets(circuits,states,2)
    assert np.allclose(targets[0,1,:DIM]+1j*targets[0,1,DIM:],circuit_unitary(circuits[0])@states[0])
    t=torch.tensor(targets); mask=torch.tensor([[1,1],[1,0]],dtype=torch.bool); assert prefix_loss(t,t,mask)==0

def test_composition_consistency_is_tautological_negative_control_for_shared_recurrence():
    torch.manual_seed(1); model=RecurrentCCNQE(16); c1=_circuit_batch([Gate('H',(0,))],2); c2=_circuit_batch([Gate('X',(1,))],2); combined=_circuit_batch([Gate('H',(0,)),Gate('X',(1,))],2); state=torch.randn(1,2*DIM)
    assert composition_consistency_loss(model,c1,c2,combined,state)<1e-6

def test_normalized_action_fidelity_prevents_scale_cheating_and_reports_norm():
    target=torch.tensor([[1+0j,0j]]); action=target*37
    fidelity,norm=normalized_action_fidelity(target,action)
    assert torch.allclose(fidelity,torch.ones_like(fidelity)) and torch.allclose(norm,torch.tensor([37.]))

def test_frozen_operator_and_recurrent_protocols():
    op=operator_protocol(); assert op['primary_allocation']=={'name':'A4','unique_circuits':58824,'probes_per_circuit':17,'pairs':1000008}
    assert op['arms']['B4']['supervision']=='PRIVILEGED OPERATOR SUPERVISION'
    rec=recurrent_protocol(); assert rec['lambda_prefix']==1.0 and 'tautological state-composition C4' in rec['excluded'] and 'free rollout' in rec['arms']['C2']

def test_diversity_and_resource_estimate_schema():
    circuits=[[Gate('RX',(0,),.1),Gate('X',(1,))],[Gate('RX',(0,),.2),Gate('X',(1,))]]
    stats=diversity_statistics(circuits); assert stats['N_unique_exact_circuits']==2 and stats['N_unique_structural_signatures']==1 and stats['N_unique_k2_motifs']==1
    estimate=resource_estimate(10**12); assert estimate['sufficient'] and all({'dataset_bytes','peak_ram_bytes','training_time_hours_estimate'}<=set(a) for a in estimate['arms'].values())

def test_synthetic_generators_are_noncommutative_and_prefixes_extrapolate():
    a=np.array([[0,1],[-1,0]]); b=np.array([[1,1],[0,1]]); assert not np.array_equal(a@b,b@a)
    seq=[a,b,a,b,a,b,a,b,a,b]; prefixes=[]; cur=np.eye(2)
    for generator in seq: cur=generator@cur; prefixes.append(cur.copy())
    assert np.array_equal(prefixes[6],seq[6]@prefixes[5]) and np.array_equal(prefixes[9],seq[9]@prefixes[8])

def test_validation_only_checkpoint_selection():
    s=CheckpointSelector(); s.update(1,{'iid_validation':.2,'composition_ood_validation':.1,'depth_ood_validation':.15}); s.update(2,{'iid_validation':.1,'composition_ood_validation':.3,'depth_ood_validation':.2})
    assert s.best['iid']['step']==1 and s.best['composition']['step']==2 and 'test' not in s.best

def test_resume_compatibility():
    config={'seed':2026}; payload={'config_hash':digest(config),'dataset_manifest_hash':'x'}; validate_resume(payload,config,'x')
    with pytest.raises(ValueError,match='config'): validate_resume(payload,{'seed':2027},'x')
    with pytest.raises(ValueError,match='dataset'): validate_resume(payload,config,'y')

def test_track_a_ordered_probes_are_deterministic_nested_and_family_balanced():
    exact=bytes(range(32)); p1=ordered_probes(exact,0,1); p64=ordered_probes(exact,0,64)
    assert np.array_equal(p1[0],p64[0])
    assert probe_seed(exact,0)==probe_seed(exact,0) and probe_seed(exact,0)!=probe_seed(exact,1)
    assert [probe_family(0,i) for i in range(4)]==['product','random-local','product','random-local']
    assert [probe_family(i,0) for i in range(4)]==['product','random-local','product','random-local']

def test_track_a_master_pool_is_one_deterministic_prefix_source(tmp_path,monkeypatch):
    import cc_nqe_p4_6_track_a as track
    monkeypatch.setattr(track,'DATA_ROOT',tmp_path); monkeypatch.setattr(track,'TRACK_ROOT',tmp_path/'track')
    manifest=generate_master_pool(24); ids=np.load(tmp_path/'master/exact_sha256.npy')
    assert manifest['exact_circuit_count']==24 and len({bytes(x) for x in ids})==24
    assert manifest['subset_policy']=='A1-A5 are prefixes of this ordered pool'
    assert [int(np.load(tmp_path/'master/masks.npy')[i].sum()) for i in range(12)]==list(range(1,7))*2

def test_track_a_validation_loader_refuses_sealed_splits():
    with pytest.raises(PermissionError,match='non-validation'): load_validation('composition_ood_test_sealed')
    with pytest.raises(PermissionError,match='non-validation'): load_validation('depth_ood_test_sealed')

def test_track_a_verdict_uses_only_allowed_terminology():
    def rows(values): return {arm:{'primary_checkpoint':{'metrics':{'balanced_validation':value}}} for arm,value in zip(FACTORIAL_ARMS,values)}
    assert track_a_verdict(rows([.3,.2,.2,.2,.1]))=='CIRCUIT-COVERAGE-DOMINANT'
    assert track_a_verdict(rows([.1,.2,.2,.3,.4]))=='PROBE-COVERAGE-DOMINANT'
    assert track_a_verdict(rows([.1,.2,.4,.2,.1]))=='MIXED-DATA-EFFECT'
    assert track_a_verdict(rows([.2,.201,.202,.203,.204]))=='NO-CLEAR-DATA-EFFECT'

def test_track_a_frozen_result_integrity():
    frozen=json.loads((Path('artifacts/cc_nqe_p4_6/factorial/summary.json')).read_text())
    assert frozen['winner']=='A3' and frozen['verdict']=='MIXED-DATA-EFFECT'
    assert frozen['integrity']=={'all_arms_completed':True,'updates_each':10000,'pair_exposures_each':10240000,'seed':2026,'xpu_residency':'PASS','finite_loss_gradients':'PASS','circuit_nesting':'PASS','probe_nesting':'PASS','distribution_controls':'PASS','sealed_test_access_count':0}
    assert all(x['primary_checkpoint']['selection']=='best balanced validation' for x in frozen['arms'].values())

def test_track_b_a4_allocation_and_supervision_contracts():
    assert ALLOCATION=={'unique_circuits':58824,'probes_per_circuit':17,'state_action_pairs':1000008,'source_arm':'A4'}
    common=['C','psi_in','psi_out']
    assert all(VARIANTS[v]['supervision']==common for v in ('B0','B1','B2','B3'))
    assert VARIANTS['B4']['label']=='PRIVILEGED OPERATOR SUPERVISION' and VARIANTS['B5']['label']=='PRIVILEGED OPERATOR SUPERVISION'
    assert VARIANTS['B2']['lambda_unitary']==.1 and VARIANTS['B5']['lambda_comp']==.3

def test_track_b_action_fidelity_scale_invariance_and_raw_norm():
    state=torch.zeros(1,2*DIM); state[0,0]=1
    target=state.clone()
    operator=torch.eye(DIM,dtype=torch.complex64)[None]; operator[0,0,0]=37
    fidelity,norm=action_metrics(operator,state,target)
    assert torch.allclose(fidelity,torch.ones_like(fidelity)) and torch.allclose(norm,torch.tensor([37.]))

def test_track_b_soft_unitarity_loss_and_basic_cayley_exact_unitarity():
    eye=torch.eye(DIM,dtype=torch.complex64)[None]
    assert soft_unitarity_loss(eye)==0
    bad=eye*2; assert torch.allclose(soft_unitarity_loss(bad),torch.tensor(9.))
    zero=torch.zeros_like(eye); assert torch.equal(torch.linalg.solve(eye+zero,eye-zero),eye)
    b=torch.randn(2,DIM,DIM,dtype=torch.complex64,requires_grad=True); a=(b-b.mH)/2; assert torch.allclose(a,-a.mH)
    u=torch.linalg.solve(eye+a,eye-a); assert raw_unitarity_error(u).max()<1e-5
    u.real.sum().backward(); assert torch.isfinite(b.grad).all() and b.grad.abs().sum()>0
    args=_circuit_batch([Gate('H',(0,)),Gate('CNOT',(0,1))]); exact=_model('B3')(*args); assert raw_unitarity_error(exact).max()<1e-5

def test_track_b_process_phase_and_frobenius_metrics():
    exact=torch.eye(DIM,dtype=torch.complex64)[None]; phased=exact*torch.exp(torch.tensor(.7j))
    assert torch.allclose(process_fidelity(phased,exact),torch.ones(1),atol=1e-6)
    assert phase_aligned_normalized_frobenius_error(phased,exact).max()<1e-6

def test_track_b_operator_to_state_application_and_composition_order():
    c1=[Gate('H',(0,))]; c2=[Gate('CNOT',(0,1))]; u1=torch.tensor(circuit_unitary(c1),dtype=torch.complex64); u2=torch.tensor(circuit_unitary(c2),dtype=torch.complex64); state=torch.tensor(generate_state(9,'Haar-random'),dtype=torch.complex64)
    real=torch.cat((state.real,state.imag))[None]; action,norm=operator_action((u2@u1)[None],real)
    got=torch.complex(action[0,:DIM],action[0,DIM:]); assert torch.allclose(got,u2@(u1@state),atol=1e-6) and torch.allclose(norm,torch.ones(1),atol=1e-6)
    assert not torch.allclose(u2@u1,u1@u2)

def test_track_b_b5_composition_loss_is_non_tautological_and_has_gradient():
    torch.manual_seed(4); direct=torch.randn(2,DIM,DIM,dtype=torch.complex64,requires_grad=True); first=torch.randn(2,DIM,DIM,dtype=torch.complex64,requires_grad=True); second=torch.randn(2,DIM,DIM,dtype=torch.complex64,requires_grad=True)
    loss=b5_composition_loss(direct,second,first); assert loss>0
    loss.backward(); assert direct.grad is not None and direct.grad.abs().sum()>0 and first.grad.abs().sum()>0 and second.grad.abs().sum()>0

def test_track_b_parameter_budget_matching_and_protocol_equality():
    counts={v:variant_config(v)['actual_parameters'] for v in VARIANTS}
    assert all(850000<=n<=1150000 for n in counts.values()) and counts['B0']==1001472 and all(counts[v]==1073312 for v in ('B1','B2','B3','B4','B5'))
    assert all(_protocol_checks().values())

def test_track_b_checkpoint_resume_and_status_schema():
    config={'variant':'B1'}; payload={'config':config,'config_hash':digest(config),'dataset_manifest_hash':'x'}; validate_operator_checkpoint(payload,config,'x')
    with pytest.raises(ValueError,match='config'): validate_operator_checkpoint(payload,{'variant':'B2'},'x')
    with pytest.raises(ValueError,match='dataset'): validate_operator_checkpoint(payload,config,'y')
    value=operator_status(); assert value['schema_version']==SCHEMA and set(value['scientific_runs'])==set(VARIANTS) and value['sealed_test_access_count']==0

def test_track_b_screen_preflights_before_any_scientific_run(monkeypatch):
    import cc_nqe_p4_6_track_b as track
    called=[]; monkeypatch.setattr(track,'require_operator_preconditions',lambda:None); monkeypatch.setattr(track,'operator_preflight',lambda:{'status':'B3-XPU-BLOCKED'}); monkeypatch.setattr(track,'operator_run',lambda variant:called.append(variant))
    with pytest.raises(RuntimeError,match='refused before any scientific variant'): track.operator_screen()
    assert called==[]

def test_track_b_completed_run_returns_frozen_metric(monkeypatch,tmp_path):
    import cc_nqe_p4_6_track_b as track
    monkeypatch.setattr(track,'OP_ROOT',tmp_path); monkeypatch.setattr(track,'require_operator_preconditions',lambda:None)
    expected={'schema_version':SCHEMA,'variant':'B1','state':'COMPLETED','latest_validation':{'iid_validation':{'predicted_operator_state_fidelity':.5}}}
    atomic_json(tmp_path/'metrics/B1.json',expected)
    assert track.operator_run('B1')==expected

@pytest.mark.skipif(not torch.xpu.is_available(),reason='native XPU unavailable')
def test_track_b_xpu_cayley_forward_backward_and_residency():
    model=_model('B3','xpu:0'); args=tuple(x.to('xpu:0') for x in _circuit_batch([Gate('H',(0,)),Gate('CNOT',(0,1))])); out=model(*args); loss=raw_unitarity_error(out).mean()+(1-process_fidelity(out,torch.eye(DIM,dtype=out.dtype,device=out.device)[None])).mean(); loss.backward()
    assert out.device.type=='xpu' and torch.isfinite(out).all() and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()) and raw_unitarity_error(out).max()<1e-4

@pytest.mark.skipif(not torch.xpu.is_available(),reason='native XPU unavailable')
def test_xpu_residency():
    model=RecurrentCCNQE(16).to('xpu'); args=tuple(x.to('xpu') for x in _circuit_batch([Gate('H',(0,))])); state=torch.randn(1,2*DIM,device='xpu'); out=model(*args,state); assert out.device.type=='xpu' and torch.isfinite(out).all()
