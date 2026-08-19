import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cc_nqe import Gate, circuit_unitary, generate_circuit, generate_state
from cc_nqe_p4_5 import (CircuitDataset, DIM, Progress, ScaledCCNQE, ShardedDataset,
                         apply_operator, atomic_json, audit_dataset, composition_fidelity,
                         config_hash, generate_dataset, load_checkpoint, operator_fidelity,
                         phase_aligned_matrix_error, save_checkpoint, state_fidelity,
                         unitarity_error, xpu_preflight)


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    root=tmp_path_factory.mktemp("p45")
    generate_dataset(root,master_samples=64,eval_per_split=12,states_per_circuit=4,shard_size=7,seed=818)
    return root


def test_nested_fixed_sharded_deterministic_and_no_leakage(mini):
    audit=audit_dataset(mini)
    assert audit["status"]=="PASS",audit
    manifests=[json.loads((mini/f"train_{s}_manifest.json").read_text()) for s in ("10k","100k","1m")]
    assert [x["sample_count"] for x in manifests]==[64,64,64]
    evaluation=json.loads((mini/"evaluation_manifest.json").read_text()); before=(mini/"evaluation_manifest.json").read_bytes()
    data=ShardedDataset(mini,"train",11); assert len(data)==11 and len(data[0])==6
    assert (mini/"evaluation_manifest.json").read_bytes()==before and evaluation["frozen"]
    first=np.asarray(data[0][-1]); assert np.isclose(np.linalg.norm(first[:DIM]+1j*first[DIM:]),1)
    # Recorded seeds reconstruct exactly.
    state_row=json.loads((mini/"states.jsonl").read_text().splitlines()[0]); states=np.load(mini/"states.npy")
    assert np.array_equal(generate_state(state_row["generator_seed"],state_row["family"]),states[0])
    assert len(CircuitDataset(mini,"train"))==16


def test_state_and_operator_metrics_phase_invariant_and_application():
    psi=generate_state(8,"Haar-random"); real=torch.tensor(np.r_[psi.real,psi.imag],dtype=torch.float32)[None]
    phase=torch.tensor(np.exp(0.7j)); phased=torch.cat(((torch.complex(real[:,:DIM],real[:,DIM:])*phase).real,(torch.complex(real[:,:DIM],real[:,DIM:])*phase).imag),1)
    assert torch.allclose(state_fidelity(real,phased),torch.ones(1),atol=1e-6)
    c1,c2=[Gate("H",(0,))],[Gate("X",(0,))]; u1,u2=circuit_unitary(c1),circuit_unitary(c2); u21=u2@u1
    assert np.isclose(operator_fidelity(np.exp(0.5j)*u21,u21),1)
    assert np.isclose(composition_fidelity(u21,u2,u1),1)
    assert np.max(unitarity_error(u21))<1e-12
    assert np.max(phase_aligned_matrix_error(np.exp(0.5j)*u21,u21))<1e-12
    op=torch.tensor(u21,dtype=torch.complex64)[None]; applied=apply_operator(op,real)
    exact=u21@psi; assert torch.allclose(applied,torch.tensor(np.r_[exact.real,exact.imag],dtype=torch.float32)[None],atol=2e-6)
    assert not np.isclose(composition_fidelity(u1@u2,u2,u1),1)  # ordering is observable for these circuits


def _inputs(device="cpu"):
    gates=torch.tensor([[1,6,0,0]],device=device); qubits=torch.full((1,4,2),4,dtype=torch.long,device=device); qubits[0,0,0]=0; qubits[0,1]=torch.tensor([0,1],device=device)
    params=torch.zeros(1,4,3,device=device); mask=gates!=0; state=torch.randn(1,32,device=device)
    return gates,qubits,params,mask,state


def test_model_family_shapes_and_cached_equivalence():
    torch.manual_seed(4)
    for task in ("state","operator"):
        model=ScaledCCNQE("60k",task).eval(); args=_inputs(); context=model.encode_context(*args[:4]); direct=model(*args[:4],args[4] if task=="state" else None); cached=model.forward_cached(context,args[4] if task=="state" else None)
        assert torch.equal(direct,cached)
        assert direct.shape==((1,32) if task=="state" else (1,16,16))


def test_progress_jsonl_atomic_status_and_tty_modes(tmp_path):
    non_tty=io.StringIO(); progress=Progress(tmp_path,interval=0,stream=non_tty); row=progress.update(experiment_id="x",phase="G3",task="state",step=1,maximum_steps=2,training_loss=.5,training_fidelity=.5,device="xpu:0",eta_seconds=None,state="RUNNING")
    assert set(row)==set(Progress.FIELDS); assert "ETA: measuring..." in non_tty.getvalue(); assert json.loads((tmp_path/"status.json").read_text())["step"]==1
    class TTY(io.StringIO):
        def isatty(self): return True
    tty=TTY(); Progress(tmp_path,stream=tty).update(experiment_id="x",phase="G3",step=2,maximum_steps=2,device="xpu:0",state="COMPLETED")
    assert "\033[2K" in tty.getvalue(); assert len((tmp_path/"progress.jsonl").read_text().splitlines())==2
    atomic_json(tmp_path/"status.json",{"complete":True}); assert not (tmp_path/"status.json.tmp").exists()


def test_checkpoint_resume_equivalence_and_hash_rejection(tmp_path):
    torch.manual_seed(9); model=ScaledCCNQE("60k"); opt=torch.optim.AdamW(model.parameters(),1e-3); scheduler=torch.optim.lr_scheduler.StepLR(opt,1); config={"task":"state","model_scale":"60k","dtype":"float32","optimizer":{"name":"AdamW"}}
    args=_inputs(); loss=(1-state_fidelity(model(*args[:4],args[4]),torch.randn(1,32))).mean(); loss.backward(); opt.step(); scheduler.step(); path=tmp_path/"checkpoint.pt"; save_checkpoint(path,model,opt,scheduler,config,"dataset-hash",1,1,.2)
    expected={k:v.clone() for k,v in model.state_dict().items()}; restored=ScaledCCNQE("60k"); restored_opt=torch.optim.AdamW(restored.parameters(),1e-3); restored_scheduler=torch.optim.lr_scheduler.StepLR(restored_opt,1); payload=load_checkpoint(path,restored,restored_opt,restored_scheduler,config,"dataset-hash")
    assert payload["step"]==1 and all(torch.equal(expected[k],restored.state_dict()[k]) for k in expected)
    with pytest.raises(ValueError,match="config"): load_checkpoint(path,restored,restored_opt,restored_scheduler,config|{"dtype":"float16"},"dataset-hash")
    with pytest.raises(ValueError,match="dataset"): load_checkpoint(path,restored,restored_opt,restored_scheduler,config,"wrong")


def test_xpu_preflight_is_real_or_explicitly_blocked(tmp_path,monkeypatch):
    monkeypatch.setattr("cc_nqe_p4_5.ROOT",tmp_path)
    result=xpu_preflight()
    assert result["status"] in ("PASS","XPU-BLOCKED")
    if not torch.xpu.is_available():
        assert result["status"]=="XPU-BLOCKED" and result["xpu_device_count"]==0
    else:
        assert all(result["checks"].values()) and result["model_device"].startswith("xpu") and result["batch_device"].startswith("xpu") and result["loss_device"].startswith("xpu")


@pytest.mark.skipif(not torch.xpu.is_available(),reason="native XPU unavailable")
def test_xpu_backward_optimizer_and_cpu_parity():
    torch.manual_seed(3); cpu=ScaledCCNQE("60k"); xpu=ScaledCCNQE("60k").to("xpu"); xpu.load_state_dict(cpu.state_dict()); args=_inputs(); xargs=tuple(x.to("xpu") for x in args); target=torch.randn(1,32,device="xpu"); opt=torch.optim.Adam(xpu.parameters(),1e-4); out=xpu(*xargs[:4],xargs[4]); loss=(1-state_fidelity(out,target)).mean(); loss.backward(); opt.step(); torch.xpu.synchronize()
    assert out.device.type==loss.device.type=="xpu" and torch.isfinite(loss)
    xpu.load_state_dict(cpu.state_dict()); cpu.eval(); xpu.eval()
    with torch.no_grad(): assert torch.allclose(cpu(*args[:4],args[4]),xpu(*xargs[:4],xargs[4]).cpu(),atol=2e-4,rtol=2e-4)
