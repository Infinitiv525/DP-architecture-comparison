"""Audit routing, immutable training sources, data schema and all20 initializations."""
import hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
import numpy as np
SOURCE=Path(__file__).resolve().parent
RUN=Path("/mnt/project/perun2601449/dp_llm_cfd/results/architecture_confirmation_20261124_v2")
def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
start=time.monotonic()
integrity=read(SOURCE/"source_integrity.json")
for rel,sha in integrity["files"].items():assert digest(SOURCE/rel)==sha,rel
plan=read(SOURCE/"plan.json");selection=read(SOURCE/"selection.json");tasks=selection["tasks"]
assert (RUN/"selection.json").read_bytes()==(SOURCE/"selection.json").read_bytes()
assert len(tasks)==63 and len({r["id"] for r in tasks})==63
for index,row in enumerate(tasks):
    assert index==row["index"]
    assert row["configuration_index"]==plan["configuration_indices"][row["model"]]
    if index<60:
        assert row["role"]=="independent_main_training"
        assert row["seed"]==plan["main_seeds"][index//3]
        assert row["model"]==plan["methods"][index%3]
    else:assert row["role"]=="technical_preflight_only"
assert set(plan["main_seeds"]).isdisjoint(range(931101,931106))
old_primary=read(Path(plan["parents"]["primary"])/"selection.json")
old_dense=read("/mnt/project/perun2601449/dp_llm_cfd/results/architecture_dense_fno_20261112_v1/run_c93033883d5f56d4/selection.json")
for method in plan["methods"]:
    reference=old_dense if method=="fno" else old_primary
    configs={r["configuration_index"] for r in reference["tasks"] if r["model"]==method}
    assert configs=={plan["configuration_indices"][method]},method
for item in read(SOURCE/"copy_provenance.json"):
    assert digest(Path(item["parent"]))==item["sha256"]==digest(Path(item["copy"]))
policies={}
fixtures=[]
for kind in ["primary","dense"]:
    parent=Path(plan["parents"][kind]);old=read(parent/"policy.json");new=read(SOURCE/kind/"policy.json")
    for key in ["dependency_source","dependency_sha256","fixture_root","fixture_ready_sha256",
                "main_budget_seconds_per_run","final_main_evaluation_period_seconds"]:
        assert new[key]==old[key],(kind,key)
    assert new["main_budget_seconds_per_run"]==14400
    for name,sha in new["dependency_sha256"].items():
        assert digest(Path(new["dependency_source"])/name)==sha
    folder=Path(new["fixture_root"])/"fixture"
    assert digest(folder/"ready.json")==new["fixture_ready_sha256"]
    ready=read(folder/"ready.json")
    assert ready["train_cases"]==640 and ready["validation_cases"]==160 and ready["test_targets_read"] is False
    expected={"train_x.npy":(640,4096,665),"train_y.npy":(640,4096,192),
              "validation_x.npy":(160,4096,665),"validation_y.npy":(160,4096,192)}
    for name,shape in expected.items():
        arr=np.load(folder/name,mmap_mode="r",allow_pickle=False);assert arr.shape==shape and arr.dtype==np.float32
        assert (folder/name).stat().st_size==ready["files_bytes"][name]
    fixtures.append(dict(kind=kind,ready_sha256=digest(folder/"ready.json"),
        fixture=str(folder),normalization=ready["normalization"]))
assert fixtures[0]["ready_sha256"]==fixtures[1]["ready_sha256"]
probes=[]
for method in plan["methods"]:
    kind="dense" if method=="fno" else "primary"
    group=[]
    for label,folder in [("reference",Path(plan["parents"][kind])),("copy",SOURCE/kind)]:
        p=subprocess.run([sys.executable,"-B",str(SOURCE/"probe_initialization.py"),"--folder",str(folder),
            "--method",method,"--seeds",json.dumps(plan["main_seeds"])],capture_output=True,text=True)
        assert p.returncode==0,(method,label,p.stderr)
        value=json.loads(p.stdout.strip().splitlines()[-1])
        assert not value["cuda_initialized"] and value["torch"]=="2.12.0+cu130" and value["numpy"]=="2.5.4"
        group.append(value)
    assert group[0]==group[1],method
    assert len(set(r["state_sha256"] for r in group[0]["records"]))==20
    assert len(set(r["first_epoch_order_sha256"] for r in group[0]["records"]))==20
    probes.append(group[0])
    print("CPU_MODEL_PARITY_PASS",method,20,flush=True)
for index in range(20):
    hashes=[row["records"][index]["first_epoch_order_sha256"] for row in probes]
    assert len(set(hashes))==1
for rel,sha in integrity["files"].items():assert digest(SOURCE/rel)==sha
result=dict(status="CPU_REFERENCE_PARITY_AND_SETUP_PASS",source_integrity_sha256=digest(SOURCE/"source_integrity.json"),
    main_models=60,models_per_method=20,technical_preflights=3,
    matching_original_initializations=60,paired_sample_orders_match=True,
    original_scientific_code_copied_byte_for_byte=True,training_targets_not_loaded_by_probe=True,
    no_GPU_used=True,test_targets_read=False,elapsed_seconds=time.monotonic()-start,
    fixture_proofs=fixtures,initialization_proofs=probes)
(RUN/"cpu_gate.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k not in ["fixture_proofs","initialization_proofs"]}),flush=True)
