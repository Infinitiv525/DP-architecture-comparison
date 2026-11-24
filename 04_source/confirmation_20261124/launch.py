"""Route one immutable fixed-configuration task to an unchanged original worker."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
SOURCE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(mode,index,root):
    assert root==Path("/mnt/project/perun2601449/dp_llm_cfd/results/architecture_confirmation_20261124_v2")
    integrity=read(SOURCE/"source_integrity.json")
    for name,sha in integrity["files"].items():
        assert digest(SOURCE/name)==sha,name
    assert read(root/"cpu_gate.json")["status"]=="CPU_REFERENCE_PARITY_AND_SETUP_PASS"
    assert read(root/"cpu_gate.json")["source_integrity_sha256"]==digest(SOURCE/"source_integrity.json")
    assert (root/"selection.json").read_bytes()==(SOURCE/"selection.json").read_bytes()
    row=read(SOURCE/"selection.json")["tasks"][index]
    assert row["index"]==index and row["model"] in ["tfno","fno","realnpu"]
    if mode=="main":
        assert 0<=index<60 and row["role"]=="independent_main_training"
        gate=read(root/"preflight_gate.json")
        assert gate["status"]=="THREE_AUXILIARY_H200_PREFLIGHTS_PASS"
        assert gate["source_integrity_sha256"]==digest(SOURCE/"source_integrity.json")
    else:
        assert mode=="preflight" and 60<=index<=62 and row["role"]=="technical_preflight_only"
    kind="dense" if row["model"]=="fno" else "primary"
    route=root/"routing";route.mkdir(exist_ok=True)
    value=dict(mode=mode,index=index,task=row,worker=str(SOURCE/kind/"worker.py"),
        worker_sha256=digest(SOURCE/kind/"worker.py"),source_integrity_sha256=digest(SOURCE/"source_integrity.json"),
        job_id=os.environ.get("SLURM_JOB_ID"),started_unix=time.time(),warm_start=False)
    target=route/(mode+"_"+str(index)+".json");assert not target.exists()
    target.write_text(json.dumps(value,indent=2)+"\n")
    env=os.environ.copy();env["DENSE_FNO_RUN_ROOT"]=str(root)
    result=subprocess.run([sys.executable,"-B","-u",str(SOURCE/kind/"worker.py"),
        "--mode",mode,"--index",str(index),"--root",str(root)],env=env)
    if result.returncode:return result.returncode
    done=read(root/mode/row["id"]/"complete.json")
    assert done["task"]["seed"]==row["seed"] and done["task"]["model"]==row["model"]
    assert done["task"]["configuration_index"]==row["configuration_index"]
    assert done["task"]["time_budget_seconds"]==(14400 if mode=="main" else 60)
    assert done["test_targets_read"] is False and done["within_time_limit"]
    assert len(done["complete_native_TRAIN_case_visits"])==640
    assert done["selected_checkpoint_sha256"]==digest(root/mode/row["id"]/"best.pt")
    return 0
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=["main","preflight"],required=True)
    p.add_argument("--index",type=int,required=True);p.add_argument("--root",type=Path,required=True)
    a=p.parse_args();sys.exit(main(a.mode,a.index,a.root))
