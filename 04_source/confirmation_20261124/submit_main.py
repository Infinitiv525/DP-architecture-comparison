"""Submit and audit the explicitly authorized60-model extension, without silent retries."""
import argparse,fcntl,hashlib,json,math,subprocess,sys
from pathlib import Path
SOURCE=Path(__file__).resolve().parent
ROOT=Path("/mnt/project/perun2601449/dp_llm_cfd")
RUN=ROOT/"results/architecture_confirmation_20261124_v2"
def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+"\n")
def finite(v):
    if isinstance(v,dict):return all(finite(x) for x in v.values())
    if isinstance(v,list):return all(finite(x) for x in v)
    return not isinstance(v,(int,float)) or math.isfinite(v)
def submit(action):
    with (RUN/"submission.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        receipt=read(RUN/"submission.json")
        if action=="main":
            if "main" in receipt["jobs"]:
                print(json.dumps(receipt));return
            assert read(RUN/"cpu_gate.json")["status"]=="CPU_REFERENCE_PARITY_AND_SETUP_PASS"
            plan=read(SOURCE/"plan.json");tasks=read(SOURCE/"selection.json")["tasks"];checks=[]
            for row in tasks[60:]:
                value=read(RUN/"preflight"/row["id"]/"complete.json")
                close=read(RUN/"allocation_closeout"/("preflight_"+str(row["index"])+".json"))
                assert close["worker_exit_code"]==0 and value["status"]=="MAIN_PREFLIGHT_COMPLETE"
                assert value["task"]["model"]==row["model"] and value["task"]["seed"]==row["seed"]
                assert value["task"]["configuration_index"]==row["configuration_index"]
                assert value["task"]["time_budget_seconds"]==60 and value["within_time_limit"]
                assert "H200" in value["hardware"] and value["test_targets_read"] is False
                assert finite(value["physical_validation_metrics"]) and math.isfinite(value["best_complete_validation_loss"])
                assert len(value["physical_validation_metrics"]["case_metrics"])==160
                assert value["selected_checkpoint_sha256"]==digest(RUN/"preflight"/row["id"]/"best.pt")
                checks.append(dict(id=row["id"],seed=row["seed"],model=row["model"],
                    initial_check_sha256=digest(RUN/"preflight"/row["id"]/"complete.json")))
            gate=dict(status="THREE_AUXILIARY_H200_PREFLIGHTS_PASS",models=checks,
                source_integrity_sha256=digest(SOURCE/"source_integrity.json"),accuracy_threshold=None,
                technical_seeds_excluded_from_main=True,test_targets_read=False)
            write(RUN/"preflight_gate.json",gate)
            if (RUN/"main").exists():assert not list((RUN/"main").glob("*/started.json"))
            cmd=["sbatch","--parsable","--chdir="+str(SOURCE),"--job-name=dp-confirm60","--partition=gpu_short",
                "--time=04:00:00","--gres=gpu:1","--array=0-59%24",
                "--output="+str(ROOT/"logs/dp-confirm60_%A_%a.out"),
                "--error="+str(ROOT/"logs/dp-confirm60_%A_%a.err"),str(SOURCE/"run.sbatch"),"main",str(RUN)]
            p=subprocess.run(cmd,cwd=SOURCE,capture_output=True,text=True);assert p.returncode==0,(p.stdout,p.stderr)
            job=p.stdout.strip().split(";")[0];assert job.isdigit()
            receipt["jobs"]["main"]=job;receipt["status"]="ALL60_NEW_TRAININGS_SUBMITTED"
            receipt.update(main_submitted=True,models_per_method=20,main_seed_blocks=20,maximum_concurrency=24,
                main_training_GPU_hours_hard_limit=240,overall_planning_budget_GPU_hours=260,
                source_integrity_sha256=digest(SOURCE/"source_integrity.json"))
            write(RUN/"submission.json",receipt)
        if action in ["main","audit"]:
            assert "main" in receipt["jobs"]
            if "completion_audit" not in receipt["jobs"]:
                cmd=["sbatch","--parsable","--account=perun2601449","--partition=cpu_short",
                    "--nodes=1","--ntasks=1","--cpus-per-task=4","--mem=8G","--time=00:30:00",
                    "--job-name=dp-confirm60-close","--chdir="+str(SOURCE),
                    "--dependency=afterany:"+receipt["jobs"]["main"],
                    "--output="+str(ROOT/"logs/dp-confirm60-close_%j.out"),
                    "--error="+str(ROOT/"logs/dp-confirm60-close_%j.err"),
                    str(SOURCE/"completion_audit.sbatch")]
                p=subprocess.run(cmd,cwd=SOURCE,capture_output=True,text=True);assert p.returncode==0,(p.stdout,p.stderr)
                receipt["jobs"]["completion_audit"]=p.stdout.strip().split(";")[0]
                write(RUN/"submission.json",receipt)
        print(json.dumps(receipt),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["main","audit"]);a=p.parse_args();submit(a.action)
