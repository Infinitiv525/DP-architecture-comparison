"""Respect Slurm's50-submitted/20-running limit without changing any model task."""
import argparse,fcntl,hashlib,json,subprocess
from pathlib import Path
SOURCE=Path(__file__).resolve().parent
ROOT=Path("/mnt/project/perun2601449/dp_llm_cfd")
RUN=ROOT/"results/architecture_confirmation_20261124_v2"
def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(value):
    target=RUN/"submission.json";tmp=target.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,indent=2)+"\n");tmp.replace(target)
def submit(cmd):
    p=subprocess.run(cmd,cwd=SOURCE,capture_output=True,text=True)
    if p.returncode:raise RuntimeError(p.stderr+p.stdout)
    job=p.stdout.strip().split(";")[0];assert job.isdigit();return job
def run(wave):
    with (RUN/"schedule.lock").open("a") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        layout=read(SOURCE/"scheduling_plan.json")
        for name,sha in layout["files"].items():assert digest(SOURCE/name)==sha,name
        assert digest(SOURCE/"source_integrity.json")==layout["training_source_integrity_sha256"]
        assert read(RUN/"cpu_gate.json")["status"]=="CPU_REFERENCE_PARITY_AND_SETUP_PASS"
        assert read(RUN/"preflight_gate.json")["status"]=="THREE_AUXILIARY_H200_PREFLIGHTS_PASS"
        assert read(RUN/"preflight_gate.json")["source_integrity_sha256"]==layout["training_source_integrity_sha256"]
        receipt=read(RUN/"submission.json");jobs=receipt["jobs"]
        assert "main" not in jobs, "Do not mix with a previous monolithic submission."
        key="main_wave"+str(wave+1)
        if wave==1:assert "main_wave1" in jobs
        if key not in jobs:
            bounds=layout["waves"][wave]
            array=str(bounds[0])+"-"+str(bounds[1])+"%20"
            jobs[key]=submit(["sbatch","--parsable","--chdir="+str(SOURCE),
                "--job-name=dp-confirm60-w"+str(wave+1),"--partition=gpu_short","--time=04:00:00",
                "--gres=gpu:1","--array="+array,
                "--output="+str(ROOT/"logs/dp-confirm60_%A_%a.out"),
                "--error="+str(ROOT/"logs/dp-confirm60_%A_%a.err"),
                str(SOURCE/"run.sbatch"),"main",str(RUN)])
            receipt.update(status="CONFIRMATION60_AUTOMATIC_WAVES_ACTIVE",main_submitted=True,
                main_models=60,models_per_method=20,maximum_running_model_jobs=20,
                all60_tasks_authorized_and_in_workflow=True,
                unsubmitted_tasks_automatically_released_after_first_wave=20,
                main_training_GPU_hours_hard_limit=240,overall_planning_budget_GPU_hours=260)
            save(receipt)
        if wave==0:
            if "release_wave2" not in jobs:
                jobs["release_wave2"]=submit(["sbatch","--parsable","--chdir="+str(SOURCE),
                    "--job-name=dp-confirm60-release","--dependency=afterany:"+jobs[key],
                    "--output="+str(ROOT/"logs/dp-confirm60-release_%j.out"),
                    "--error="+str(ROOT/"logs/dp-confirm60-release_%j.err"),
                    str(SOURCE/"release_wave2.sbatch")])
                save(receipt)
        elif "completion_audit" not in jobs:
            jobs["completion_audit"]=submit(["sbatch","--parsable","--account=perun2601449",
                "--partition=cpu_short","--nodes=1","--ntasks=1","--cpus-per-task=4","--mem=8G",
                "--time=00:30:00","--job-name=dp-confirm60-close","--chdir="+str(SOURCE),
                "--dependency=afterany:"+jobs[key],
                "--output="+str(ROOT/"logs/dp-confirm60-close_%j.out"),
                "--error="+str(ROOT/"logs/dp-confirm60-close_%j.err"),
                str(SOURCE/"completion_audit.sbatch")])
            receipt["unsubmitted_tasks_automatically_released_after_first_wave"]=0
            save(receipt)
        print(json.dumps(receipt),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--wave",type=int,choices=[0,1],required=True)
    a=p.parse_args();run(a.wave)
