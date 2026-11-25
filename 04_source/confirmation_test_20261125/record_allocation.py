import json,os,sys,subprocess,time
from pathlib import Path
root=Path(sys.argv[1]);role=sys.argv[2];index=sys.argv[3];code=int(sys.argv[4])
q=subprocess.run(['scontrol','show','job','-o',os.environ['SLURM_JOB_ID']],capture_output=True,text=True)
out=root/'allocation_closeout';out.mkdir(exist_ok=True)
(out/(role+'_'+index+'.json')).write_text(json.dumps(dict(job_id=os.environ['SLURM_JOB_ID'],role=role,index=index,
 worker_exit_code=code,allocated_GPUs=1 if role in ['preflight','test'] else 0,allocated_CPUs=4,
 captured_unix=time.time(),scheduler_returncode=q.returncode,scheduler_stdout=q.stdout,scheduler_stderr=q.stderr,
 closeout_is_final_portal_billing=False),indent=2)+'\n')
