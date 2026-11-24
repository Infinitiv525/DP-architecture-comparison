"""Persist scheduler runtime at step closeout before short-lived records expire."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time


def main(root,mode,code):
    job=os.environ['SLURM_JOB_ID'];index=os.environ['SLURM_ARRAY_TASK_ID']
    r=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True)
    value=dict(job_id=job,array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),index=int(index),
        mode=mode,worker_exit_code=code,scheduler_stdout=r.stdout,scheduler_stderr=r.stderr,
        scheduler_returncode=r.returncode,captured_unix=time.time(),allocated_GPUs=1,allocated_CPUs=4,
        timing_scope='At worker step closeout while allocation may still be RUNNING; excludes tiny final batch teardown. Not final portal billing.')
    p=Path(root)/'allocation_closeout';p.mkdir(parents=True,exist_ok=True)
    target=p/(mode+'_'+index+'.json');tmp=target.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(target)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--mode',required=True)
    p.add_argument('--exit-code',type=int,required=True);a=p.parse_args();main(a.root,a.mode,a.exit_code)
