"""Exactly two one-hour H200 trials for each family, after clock integration checks."""
import json
from pathlib import Path
from deploy import ssh_python

HERE=Path(__file__).resolve().parent


def main():
    smoke=json.loads((HERE/'smoke_submission.json').read_text())
    program=r'''import sys,json,pathlib,subprocess,hashlib,fcntl
p=json.load(sys.stdin);source=pathlib.Path(p['source']);smoke=pathlib.Path(p['run_root']);root=pathlib.Path('/mnt/project/perun2601449/dp_llm_cfd')
for name,expected in p['sha256'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==expected
names=['tfno','chebkan','pinet','dendrite','realnpu','rbf','elm','scn','cascor']
for name in names:
 r=json.loads((smoke/'smoke'/(name+'_smoke')/'complete.json').read_text())
 assert r['status']=='BUDGET_WORKER_SMOKE_COMPLETE' and r['within_time_limit'] and not r['test_targets_read']
 assert r['limit_seconds']==60 and len(r['complete_native_TRAIN_case_visits'])==640
 log=root/'logs'/('dp-budget-smoke_'+p['job_id']+'_'+str(names.index(name))+'.err')
 assert not log.read_text().strip()
run=root/'results/architecture_budgeted_20261111_v3'/('tuning_'+source.name.removeprefix('r_'));run.mkdir(parents=True,exist_ok=True)
receipt=run/'submission.json'
with (run/'submission.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if receipt.exists():result=json.loads(receipt.read_text())
 else:
  cmd=['sbatch','--parsable','--chdir='+str(source),'--job-name=dp-budget-tune','--partition=gpu_short','--time=01:00:00','--gres=gpu:1','--array=0-17%4','--dependency=afterok:'+p['job_id'],'--output='+str(root/'logs/%x_%A_%a.out'),'--error='+str(root/'logs/%x_%A_%a.err'),'run.sbatch','tuning',str(run)]
  r=subprocess.run(cmd,cwd=source,capture_output=True,text=True,check=True);job=r.stdout.strip().split(';')[0];assert job.isdigit()
  result=dict(status='EQUAL_BUDGET_TUNING_SUBMITTED',source=str(source),run_root=str(run),sha256=p['sha256'],job_id=job,array='0-17%4',per_task_seconds=3600,maximum_GPU_hours=18,CPUs_per_task=4,smoke_job=p['job_id'],main_started=False)
  receipt.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
'''
    result=json.loads(ssh_python(program,smoke))
    (HERE/'tuning_submission.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
