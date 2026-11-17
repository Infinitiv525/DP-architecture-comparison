"""Bounded full-data pilot after both paired one-case benchmarks completed."""
import base64
import hashlib
import json
from pathlib import Path
from deploy import ssh_python

HERE=Path(__file__).resolve().parent
NAMES=['common.py','field_adapter.py','operator_models.py','capacity_plan.json','weighted_fit.py',
       'benchmark.py','prepare.py','train.py','smoke.py','run.sbatch','pilot_tasks.json','resource_decision.json']


def main():
    benchmark=json.loads((HERE/'submission.json').read_text())
    files={n:(HERE/n).read_bytes().replace(b'\r\n',b'\n') for n in NAMES}
    hashes={n:hashlib.sha256(v).hexdigest() for n,v in files.items()}
    version=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    receipt=HERE/'pilot_submission.json'
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if not old['source'].endswith('/r_'+version):
            raise RuntimeError('Existing different pilot submission: preserve and review before any new launch')
    program=r'''import sys,json,pathlib,base64,hashlib,subprocess,fcntl,time
p=json.load(sys.stdin);root=pathlib.Path('/mnt/project/perun2601449/dp_llm_cfd')
bench=pathlib.Path(p['benchmark_root'])
for device in ['cpu','cuda']:
 ready=json.loads((bench/('benchmark_'+device)/'complete.json').read_text())
 assert ready['status']=='ALL9_ONE_CASE_BENCHMARK_COMPLETE' and not ready['test_targets_read']
 for name in ready['models']:
  row=json.loads((bench/('benchmark_'+device)/(name+'.json')).read_text())
  assert row['status']=='ONE_CASE_TIMING_AND_TECHNICAL_CHECKS_COMPLETE'
source=root/'test/architecture_fair_pilot_20261111_v2/revisions'/('r_'+p['version']);source.mkdir(parents=True,exist_ok=True)
for name,data in p['files'].items():
 assert pathlib.Path(name).name==name
 value=base64.b64decode(data);path=source/name
 if path.exists():assert path.read_bytes()==value
 else:path.write_bytes(value)
 assert hashlib.sha256(value).hexdigest()==p['sha256'][name]
 if name.endswith('.py'):compile(value.decode(),str(path),'exec')
subprocess.run(['bash','-n',str(source/'run.sbatch')],check=True)
tasks=json.loads((source/'pilot_tasks.json').read_text())['tasks'];assert len(tasks)==18
assert len([t for t in tasks if t['device']=='cuda'])==4
run=root/'results/architecture_fair_pilot_20261111_v2'/('pilot_'+p['version']);run.mkdir(parents=True,exist_ok=True)
receipt=run/'submission.json'
with (run/'submission.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 result=json.loads(receipt.read_text()) if receipt.exists() else dict(source=str(source),run_root=str(run),sha256=p['sha256'],jobs={},benchmark_root=str(bench))
 def store():receipt.write_text(json.dumps(result,indent=2)+'\n')
 def submit(role,device,limit,extra):
  if role in result['jobs']:return
  cmd=['sbatch','--parsable','--chdir='+str(source),'--job-name=dp-fair-'+role,'--partition='+('gpu_short' if device=='cuda' else 'cpu_short'),'--time='+limit,'--output='+str(root/'logs/%x_%A_%a.out'),'--error='+str(root/'logs/%x_%A_%a.err')]
  if device=='cuda':cmd+=['--gres=gpu:1']
  cmd+=extra+['run.sbatch',role,str(run)]
  r=subprocess.run(cmd,cwd=source,capture_output=True,text=True,check=True);job=r.stdout.strip().split(';')[0];assert job.isdigit()
  result['jobs'][role]=dict(job_id=job,device=device,cpus=4,gpus=int(device=='cuda'),time_limit=limit,submitted_unix=time.time());store()
 submit('prepare','cpu','02:00:00',[])
 dep=['--dependency=afterok:'+result['jobs']['prepare']['job_id']]
 submit('pilot-cpu','cpu','06:00:00',dep+['--array=0-13%6'])
 submit('pilot-cuda','cuda','04:00:00',dep+['--array=14-17%2'])
 result['status']='BOUNDED_FULL_DATA_PILOT_SUBMITTED';result['automatic_final_launch']=False;store();print(json.dumps(result))
'''
    result=json.loads(ssh_python(program,dict(version=version,sha256=hashes,
        benchmark_root=benchmark['run_root'],files={n:base64.b64encode(v).decode() for n,v in files.items()})))
    if receipt.exists():
        old=json.loads(receipt.read_text())
        assert old['source']==result['source'],'Preserve and explicitly review an existing different submission'
    receipt.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
