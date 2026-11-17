import base64
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    spec=importlib.util.spec_from_file_location('remote_deploy',HERE.parent/'architecture_main_20261111_v1/deploy.py')
    remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
    files={n:(HERE/n).read_bytes().replace(b'\r\n',b'\n') for n in ['verify_fit.py','run.sbatch','launch.py']}
    hashes={n:hashlib.sha256(v).hexdigest() for n,v in files.items()};version=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    program=r'''import sys,json,pathlib,hashlib,base64,subprocess,fcntl
p=json.load(sys.stdin);root=pathlib.Path('/mnt/project/perun2601449/dp_llm_cfd');source=root/'test/architecture_fit_diagnostic_20261111_v1/revisions'/('r_'+p['version']);source.mkdir(parents=True,exist_ok=True)
for name,data in p['files'].items():
 value=base64.b64decode(data);target=source/name
 if target.exists():assert target.read_bytes()==value
 else:target.write_bytes(value)
 assert hashlib.sha256(value).hexdigest()==p['sha256'][name]
 if name.endswith('.py'):compile(value.decode(),str(target),'exec')
subprocess.run(['bash','-n',str(source/'run.sbatch')],check=True)
output=root/'results/architecture_fit_diagnostic_20261111_v1';output.mkdir(exist_ok=True);receipt=output/('submission_'+p['version']+'.json')
with (output/'submission.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if receipt.exists():result=json.loads(receipt.read_text());assert result['source']==str(source)
 else:
  r=subprocess.run(['sbatch','--parsable','--chdir='+str(source),'--job-name=dp-afit','run.sbatch'],cwd=source,capture_output=True,text=True,check=True)
  job=r.stdout.strip().split(';')[0];assert job.isdigit();result=dict(source=str(source),job_id=job,source_sha256=p['sha256'],no_new_training=True);receipt.write_text(json.dumps(result,indent=2))
print(json.dumps(result))
'''
    result=json.loads(remote.ssh_python(program,dict(version=version,files={n:base64.b64encode(v).decode() for n,v in files.items()},sha256=hashes)))
    (HERE/'submission.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
