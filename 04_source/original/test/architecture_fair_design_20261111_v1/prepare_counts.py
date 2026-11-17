"""Upload draft source and construct untrained CPU objects for accounting only."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    spec=importlib.util.spec_from_file_location('count_remote',HERE.parent/'architecture_final_20261111_v1/deploy.py')
    remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
    files={name:(HERE/name).read_bytes().replace(b'\r\n',b'\n') for name in ['field_adapter.py','operator_models.py','capacity_plan.json','count_constructors.py']}
    hashes={name:hashlib.sha256(value).hexdigest() for name,value in files.items()}
    version=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    program=r'''import sys,json,pathlib,hashlib,base64,subprocess
p=json.load(sys.stdin);root=pathlib.Path('/mnt/project/perun2601449/dp_llm_cfd');folder=root/'test/architecture_fair_design_20261111_v1/revisions'/('r_'+p['version']);folder.mkdir(parents=True,exist_ok=True)
for name,data in p['files'].items():
 assert pathlib.Path(name).name==name
 value=base64.b64decode(data);target=folder/name
 if target.exists():assert target.read_bytes()==value
 else:target.write_bytes(value)
 assert hashlib.sha256(value).hexdigest()==p['sha256'][name]
 if name.endswith('.py'):compile(value.decode(),str(target),'exec')
r=subprocess.run([str(root/'envs/architecture-pilot-20261110-v1/bin/python'),'-B',str(folder/'count_constructors.py')],cwd=folder,capture_output=True,text=True,check=True)
result=json.loads(r.stdout);result.update(source=str(folder),source_sha256=p['sha256']);print(json.dumps(result))
'''
    result=json.loads(remote.ssh_python(program,dict(version=version,files={name:base64.b64encode(value).decode() for name,value in files.items()},sha256=hashes)))
    (HERE/'constructor_counts.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for row in result['models']:print(row['model'],row['ledger']['total_core_coefficients_plus_normalization'],row['ledger']['learned_or_fitted_total'])
    print(result['status'],'forward calls:',result['forward_calls'],'training updates:',result['training_updates'],'GPU allocated:',result['GPU_allocated'])


if __name__=='__main__':main()
