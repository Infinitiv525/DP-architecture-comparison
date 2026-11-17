"""Create the final50-model manifest after all45 TRAIN/VAL runs have passed."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'architecture_main_20261111_v1'))
from deploy import ssh_python


def main():
    if (HERE/'models.json').exists():raise RuntimeError('Final manifest already frozen; do not replace it')
    main_folder=HERE.parent/'architecture_main_20261111_v1'
    payload=dict(receipt=json.loads((main_folder/'submission.json').read_text()),source=json.loads((main_folder/'source_upload.json').read_text()))
    program=r'''import sys,json,pathlib,hashlib,ast,shutil,time
p=json.load(sys.stdin);root=pathlib.Path('/mnt/project/perun2601449/dp_llm_cfd');source=pathlib.Path(p['source']['source']);run=pathlib.Path(p['receipt']['run_root'])
def read(path):return json.loads(pathlib.Path(path).read_text())
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as stream:
  for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
for name,h in p['source']['sha256'].items():assert sha(source/name)==h
selection=read(source/'selection.json');validated=read(run/'validation/summary.json')
assert validated['status']=='ARCHITECTURE_MAIN_VALIDATION_COMPLETE' and validated['completed_models']==45
assert not validated['test_targets_read'] and len(selection['tasks'])==45
metadata={str(source/'selection.json'):sha(source/'selection.json'),str(run/'validation/summary.json'):sha(run/'validation/summary.json')};models=[]
for task in selection['tasks']:
 path=run/'models'/task['id']/'summary.json';s=read(path);assert s['status']=='ARCHITECTURE_MAIN_MODEL_COMPLETE' and s['task']==task
 v=next(r for r in validated['models'] if r['id']==task['id']);assert v['status']=='MAIN_VALIDATION_PASS'
 best=s['best'];assert sha(best['checkpoint'])==best['checkpoint_sha256']==v['checkpoint_sha256']
 score=run/'validation'/task['id']/'validation.json';metadata[str(path)]=sha(path);metadata[str(score)]=sha(score)
 reference=read(score)
 models.append(dict(id=task['id'],model=task['model'],seed=task['seed'],kind='new_architecture',checkpoint=best['checkpoint'],checkpoint_sha256=best['checkpoint_sha256'],
  expected_validation=[dict(index=r['index'],mse=r['mse'],velocity_l2=r['velocity_l2'],pressure_l2=r['pressure_l2']) for r in reference['cases'][:2]],counts=v['counts']))
old_path=pathlib.Path(selection['archived_pinn_manifest']);assert sha(old_path)==selection['archived_pinn_manifest_sha256'];metadata[str(old_path)]=sha(old_path)
old=read(old_path)
for selected in selection['archived_pinn']:
 r=next(r for r in old['models'] if r['id']==selected['id']);assert r['kind']=='fixed' and sha(r['checkpoint'])==selected['checkpoint_sha256']==r['checkpoint_sha256']
 score=pathlib.Path(r['checkpoint']).parent/'selected_validation/summary.json';reference=read(score);metadata[str(score)]=sha(score)
 split=read(root/'results/ldc_pinn_prepare_20261104_v25/split.json')['indices']
 expected=[]
 for index in split['validation'][:2]:
  case=next(c for c in reference['cases'] if c['sample']==index)
  expected.append(dict(index=index,mse=case['mse_all'],velocity_l2=case['velocity_relative_l2'],pressure_l2=case['pressure_relative_l2']))
 models.append(dict(id=r['id'],model='pinn',seed=r['config']['seed'],kind='archived_auxiliary_pinn',checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],config=r['config'],expected_validation=expected))
assert len(models)==50 and len({r['id'] for r in models})==50
prepared_path=root/'results/ldc_pinn_prepare_20261104_v25/prepared.json';prepared=read(prepared_path);cache=pathlib.Path(prepared['cache']);split_path=prepared_path.parent/'split.json'
for path in [prepared_path,split_path,cache/'normalization.json',cache/'inputs.json']:metadata[str(path)]=sha(path)
files=[cache/'fluid.npy',cache/'Y_validation.npy',cache.parent/'raw/X_original.npy',pathlib.Path(prepared['rawtest_targets_sealed_in'])]
def fingerprint(path):
 s=path.stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino)
fingerprints={str(path):fingerprint(path) for path in files}
# Freeze the local-import closure used by the legacy shared PINN/query and target canonicalizer.
folders=[root/'test/analysis',root/'test/llm',root/'test/training'];available={}
for folder in folders:
 for path in folder.glob('*.py'):available[path.stem]=path
todo=['ldc_shared_pinn_v26','ldc_store_v26','ldc_data_v25'];seen={}
while todo:
 name=todo.pop()
 if name in seen or name not in available:continue
 path=available[name];seen[name]=path
 for node in ast.walk(ast.parse(path.read_text())):
  names=[a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else ([node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else [])
  todo.extend(n for n in names if n in available and n not in seen)
legacy={str(path):sha(path) for path in seen.values()}
old_freeze=read(root/'results/ldc_main_20261104_v33/freeze.json')['files']
for path,h in legacy.items():
 if path in old_freeze:assert h==old_freeze[path]
raw=pathlib.Path(prepared['rawtest_targets_sealed_in']);old_final=read(root/'results/ldc_final_test_20261108_v49/freeze.json')
assert fingerprint(raw)==old_final['input_fingerprints'][str(raw)]
assert shutil.disk_usage(root).free>40*1024**3
value=dict(status='FINAL50_MODEL_MANIFEST_FROZEN',models=models,full_test_cases=200,selection_uses_test=False,test_previously_opened_in_old_study=True,created_unix=time.time())
config=dict(name='architecture_final_20261111_v1',root=str(root),main_source=str(source),main_source_sha256=p['source']['sha256'],main_run=str(run),
 legacy_paths=[str(f) for f in folders],legacy_source_sha256=legacy,metadata_sha256=metadata,data_fingerprints=fingerprints,
 prepared_path=str(prepared_path),split_path=str(split_path),latency_warmup=100,latency_repetitions=1000,latency_target_ms=5.,
 preflight_validation_cases=2,preflight_warmup=10,preflight_repetitions=20,save_all_native_predictions=True,
 prediction_shape=[200,512,512,3],precision='float32, TF32 disabled for matmul and cudnn',
 timing_scope='serialized batch1 full native field with inputs resident on GPU; record CUDA events and synchronized instrumented host wall time; no input-dependent feature caching',
 accuracy_metric='same native512x512 physical all-cell u/v/p MSE, fluid relativeL2 and frozen FD physics diagnostics for every method',
 test_targets_allowed_only_after_validation_preflight=True)
print(json.dumps(dict(manifest=value,config=config)))
'''
    result=json.loads(ssh_python(program,payload));manifest=json.dumps(result['manifest'],indent=2,allow_nan=False)+'\n'
    (HERE/'models.json').write_text(manifest,encoding='utf-8',newline='\n');config=result['config']
    config['models_sha256']=hashlib.sha256(manifest.encode()).hexdigest()
    (HERE/'config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('FROZEN:',len(result['manifest']['models']),'models; legacy source files:',len(config['legacy_source_sha256']))


if __name__=='__main__':main()
