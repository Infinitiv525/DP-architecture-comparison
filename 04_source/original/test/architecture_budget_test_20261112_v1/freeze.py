"""Freeze all45 complete models and5 archived PINN references before TEST access."""
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def ssh_python(program,payload=None):
    spec=importlib.util.spec_from_file_location('final_remote',HERE.parent/'architecture_final_20261111_v1/deploy.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.ssh_python(program,payload)


def main():
    main=json.loads((HERE.parent/'architecture_budget_main_20261111_v1/submission.json').read_text())
    old=json.loads((HERE.parent/'architecture_final_20261111_v1/config.json').read_text())
    archived=json.loads((HERE.parent/'architecture_final_20261111_v1/models.json').read_text())['models']
    references=[r for r in archived if r['model']=='pinn']
    program=r'''import sys,json,pathlib,hashlib,subprocess
p=json.load(sys.stdin);root=pathlib.Path('/mnt/project/perun2601449/dp_llm_cfd');source=pathlib.Path(p['main']['source']);run=pathlib.Path(p['main']['run_root'])
def read(path):return json.loads(pathlib.Path(path).read_text())
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def fingerprint(path):
 s=pathlib.Path(path).stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino)
for name,expected in p['main']['sha256'].items():assert sha(source/name)==expected
policy=read(source/'policy.json');selection=read(source/'selection.json')
assert len(selection['tasks'])==45 and len(list((run/'main').glob('*/complete.json')))==45
split_path=root/'results/ldc_pinn_prepare_20261104_v25/split.json';prepared_path=split_path.parent/'prepared.json'
split=read(split_path)['indices'];assert [len(split[x]) for x in ['train','validation','test']]==[640,160,200]
validation_ids=split['validation'];metadata={str(prepared_path):sha(prepared_path),str(split_path):sha(split_path)}
metadata[str(source/'selection.json')]=sha(source/'selection.json')
models=[];closed=[]
for index,task in enumerate(selection['tasks']):
 path=run/'main'/task['id']/'complete.json';r=read(path);close=read(run/'allocation_closeout'/('main_'+str(index)+'.json'))
 assert r['status']=='BUDGETED_TRAINING_COMPLETE' and r['within_time_limit'] and r['limit_seconds']==14400
 assert not r['test_targets_read'] and r['accuracy_threshold'] is None and r['hardware']=='NVIDIA H200'
 assert r['task']['model']==task['model'] and r['task']['seed']==task['seed'] and r['task']['configuration_index']==task['configuration_index']
 assert close['worker_exit_code']==0 and close['index']==index
 assert len(r['complete_native_TRAIN_case_visits'])==640 and min(r['complete_native_TRAIN_case_visits'])>0
 assert len(r['physical_validation_metrics']['case_metrics'])==160
 assert r['best_complete_validation_loss']==min(h['validation']['mean_raw_case_balanced_loss'] for h in r['history'])
 checkpoint=pathlib.Path(r['selected_checkpoint']);assert sha(checkpoint)==r['selected_checkpoint_sha256']
 expected=[]
 for slot,case_id in enumerate(validation_ids[:2]):
  v=r['physical_validation_metrics']['case_metrics'][slot]
  expected.append(dict(index=case_id,velocity_l2=v['velocity_relative_L2'],pressure_l2=v['pressure_relative_L2']))
 models.append(dict(id=task['id'],model=task['model'],seed=task['seed'],group=task['seed']-931101,
  kind='primary_resource_matched_model',checkpoint=str(checkpoint),checkpoint_sha256=r['selected_checkpoint_sha256'],
  expected_validation=expected,parameter_ledger=r['selected_parameter_ledger'],training_summary=str(path),training_summary_sha256=sha(path),
  training_elapsed_seconds=r['allocation_elapsed_seconds'],training_stop_reason=r['stop_reason']))
 metadata[str(path)]=sha(path);metadata[str(run/'allocation_closeout'/('main_'+str(index)+'.json'))]=sha(run/'allocation_closeout'/('main_'+str(index)+'.json'))
 closed.append(close)
for old in p['references']:
 assert sha(old['checkpoint'])==old['checkpoint_sha256']
 row=dict(old);row.update(group=old['seed']-150001,kind='archived_auxiliary_PINN_reference')
 models.append(row)
assert len(models)==50 and len({r['id'] for r in models})==50
assert all(len([r for r in models if r['group']==g])==10 for g in range(5))
assert all(len([r for r in models if r['model']==name])==5 for name in {r['model'] for r in models})
cache=pathlib.Path(read(prepared_path)['cache'])
for path in [cache/'normalization.json',cache/'inputs.json']:metadata[str(path)]=sha(path)
frozen_metrics=root/'test/architecture_pilot_20261110_v1/revisions/r_e9dafab653bac9e5/data_metrics.py'
assert sha(frozen_metrics)=='3a004a845e1bc83f9fea41db3732b494a132358e9b24838bd6cdaf52dc75458a'
fields=pathlib.Path(policy['dependency_source']);field_names=['field_adapter.py','operator_models.py']
for name in field_names:assert sha(fields/name)==policy['dependency_sha256'][name]
manifest=dict(status='RESOURCE_MATCHED_FINAL50_MANIFEST_FROZEN',models=models,primary_models=45,auxiliary_PINN_models=5,
 train_cases=640,validation_cases=160,test_cases=200,selection_uses_test=False,
 test_previously_opened_in_old_studies=True,all_main_worker_exit_codes_zero=True,
 main_GPU_process_hours=sum(r['training_elapsed_seconds'] for r in models if r['model']!='pinn')/3600,
 groups='Five seed groups, each10families serialized on oneH200; no concurrent forwards within one GPU')
data_files={path:value for path,value in p['old']['data_fingerprints'].items()}
for path,value in data_files.items():assert fingerprint(path)==value
config=dict(name='architecture_budget_test_20261112_v1',root=str(root),
 main_source=str(source),main_source_sha256=p['main']['sha256'],main_run=str(run),
 field_source=str(fields),field_source_sha256={name:sha(fields/name) for name in field_names},
 metric_source=str(frozen_metrics),metric_source_sha256=sha(frozen_metrics),
 legacy_paths=p['old']['legacy_paths'],legacy_source_sha256=p['old']['legacy_source_sha256'],
 metadata_sha256=metadata,data_fingerprints=data_files,prepared_path=str(prepared_path),split_path=str(split_path),
 latency_warmup=100,latency_repetitions=1000,latency_target_ms=5.,preflight_validation_cases=2,
 preflight_warmup=10,preflight_repetitions=20,save_all_native_predictions=True,prediction_shape=[200,512,512,3],
 precision='FP32 with matmul and cuDNN TF32 disabled',
 timing_scope=p['old']['timing_scope'],accuracy_metric=p['old']['accuracy_metric'],
 test_targets_allowed_only_after_validation_preflight=True,test_group_count=5,
 preflight_GPU_time_limit_seconds=1800,test_group_GPU_time_limit_seconds=42840,
 total_GPU_allocation_hours_limit=60,notes='0.5h preflight + five11.9h test allocations; CPU saved-field audit separate.')
for path,expected in config['legacy_source_sha256'].items():assert sha(path)==expected
print(json.dumps(dict(manifest=manifest,config=config,closeout_count=len(closed))))
'''
    result=json.loads(ssh_python(program,dict(main=main,old=old,references=references)))
    manifest_path=HERE/'models.json'
    text=json.dumps(result['manifest'],indent=2)+'\n'
    if manifest_path.exists():assert manifest_path.read_text(encoding='utf-8')==text,'Existing different final manifest requires explicit review'
    manifest_path.write_text(text,encoding='utf-8',newline='\n')
    import hashlib
    config=result['config'];config['models_sha256']=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (HERE/'config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Frozen',len(result['manifest']['models']),'models; all45 closeouts verified; no TEST targets read.')
    print('Main GPU-process-hours:',result['manifest']['main_GPU_process_hours'])


if __name__=='__main__':main()
