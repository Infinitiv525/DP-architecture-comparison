"""Freeze nine recipes only after both independent review jobs have passed."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
REVIEW=HERE.parent/'architecture_review_20261111_v1'
sys.path.insert(0,str(REVIEW))
from deploy import ssh_python


def read(path):return json.loads(path.read_text())


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target=HERE/'selection.json'
    if target.exists():raise RuntimeError('Selection already frozen: inspect it instead of replacing it')
    report=REVIEW/'report';audit=read(report/'audit/summary.json');extended=read(report/'extended_evaluation/summary.json')
    assert audit['status']==extended['status']=='CALIBRATION_REVIEW_COMPLETE'
    assert len(audit['models'])==18 and len(extended['models'])==2
    assert not audit['test_targets_read'] and not extended['test_targets_read']
    rc=read(REVIEW/'config.json');calibration=read(HERE.parent/'architecture_calibration_20261111_v1/config.json')
    assert calibration['final_seeds']==[921101,921102,921103,921104,921105]
    rows=[];evidence=[]
    for role,summary in [('audit',audit),('extended_evaluation',extended)]:
        for row in summary['models']:
            path=report/role/row['id']/'artifact_audit.json';proof=read(path)
            assert proof['status']=='MODEL_ARTIFACT_AUDIT_PASS' and proof['validation_cases_recomputed']==160
            if role=='audit':assert proof['stored_fields_recomputed']==4
            evidence.append(dict(id=role+'/'+row['id'],status=proof['status'],artifact_audit_sha256=sha(path)))
            rows.append(dict(row,review_role=role))
    selected=[]
    for model in dict.fromkeys(r['model'] for r in calibration['recipes']):
        options=[r for r in rows if r['model']==model]
        # For KAN the two declared continued variants supersede the 200-epoch
        # calibration variants. Each continued best retains its epoch200 start.
        if model=='chebkan':options=[r for r in options if r['review_role']=='extended_evaluation']
        feasible=[r for r in options if r['latency_pass']]
        chosen=min(feasible or options,key=lambda r:r['validation_mse'])
        recipe=next(r for r in calibration['recipes'] if r['id']==chosen['id'])
        selected.append(dict(model=model,recipe=recipe,checkpoint=chosen['checkpoint'],checkpoint_sha256=chosen['checkpoint_sha256'],
            calibration_validation_mse=chosen['validation_mse'],review_gpu_p99_ms=chosen['worst_gpu_p99_ms'],
            latency_criterion_met=chosen['latency_pass'],selection_status='FEASIBLE' if feasible else 'BEST_AVAILABLE_UNMET_LATENCY',
            continuation=model=='chebkan',review_role=chosen['review_role'],counts=chosen['counts']))
    program=r'''import json,pathlib,hashlib
root=pathlib.Path('/mnt/project/perun2601449/dp_llm_cfd');path=root/'results/ldc_final_test_20261108_v49/models.json'
data=json.loads(path.read_text());models=[r for r in data['models'] if r['id'] in ['fixed_k00','fixed_k01','fixed_k02','fixed_k03','fixed_k04']]
assert len(models)==5
result=[]
for r in models:
 p=pathlib.Path(r['checkpoint']);h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==r['checkpoint_sha256']
 result.append(dict(id=r['id'],checkpoint=str(p),checkpoint_sha256=h,kind=r['kind'],config=r['config']))
print(json.dumps(dict(models=result,manifest=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
'''
    references=json.loads(ssh_python(program));tasks=[]
    for choice in selected:
        for seed in calibration['final_seeds']:
            tasks.append(dict(index=len(tasks),id=choice['model']+'_s'+str(seed),model=choice['model'],seed=seed,device=choice['recipe']['device']))
    value=dict(status='ARCHITECTURE_SELECTION_FROZEN',selection_rule='minimum full160 VAL physical all-cell MSE among reviewed GPU-p99<=5ms variants; retain best available if none feasible',
        review_evidence=dict(audit_summary_sha256=sha(report/'audit/summary.json'),extended_summary_sha256=sha(report/'extended_evaluation/summary.json')),
        audit_evidence=evidence,selected=selected,final_seeds=calibration['final_seeds'],tasks=tasks,
        archived_pinn=references['models'],archived_pinn_manifest=references['manifest'],archived_pinn_manifest_sha256=references['sha256'],
        test_targets_read_for_selection=False,test_previously_opened_in_old_study=True)
    target.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    config=dict(rc,name='architecture_main_20261111_v1',purpose='five independent full TRAIN/VAL runs for each of nine selected pipelines',
        review_source=read(REVIEW/'source_upload.json')['source'],review_source_sha256=read(REVIEW/'source_upload.json')['sha256'],
        selection_sha256=sha(target),final_repetitions=5,cpu_concurrency=12,gpu_concurrency=5,
        test_targets_allowed=False,automatic_test_evaluation=False)
    (HERE/'config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8',newline='\n')
    for r in selected:print(r['model'],r['recipe']['id'],r['calibration_validation_mse'],r['review_gpu_p99_ms'],r['selection_status'])
    print('FROZEN:45 new models +5 archived auxiliary PINN references; no TEST target access')


if __name__=='__main__':main()
