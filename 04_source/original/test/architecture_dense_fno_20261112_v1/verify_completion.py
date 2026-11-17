"""Verify the five-model FNO supplement using durable reports and remote SHA."""
import csv
from datetime import datetime,timezone,timedelta
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from deploy import ssh_python

HERE=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    receipt=read(HERE/'submission.json');report=HERE/'report'
    for name,value in receipt['sha256'].items():
        assert hashlib.sha256((HERE/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value,name
    freeze=read(report/'frozen_evaluation/freeze_receipt.json')
    assert sha(report/'frozen_evaluation/config.json')==freeze['config_sha256']
    assert sha(report/'frozen_evaluation/models.json')==freeze['models_sha256']
    config=read(report/'frozen_evaluation/config.json');manifest=read(report/'frozen_evaluation/models.json')['models']
    assert len(manifest)==5 and sha(report/'frozen_evaluation/models.json')==config['models_sha256']
    audit=read(report/'audit/summary.json');test=read(report/'test/summary.json');ready=read(report/'preflight/ready.json')
    assert audit['status']=='DENSE_FNO5_TEST_AND_SAVED_OUTPUT_AUDIT_COMPLETE'
    assert audit['saved_native_fields_checked']==test['total_case_predictions']==1000
    assert audit['raw_timing_pairs_checked']==1000000
    assert audit['no_weight_updates'] and audit['no_test_selection']
    assert ready['status']=='DENSE_FNO5_VALIDATION_PREFLIGHT_PASS' and len(ready['models'])==5
    assert ready['validation_fields']==10 and not ready['test_targets_read']
    with (report/'audit/case_metrics.csv').open(newline='') as f:cases=list(csv.DictReader(f))
    assert len(cases)==len({(r['model_id'],r['index']) for r in cases})==1000
    old=HERE.parent/'architecture_budget_test_20261112_v1/report/test/models/tfno_s931101/summary.json'
    old_cases=read(old)['cases'];reference_inputs={(r['index'],r['geometry_hash'],r['reynolds']) for r in old_cases}
    devices={};training_seconds=0.;refs={r['index']:r['reference_physics'] for r in old_cases}
    for i,row in enumerate(manifest):
        assert row['model']=='fno' and row['seed']==931101+i and row['group']==i
        trained=read(report/'main'/row['id']/'complete.json')
        assert trained['within_time_limit'] and trained['limit_seconds']==14400 and not trained['test_targets_read']
        assert min(trained['complete_native_TRAIN_case_visits'])>0 and len(trained['complete_native_TRAIN_case_visits'])==640
        assert trained['selected_checkpoint_sha256']==row['checkpoint_sha256']
        closed=read(report/'allocation_closeout'/('main_'+str(i)+'.json'))
        assert closed['worker_exit_code']==0
        runtime=re.search(r'\bRunTime=(\S+)',closed['scheduler_stdout']).group(1)
        seconds=sum(int(x)*60**j for j,x in enumerate(reversed(runtime.split(':'))));assert seconds<=14400
        training_seconds+=seconds
        result=read(report/'test/models'/row['id']/'summary.json')
        assert result['model']==row and result['status']=='DENSE_FNO5_MODEL_TEST_COMPLETE'
        assert result['prediction_shape']==[200,512,512,3] and result['timing_shape']==[200,1000]
        assert result['no_weight_updates'] and result['no_test_selection']
        assert {(r['index'],r['geometry_hash'],r['reynolds']) for r in result['cases']}==reference_inputs
        audited=next(r for r in audit['models'] if r['id']==row['id'])
        for key in ['all_cell_mse','fluid_velocity_relative_l2','fluid_pressure_relative_l2']:
            values=[r[key] for r in result['cases']];assert all(math.isfinite(v) and v>=0 for v in values)
            assert math.isclose(statistics.mean(values),audited[key],rel_tol=1e-11,abs_tol=1e-13)
        for case in result['cases']:
            assert case['reference_physics']==refs[case['index']]
        for label,key,flag in [('gpu','worst_gpu_p99_ms','gpu_5ms_pass'),('host_instrumented','worst_host_p99_ms','host_instrumented_5ms_pass')]:
            value=max(r['timing'][label]['p99_ms'] for r in result['cases'])
            assert math.isclose(value,audited[key],rel_tol=1e-12)
            assert result[flag]==audited[flag]==(value<=5)
        before=result['environment_before'];after=result['environment_after']
        assert before['gpu']==after['gpu']=='NVIDIA H200' and before['threads']==after['threads']==4
        assert not any(e[k] for e in [before,after] for k in ['matmul_tf32','cudnn_tf32'])
        devices[str(i)]=before['hardware'].split(',')[1].strip()
        assert after['hardware'].split(',')[1].strip()==devices[str(i)]
    assert len(set(devices.values()))==5
    assert all(not f.read_text().strip() for f in (report/'logs').glob('*.err'))
    assert not read(report/'status.json')['queue'].strip()
    program=r'''import sys,json,pathlib,hashlib
p=json.load(sys.stdin);root=pathlib.Path(p['run_root']);files={}
for path in root.rglob('*'):
 if path.is_file() and path.suffix in ['.json','.csv'] and path.stat().st_size<12000000:files[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
models=json.loads((root/'frozen_evaluation/models.json').read_text())['models'];totals={'selected_checkpoint_bytes':0,'prediction_bytes':0,'raw_timing_bytes':0}
for m in models:
 folder=root/'test/models'/m['id']
 for key,path in [('selected_checkpoint_bytes',pathlib.Path(m['checkpoint'])),('prediction_bytes',folder/'predictions.npy'),('raw_timing_bytes',folder/'raw_timings.npz')]:totals[key]+=path.stat().st_size
print(json.dumps({'small_report_sha256':files,'Perun_only_sizes':totals}))
'''
    inventory=json.loads(ssh_python(program,receipt))
    for name,value in inventory['small_report_sha256'].items():assert sha(report/name)==value,name
    result=dict(status='DENSE_FNO5_COMPLETION_VERIFIED',checked_at=datetime.now(timezone(timedelta(hours=1))).isoformat(),
        source=receipt['source'],run_root=receipt['run_root'],models=5,TEST_fields=1000,
        timing_pairs_checked_by_CPU_audit=1000000,validation_preflight_fields=10,
        training_closeout_seconds_sum=training_seconds,training_H200_hours=training_seconds/3600,
        all5_training_worker_exit_codes_zero=True,common_cases_and_reference_physics_with_original50=True,
        GPU_UUIDs=devices,training_sources_and_frozen_models_unchanged=True,
        scheduler_final_records_expired=True,
        completion_evidence='Five training closeouts with worker exit0, five full TEST summaries, saved-output CPU audit PASS, empty error logs and queue. Expired final scheduler entries are not claimed verified.',
        remote_inventory=inventory)
    (HERE/'completion_check.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(result['status'],'5 models / 1000 TEST fields / 1M timing pairs')
    print('Main training H200-h:',result['training_H200_hours'])


if __name__=='__main__':main()
