"""Verify four exploratory lineages, all audited losses and cumulative costs."""
from datetime import datetime,timezone,timedelta
import hashlib
import json
import math
from pathlib import Path
import statistics
from deploy import ssh_python

HERE=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    receipt=read(HERE/'submission.json');parents=read(HERE/'parents.json')['tasks'];report=HERE/'report/continuation'
    for name,value in receipt['sha256'].items():assert hashlib.sha256((HERE/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value,name
    summary=read(report/'summary/complete.json')
    assert summary['status']=='SEPARATE_SECOND_CHANCE_COMPLETE' and len(summary['models'])==4
    assert summary['no_TEST_targets'] and summary['excluded_from_primary_experiment']
    assert not summary['additional_16h_extension_not_scheduled'] is False
    saturated=[];extra_seconds=0.;verified_values=0
    for i,row in enumerate(parents):
        done=read(report/'continuation'/row['id']/'complete.json')
        audit=read(report/'audit'/row['id']/'complete.json')
        item=next(r for r in summary['models'] if r['id']==row['id'])
        assert done['status']=='EXPLORATORY_CONTINUATION_COMPLETE' and done['within_additional_limit']
        assert audit['status']=='EXPLORATORY_SECOND_CHANCE_CPU_AUDIT_PASS'
        assert not done['test_targets_read'] and audit['no_TEST_targets'] and done['excluded_from_primary_experiment']
        assert done['task']['constant_saved_learning_rates']==row['constant_learning_rates']
        assert done['task']['no_optimizer_reset'] and done['task']['no_architecture_change']
        assert done['history'][0]['new_updates']==0 and done['history'][0]['updates']==row['resume_cursor']['global_updates']
        counts=done['final_case_counts'];initial=done['initial_case_counts'];updates=sum(counts);full,pos=divmod(updates,640)
        assert len(counts)==len(initial)==640 and updates==done['global_updates']
        assert done['additional_updates']==updates-sum(initial)>0 and all(a>b for a,b in zip(counts,initial))
        assert done['final_cursor']==dict(epoch=full+1,position=pos,global_updates=updates)
        assert all(v in [full,full+1] for v in counts) and sum(v==full+1 for v in counts)==pos
        assert all(v['dtype']=='torch.float32' and v['step']==min(updates,2**24) for v in done['optimizer_steps'])
        if updates>2**24:saturated.append(dict(id=row['id'],logical_updates=updates,Adam_float32_step=2**24))
        closed=read(report/'allocation_closeout'/('continuation_'+str(i)+'.json'));assert closed['worker_exit_code']==0
        assert 0<audit['additional_GPU_seconds']<=14400
        extra_seconds+=audit['additional_GPU_seconds']
        assert audit['cumulative_observed_GPU_seconds']==row['original_allocated_GPU_seconds']+audit['additional_GPU_seconds']
        assert math.isclose(item['additional_GPU_h'],audit['additional_GPU_seconds']/3600,rel_tol=1e-12)
        assert math.isclose(item['cumulative_GPU_h'],audit['cumulative_observed_GPU_seconds']/3600,rel_tol=1e-12)
        for label in ['baseline_4h','continuation_best','continuation_last']:
            value=audit['metrics'][label]
            for phase,n in [('train',640),('validation',160)]:
                metric=value['phase_metrics'][phase];assert metric['cases']==len(metric['case_losses'])==n
                assert all(math.isfinite(v) and v>=0 for v in metric['case_losses'])
                assert math.isclose(statistics.mean(metric['case_losses']),metric['mean_raw_case_balanced_loss'],rel_tol=1e-12)
                verified_values+=n
            physical=value['physical_validation'];assert len(physical['case_metrics'])==160
            for key,mean in [('velocity_relative_L2','mean_velocity_relative_L2'),('pressure_relative_L2','mean_pressure_relative_L2')]:
                assert math.isclose(statistics.mean(r[key] for r in physical['case_metrics']),physical[mean],rel_tol=1e-12)
        old=audit['metrics']['baseline_4h']['phase_metrics']['validation']['mean_raw_case_balanced_loss']
        best=audit['metrics']['continuation_best']['phase_metrics']['validation']['mean_raw_case_balanced_loss']
        assert math.isclose(100*(1-best/old),item['phase_best_reduction_percent'],rel_tol=1e-12)
        assert audit['original_files_unchanged'] and audit['source_checkpoint_and_Adam_restoration_verified']
    assert verified_values==9600
    assert math.isclose(summary['total_additional_observed_GPU_hours'],extra_seconds/3600,rel_tol=1e-12)
    assert all(not f.read_text().strip() for f in (report/'logs').glob('*.err'))
    assert not (HERE/'report/queue.txt').read_text().strip()
    program=r'''import sys,json,pathlib,hashlib
p=json.load(sys.stdin);root=pathlib.Path(p['run_root']);files={}
for path in root.rglob('*'):
 if path.is_file() and path.suffix in ['.json','.csv'] and path.stat().st_size<12000000:files[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
weights=[]
for path in sorted((root/'continuation').glob('*/*.pt')):weights.append({'path':str(path),'bytes':path.stat().st_size})
print(json.dumps({'small_report_sha256':files,'Perun_only_checkpoints':weights}))
'''
    inventory=json.loads(ssh_python(program,receipt))
    for name,value in inventory['small_report_sha256'].items():assert sha(report/name)==value,name
    assert len(inventory['Perun_only_checkpoints'])==8
    result=dict(status='ALL_FOUR_SECOND_CHANCE_LINEAGES_VERIFIED',checked_at=datetime.now(timezone(timedelta(hours=1))).isoformat(),
        source=receipt['source'],run_root=receipt['run_root'],lineages=4,TRAIN_VAL_case_loss_comparisons=9600,
        additional_observed_H200_hours=extra_seconds/3600,additional_training_CPU_core_hours=4*extra_seconds/3600,
        excluded_from_primary_experiment=True,no_TEST_targets=True,no16h_extension_scheduled=True,
        float32_Adam_counter_saturation=saturated,integer_visit_and_update_counters_verified=True,
        completion_evidence='Four worker exit0 closeouts, four independent CPU audit PASS records, complete aggregate, empty error logs and queue. Final Slurm records expired.',
        inventory=inventory)
    (HERE/'completion_check.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(result['status'],'4 lineages, 9600 case-loss comparisons')
    print('Additional H200-hours:',extra_seconds/3600,'float32 counter saturation records:',len(saturated))


if __name__=='__main__':main()
