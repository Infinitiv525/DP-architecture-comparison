"""Verify all 45 frozen TRAIN/VAL diagnostic records and report provenance."""
import hashlib
import json
import math
from pathlib import Path
import statistics
from datetime import datetime,timezone,timedelta
from deploy import ssh_python

HERE=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text())


def main():
    receipt=read(HERE/'submission.json');config=read(HERE/'config.json');manifest=read(HERE/'models.json')
    for name,expected in receipt['sha256'].items():
        assert hashlib.sha256((HERE/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==expected,name
    assert hashlib.sha256((HERE/'models.json').read_bytes()).hexdigest()==config['models_sha256']
    rows=[];expected_ids={r['id'] for r in manifest['models']};assert len(expected_ids)==45
    for name in config['families']:
        complete=read(HERE/'report'/name/'complete.json')
        assert complete['status']=='FROZEN_TRAIN_VAL_CPU_DIAGNOSTIC_COMPLETE'
        assert complete['no_training'] and not complete['test_targets_read']
        assert complete['threads']==4 and len(complete['models'])==5
        assert complete['all_CPU_validation_means_match_stored_GPU']
        assert complete['all_CPU_validation_cases_match_stored_GPU']
        for r in complete['models']:
            assert r==read(HERE/'report'/name/(r['id']+'.json'))
            frozen=next(m for m in manifest['models'] if m['id']==r['id'])
            assert r['checkpoint_sha256']==frozen['checkpoint_sha256'] and r['seed']==frozen['seed']
            assert r['no_weight_updates'] and not r['test_targets_read']
            assert r['CPU_validation_mean_matches_stored_GPU'] and r['CPU_validation_cases_match_stored_GPU']
            for phase,n in [('train',640),('validation',160)]:
                p=r['phase_metrics'][phase];assert p['cases']==len(p['case_losses'])==n
                assert all(math.isfinite(v) and v>=0 for v in p['case_losses'])
                assert math.isclose(statistics.mean(p['case_losses']),p['mean_raw_case_balanced_loss'],rel_tol=1e-12)
            rows.append(r)
    assert len(rows)==len({r['id'] for r in rows})==45
    assert {r['id'] for r in rows}==expected_ids
    assert not list((HERE/'report').glob('failure_*.json'))
    logs=list((HERE/'report/logs').glob('*.err'))
    assert len(logs)==9 and all(not f.read_text().strip() for f in logs)
    status=read(HERE/'report/status.json');assert not status['queue'].strip()
    # Completed Slurm array elements may have expired; all nine complete.json
    # records and their empty error logs remain the durable completion evidence.
    program=r'''import json,sys,pathlib,hashlib
p=json.load(sys.stdin);root=pathlib.Path(p['run_root']);result={}
for name in p['families']:
 for path in sorted((root/name).glob('*.json')):
  result[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
print(json.dumps(result))
'''
    hashes=json.loads(ssh_python(program,dict(run_root=receipt['run_root'],families=config['families'])))
    assert len(hashes)==54
    for name,expected in hashes.items():assert hashlib.sha256((HERE/'report'/name).read_bytes()).hexdigest()==expected,name
    output=dict(status='ALL45_TRAIN_VAL_DIAGNOSTICS_VERIFIED',
        checked_at=datetime.now(timezone(timedelta(hours=1))).isoformat(),source=receipt['source'],
        run_root=receipt['run_root'],models=45,case_loss_evaluations=36000,
        TRAIN_evaluations=28800,VAL_evaluations=7200,
        all_individual_VAL_cases_match_GPU_reference=True,
        no_weight_updates=True,no_TEST_targets=True,no_new_training=True,no_GPU_used=True,
        completion_evidence='Nine complete records, all 45 individual records, nine empty stderr logs, empty array queue; expired scheduler elements are not claimed individually verified.',
        remote_report_sha256=hashes)
    (HERE/'completion_check.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(output['status'],'45 models, 36000 finite case losses, 54 remote report hashes')


if __name__=='__main__':main()
