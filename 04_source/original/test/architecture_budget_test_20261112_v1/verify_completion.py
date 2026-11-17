"""Read-only completion checks; writes a small local closeout record only."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from datetime import datetime, timezone, timedelta

from deploy import ssh_python

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    report = HERE / 'report'
    receipt = read(HERE / 'submission.json')
    manifest = read(HERE / 'models.json')['models']
    for name, expected in receipt['sha256'].items():
        assert hashlib.sha256((HERE / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == expected, name
    audit = read(report / 'audit/summary.json')
    test = read(report / 'test/summary.json')
    assert audit['status'] == 'FINAL50_TEST_AND_SAVED_OUTPUT_AUDIT_COMPLETE'
    assert test['status'] == 'FINAL50_TEST_COMPLETE'
    assert audit['saved_native_fields_checked'] == test['total_case_predictions'] == 10000
    assert audit['raw_timing_pairs_checked'] == 10000000
    assert audit['no_weight_updates'] and audit['no_test_selection']
    assert audit['models_sha256'] == receipt['sha256']['models.json']
    with (report / 'audit/case_metrics.csv').open(newline='') as stream:
        cases = list(csv.DictReader(stream))
    with (report / 'audit/model_metrics.csv').open(newline='') as stream:
        models = list(csv.DictReader(stream))
    assert len(manifest) == len(models) == len(audit['models']) == 50
    assert len(cases) == len({(r['model_id'], r['index']) for r in cases}) == 10000
    assert {m['id'] for m in manifest} == {m['id'] for m in models}
    metric_keys = ['all_cell_mse', 'fluid_velocity_relative_l2', 'fluid_pressure_relative_l2']
    input_cases = None
    devices = {}
    for model in manifest:
        rows = [r for r in cases if r['model_id'] == model['id']]
        assert len(rows) == 200
        identifiers = {(r['index'], r['geometry_hash'], r['reynolds']) for r in rows}
        if input_cases is None:
            input_cases = identifiers
        assert identifiers == input_cases
        summary = read(report / 'test/models' / model['id'] / 'summary.json')
        assert summary['model'] == model
        assert summary['status'] == 'FINAL50_MODEL_TEST_COMPLETE'
        assert summary['prediction_shape'] == [200, 512, 512, 3]
        assert summary['timing_shape'] == [200, 1000]
        assert summary['no_weight_updates'] and summary['no_test_selection']
        record = next(m for m in models if m['id'] == model['id'])
        for key in metric_keys:
            values = [float(r[key]) for r in rows]
            assert all(math.isfinite(v) and v >= 0 for v in values)
            assert math.isclose(statistics.mean(values), float(record[key]), rel_tol=1e-11, abs_tol=1e-13)
        for label, key in [('gpu', 'worst_gpu_p99_ms'), ('host_instrumented', 'worst_host_p99_ms')]:
            worst = max(float(r[label + '_p99_ms']) for r in rows)
            assert math.isclose(worst, float(record[key]), rel_tol=1e-12)
            flag = 'gpu_5ms_pass' if label == 'gpu' else 'host_instrumented_5ms_pass'
            assert (record[flag] == 'True') == (worst <= 5)
        for position in ['environment_before', 'environment_after']:
            env = summary[position]
            assert env['gpu'] == 'NVIDIA H200' and env['threads'] == 4
            assert not env['matmul_tf32'] and not env['cudnn_tf32']
            uuid = env['hardware'].split(',')[1].strip()
            assert uuid.startswith('GPU-')
            group = str(model['group'])
            if group in devices:
                assert devices[group] == uuid
            else:
                devices[group] = uuid
    assert len(devices) == len(set(devices.values())) == 5
    slurm = read(report / 'status.json')
    assert not slurm['queue'].strip()
    durations = {}
    for role, value in slurm['status'].items():
        assert value['returncode'] == 0 and not value['stderr'].strip()
        states = re.findall(r'JobState=(\w+)', value['stdout'])
        exits = re.findall(r'ExitCode=([^\s]+)', value['stdout'])
        assert states == ['COMPLETED'] * (5 if role == 'test' else 1)
        assert exits == ['0:0'] * len(states)
        durations[role] = re.findall(r'RunTime=([^\s]+)', value['stdout'])
    assert all(not p.read_text().strip() for p in (report / 'logs').glob('*.err'))
    program = r'''import sys,json,pathlib,hashlib
p=json.load(sys.stdin);root=pathlib.Path(p['run_root']);manifest=json.loads((pathlib.Path(p['source'])/'models.json').read_text())['models']
totals={'checkpoint_bytes':0,'prediction_bytes':0,'raw_timing_bytes':0};records=[]
for model in manifest:
 folder=root/'test/models'/model['id']; paths={'checkpoint':pathlib.Path(model['checkpoint']),'prediction':folder/'predictions.npy','raw_timing':folder/'raw_timings.npz'}
 sizes={key:path.stat().st_size for key,path in paths.items()};assert sizes['prediction']>200*512*512*3*4 and all(x>0 for x in sizes.values())
 for key,size in sizes.items():totals[key+'_bytes']+=size
 records.append({'id':model['id'],'bytes':sizes})
small={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'audit/summary.json',root/'audit/case_metrics.csv',root/'audit/model_metrics.csv',root/'test/summary.json']}
print(json.dumps({'totals':totals,'files':records,'small_report_sha256':small}))
'''
    inventory = json.loads(ssh_python(program, receipt))
    for name, expected in inventory['small_report_sha256'].items():
        assert hashlib.sha256((report / name).read_bytes()).hexdigest() == expected, name
    result = dict(status='FINAL50_COMPLETION_VERIFIED',
                  checked_at=datetime.now(timezone(timedelta(hours=1))).isoformat(),
                  source=receipt['source'], run_root=receipt['run_root'],
                  models=50, primary_models=45, auxiliary_pinn_models=5,
                  test_cases_per_model=200, verified_native_fields=10000,
                  raw_timing_pairs_verified_by_cpu_audit=10000000,
                  common_test_cases=True, independent_gpu_uuids=devices,
                  slurm_runtime=durations, all_job_exits_zero=True,
                  training_or_configuration_changes=False,
                  report_checksums_and_remote_inventory=inventory)
    (HERE / 'completion_check.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(result['status'])
    print('Runtime:', durations)
    print('Perun-only storage:', inventory['totals'])


if __name__ == '__main__':
    main()
