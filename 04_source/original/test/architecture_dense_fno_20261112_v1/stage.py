"""CPU gates: verify smoke, select on VAL, freeze all five checkpoints."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def once(path, value):
    path = Path(path)
    if path.exists():
        assert read(path) == value, 'Preserve different existing result: ' + str(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def check_training(root, mode, index, task_id, seconds):
    p = root/mode/task_id/'complete.json'
    r = read(p)
    assert r['task']['model']=='fno' and r['task']['mode']==mode
    assert r['limit_seconds']==seconds and r['within_time_limit']
    assert 0 < r['allocation_elapsed_seconds'] <= seconds
    assert not r['test_targets_read'] and r['accuracy_threshold'] is None
    assert r['hardware']=='NVIDIA H200' and r['allocated_CPU_cores']==4 and r['allocated_H200_GPUs']==1
    assert len(r['physical_validation_metrics']['case_metrics'])==160
    assert math.isfinite(r['best_complete_validation_loss'])
    assert digest(r['selected_checkpoint'])==r['selected_checkpoint_sha256']
    closed = read(root/'allocation_closeout'/(mode+'_'+str(index)+'.json'))
    assert closed['worker_exit_code']==0
    assert r['task']['time_budget_seconds']==seconds
    if mode != 'smoke':
        assert len(r['complete_native_TRAIN_case_visits'])==640
        assert min(r['complete_native_TRAIN_case_visits'])>0
        assert r['gradient_updates']>=640
    return r


def main(action, root):
    policy = read(HERE/'policy.json')
    receipt = read(root/'submission.json')
    for name, sha in receipt['sha256'].items():
        assert digest(HERE/name)==sha, name
    smoke = check_training(root,'smoke',0,'fno_smoke',60)
    if action=='check-smoke':
        once(root/'smoke_gate.json',dict(status='DENSE_FNO_SMOKE_PASS',
             parameter_ledger=smoke['selected_parameter_ledger'],test_targets_read=False,
             native_train_validation_fixture=True,full_training_authorized=True))
        print('DENSE_FNO_SMOKE_PASS',smoke['selected_parameter_ledger'],flush=True)
        return
    candidates = []
    for i in range(2):
        r = check_training(root,'tuning',i,'fno_t'+str(i),3600)
        assert r['task']['seed']==policy['tuning_seed'] and r['task']['configuration_index']==i
        candidates.append(dict(configuration_index=i,validation_loss=r['best_complete_validation_loss'],
           checkpoint=r['selected_checkpoint'],checkpoint_sha256=r['selected_checkpoint_sha256'],
           allocation_elapsed_seconds=r['allocation_elapsed_seconds'],stop_reason=r['stop_reason']))
    chosen = min(candidates,key=lambda r:(r['validation_loss'],r['configuration_index']))
    selection = dict(status='DENSE_FNO_CONFIGURATION_FROZEN_FROM_TWO_ONE_HOUR_VAL_TRIALS',
       selection_uses_test=False,prior_original_test_results_known=True,
       selection_metric='minimum complete mean raw case-balanced VAL loss; ties by configuration index',
       candidates=candidates,selected_configuration_index=chosen['configuration_index'],
       selected_tuning_checkpoint=chosen['checkpoint'],selected_tuning_checkpoint_sha256=chosen['checkpoint_sha256'],
       tasks=[dict(id='fno_s'+str(seed),model='fno',seed=seed,configuration_index=chosen['configuration_index'])
              for seed in policy['final_seeds']])
    once(root/'selection.json',selection)
    if action=='select':
        print('DENSE_FNO_CONFIGURATION_SELECTED',chosen['configuration_index'],flush=True)
        return
    assert action=='freeze-test'
    split = read(Path(policy['root'])/policy['split_path']) if 'root' in policy else None
    config = read(HERE/'evaluation_template.json')
    indices = read(config['split_path'])['indices']['validation'][:2]
    models = []
    for i,task in enumerate(selection['tasks']):
        r = check_training(root,'main',i,task['id'],14400)
        assert r['task']['seed']==task['seed'] and r['task']['configuration_index']==task['configuration_index']
        expected = []
        for j,index in enumerate(indices):
            v = r['physical_validation_metrics']['case_metrics'][j]
            expected.append(dict(index=index,velocity_l2=v['velocity_relative_L2'],pressure_l2=v['pressure_relative_L2']))
        models.append(dict(id=task['id'],model='fno',seed=task['seed'],group=i,
            kind='posthoc_resource_matched_dense_fno_supplement',
            checkpoint=r['selected_checkpoint'],checkpoint_sha256=r['selected_checkpoint_sha256'],
            expected_validation=expected,parameter_ledger=r['selected_parameter_ledger'],
            training_summary=str(root/'main'/task['id']/'complete.json'),
            training_summary_sha256=digest(root/'main'/task['id']/'complete.json'),
            training_elapsed_seconds=r['allocation_elapsed_seconds'],training_stop_reason=r['stop_reason']))
        config['metadata_sha256'][str(root/'main'/task['id']/'complete.json')]=models[-1]['training_summary_sha256']
        closed=root/'allocation_closeout'/('main_'+str(i)+'.json')
        config['metadata_sha256'][str(closed)]=digest(closed)
    out = root/'frozen_evaluation'
    manifest = dict(status='DENSE_FNO5_MANIFEST_FROZEN',models=models,
        selection_uses_test=False,prior_original_test_results_known=True,
        posthoc_extension=True,all_training_complete_before_test=True,
        selection_sha256=digest(root/'selection.json'))
    once(out/'models.json',manifest)
    config.update(main_source=str(HERE),main_source_sha256=receipt['sha256'],
        models_sha256=digest(out/'models.json'))
    once(out/'config.json',config)
    once(out/'freeze_receipt.json',dict(status='ALL_FIVE_FNO_WEIGHTS_FROZEN',
        models_sha256=digest(out/'models.json'),config_sha256=digest(out/'config.json'),
        selection_sha256=digest(root/'selection.json'),source_sha256=receipt['sha256'],
        test_targets_read_by_freezing=False))
    print('ALL_FIVE_FNO_WEIGHTS_FROZEN',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--action',choices=['check-smoke','select','freeze-test'],required=True)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    main(a.action,a.root)
