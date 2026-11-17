"""CPU-only full TRAIN/VAL re-score of baseline, phase best and phase last."""
import argparse
from pathlib import Path
import re
import time
import gc
import numpy as np
import torch
from runtime import read,digest,dependencies,check_parents,recovered_cursor


def main(index,root):
    policy,parents,common,training,models=dependencies();row=parents['tasks'][index]
    common.setup('cpu',911120);folder=root/'continuation'/row['id'];done=read(folder/'complete.json')
    assert done['status']=='EXPLORATORY_CONTINUATION_COMPLETE' and done['within_additional_limit']
    assert done['excluded_from_primary_experiment'] and not done['test_targets_read']
    assert done['additional_allocation_seconds']<=14400
    recovered_cursor(done['final_case_counts'],row['seed'],done['global_updates'])
    assert np.array_equal(done['initial_case_counts'],read_parent_counts(row))
    assert done['additional_updates']==sum(done['final_case_counts'])-sum(done['initial_case_counts'])>0
    assert min(np.asarray(done['final_case_counts'])-np.asarray(done['initial_case_counts']))>0
    assert all(v['step']>=row['resume_cursor']['global_updates'] for v in done['optimizer_steps'])
    check_parents(row)
    closed=read(root/'allocation_closeout'/('continuation_'+str(index)+'.json'))
    assert closed['worker_exit_code']==0
    runtime=re.search(r'\bRunTime=(\S+)',closed['scheduler_stdout']).group(1)
    actual_extra=sum(int(x)*60**i for i,x in enumerate(reversed(runtime.split(':'))));assert 0<actual_extra<=14400
    fixture=training.Fixture(policy['fixture_root'],'cpu')
    original=read(row['original_complete'])
    old_expected=min(original['history'],key=lambda v:v['validation']['mean_raw_case_balanced_loss'])['validation']
    checkpoints=[('baseline_4h',row['original_best'],row['original_best_sha256'],old_expected),
                 ('continuation_best',done['new_best_checkpoint'],done['new_best_sha256'],done['best_new_validation']),
                 ('continuation_last',done['new_last_checkpoint'],done['new_last_sha256'],done['last_validation'])]
    metrics={};started=time.monotonic()
    for label,path,sha,expected in checkpoints:
        assert digest(path)==sha
        cp=torch.load(path,map_location='cpu',weights_only=False)
        assert cp['model']==row['model'] and cp['normalization']==fixture.normalization
        if label!='baseline_4h':
            assert cp['task']['mode']=='exploratory_continuation' and cp['excluded_from_primary_experiment']
            assert cp['task']['constant_saved_learning_rates']==row['constant_learning_rates']
        model=models.restore_model(cp).cpu().eval().requires_grad_(False)
        scores={}
        with torch.inference_mode():
            for phase,count in [('train',640),('validation',160)]:
                values=[float(common.patch_loss(common.patch_prediction(model.core,row['model'],x),y,w)) for x,y,w in fixture.iterator(phase)]
                assert len(values)==count and np.isfinite(values).all() and min(values)>=0
                scores[phase]=dict(mean_raw_case_balanced_loss=float(np.mean(values)),case_losses=values,cases=count)
            physical=training.physical_validation_metrics(model,fixture)
        assert np.allclose(scores['validation']['case_losses'],expected['case_losses'],rtol=.005,atol=1e-7)
        assert len(physical['case_metrics'])==160
        if label=='baseline_4h':
            for key in ['mean_velocity_relative_L2','mean_pressure_relative_L2']:
                assert np.isclose(physical[key],original['physical_validation_metrics'][key],rtol=.005,atol=2e-6)
        assert digest(path)==sha
        metrics[label]=dict(checkpoint=path,checkpoint_sha256=sha,phase_metrics=scores,physical_validation=physical)
        print('AUDITED',row['id'],label,'TRAIN',scores['train']['mean_raw_case_balanced_loss'],'VAL',scores['validation']['mean_raw_case_balanced_loss'],flush=True)
        del model,cp;gc.collect()
    check_parents(row)
    chosen='baseline_4h' if done['selected_is_original_baseline'] else 'continuation_best'
    assert metrics[chosen]['checkpoint_sha256']==done['selected_checkpoint_sha256']
    baseline=metrics['baseline_4h']['phase_metrics']['validation']['mean_raw_case_balanced_loss']
    result=dict(status='EXPLORATORY_SECOND_CHANCE_CPU_AUDIT_PASS',id=row['id'],model=row['model'],seed=row['seed'],
        metrics=metrics,selected_best_so_far=chosen,
        original_GPU_seconds=row['original_allocated_GPU_seconds'],additional_GPU_seconds=actual_extra,
        cumulative_observed_GPU_seconds=row['original_allocated_GPU_seconds']+actual_extra,
        cumulative_nominal_GPU_seconds=28800,
        phase_best_reduction_percent=100*(1-metrics['continuation_best']['phase_metrics']['validation']['mean_raw_case_balanced_loss']/baseline),
        phase_last_reduction_percent=100*(1-metrics['continuation_last']['phase_metrics']['validation']['mean_raw_case_balanced_loss']/baseline),
        selected_reduction_percent=100*(1-metrics[chosen]['phase_metrics']['validation']['mean_raw_case_balanced_loss']/baseline),
        source_checkpoint_and_Adam_restoration_verified=True,original_files_unchanged=True,
        CPU_GPU_replay_tolerance=dict(relative=.005,absolute=1e-7,numerical_only=True),
        no_TEST_targets=True,excluded_from_primary_experiment=True,elapsed_CPU_wall_seconds=time.monotonic()-started,
        final_optimizer_step_counters=done['optimizer_steps'],logical_updates=done['global_updates'])
    common.save(root/'audit'/row['id']/'complete.json',result)


def read_parent_counts(row):
    assert digest(row['original_last'])==row['original_last_sha256']
    return torch.load(row['original_last'],map_location='cpu',weights_only=False)['case_counts']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();main(a.index,a.root)
