"""Independent model seeds using the already audited training implementation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from common import SOURCE,read,save,digest,engine,manifest,load_module


def task_config(config,index):
    selected=manifest(config);task=selected['tasks'][index]
    assert task['index']==index
    candidate=next(r for r in selected['selected'] if r['model']==task['model'])
    e,c=engine(config);c['calibration_seed']=task['seed']
    return e,c,task,candidate


def preflight(config,output):
    e,c=engine(config);e.configure('cpu',4,911112);selected=manifest(config)
    from checkpoint_io import restore_model
    from data_metrics import field_metrics
    from checks import run
    output.mkdir(parents=True,exist_ok=False);checks=run();torch.set_num_threads(4);store=e.Store(c,'cpu')
    x,y,fixture=e.point_data(c,'cpu');rows=[]
    assert fixture['data_plan']['train_ids']==store.train_ids and fixture['data_plan']['normalization']==store.normalization
    for candidate in selected['selected']:
        path=Path(candidate['checkpoint']);assert digest(path)==candidate['checkpoint_sha256']
        cp=torch.load(path,map_location='cpu',weights_only=True)
        assert cp['model']==candidate['model'] and cp['normalization']==store.normalization
        expected=e.recipe_config(c,candidate['recipe']);assert cp['config']==expected
        for name,sha in cp['source_sha256'].items():assert digest(Path(cp['source'])/name)==sha
        model=restore_model(cp).eval();index=store.val_ids[0];known,target=store.get(index)
        with torch.inference_mode():prediction=model(known).numpy()
        metrics=field_metrics(prediction,target.numpy(),known['fluid'].numpy(),known['reynolds'])
        assert np.isfinite(metrics['all_cell_mse'])
        for seed in selected['final_seeds']:
            final=e.recipe_config(dict(c,calibration_seed=seed),candidate['recipe'])
            assert final==dict(cp['config'],seed=seed)
        rows.append(dict(model=candidate['model'],strict_restore=True,first_validation_mse=metrics['all_cell_mse']))
        del model,cp
    for reference in selected['archived_pinn']:
        assert digest(Path(reference['checkpoint']))==reference['checkpoint_sha256']
    # Exercise the reused training/checkpoint/scheduler path on tiny TRAIN data.
    # These diagnostic weights are never counted among the 45 final models.
    tiny=e.Store(c,'cpu');tiny.train_ids=tiny.train_ids[:2];tiny.val_ids=tiny.val_ids[:2]
    recipe=next(r['recipe'] for r in selected['selected'] if r['model']=='pinet')
    small_config=dict(c,calibration_seed=911112,max_epochs=2,assess_every=1,minimum_epochs=1)
    mc=e.recipe_config(small_config,recipe);torch.manual_seed(mc['seed'])
    model=e.make_model('pinet',mc,store.normalization);small=output/'tiny';small.mkdir();e.SOURCE=SOURCE
    result=e.train_gradient(model,'pinet',dict(mc,gradient_batch_points=16),small_config,tiny,x[:32],y[:32],small,'cpu')
    cp=torch.load(small/'best.pt',map_location='cpu',weights_only=True);restored=restore_model(cp)
    assert cp['source']==str(SOURCE) and cp['config']['seed']==911112
    assert np.isclose(e.validate(restored,tiny,tiny.val_ids)['mse'],cp['validation_mse'],rtol=1e-12,atol=1e-14)
    save(output/'ready.json',dict(status='ARCHITECTURE_MAIN_PREFLIGHT_PASS',core_checks=checks,models=rows,
        task_count=45,existing_pinn_references=5,tiny_optimizer_updates=result['optimizer_updates'],
        selection_sha256=config['selection_sha256'],point_sample_sha256=fixture['point_sample_sha256'],test_targets_read=False))
    print('ARCHITECTURE_MAIN_PREFLIGHT_PASS',flush=True)


def phase(config,index,root,continuation=False):
    e,c,task,candidate=task_config(config,index);folder=root/'models'/task['id']
    if not continuation:
        e.configure(task['device'],4,task['seed']);e.SOURCE=SOURCE
        e.trial(c,candidate['recipe'],folder/'training'/candidate['recipe']['id'],task['device'])
    else:
        # Fresh process: the frozen extension restores parent Adam moments and
        # torch/NumPy RNG after initialization. The actual model seed is in cp.
        module=load_module('frozen_kan_extension',Path(config['review_source'])/'extend_kan.py')
        module.SOURCE=SOURCE
        runtime=dict(config,calibration_run=str(folder),extra_epochs=200,run_seed=task['seed'])
        module.main(runtime,candidate['recipe']['id'],folder/'continuation')


def train(config,index,root):
    e,c,task,candidate=task_config(config,index);folder=root/'models'/task['id']
    assert read(root/'prepared/ready.json')['status']=='ARCHITECTURE_MAIN_PREFLIGHT_PASS'
    folder.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    save(folder/'task.json',dict(task=task,selected=candidate,selection_sha256=config['selection_sha256']))
    def child(extra):
        subprocess.run([sys.executable,'-B','-u',str(SOURCE/'train_main.py'),'--phase',extra,
                        '--index',str(index),'--root',str(root)],check=True)
    child('phase1');phase1=read(folder/'training'/candidate['recipe']['id']/'summary.json')
    assert phase1['status']=='CALIBRATION_TRAINING_COMPLETE' and not phase1['test_targets_read']
    cp1=torch.load(phase1['best_checkpoint'],map_location='cpu',weights_only=True)
    assert cp1['config']['seed']==task['seed']
    choices=[dict(checkpoint=phase1['best_checkpoint'],checkpoint_sha256=phase1['best_checkpoint_sha256'],
                  validation_mse=cp1['validation_mse'],epoch=cp1['epoch'],phase='cosine')]
    extended=False;continuation=None
    if candidate['continuation'] and phase1['training']['last_epoch']==200:
        child('phase2');continuation=read(folder/'continuation/summary.json')
        assert continuation['status']=='KAN_CONTINUATION_COMPLETE'
        choices.append(dict(checkpoint=continuation['best_checkpoint'],checkpoint_sha256=continuation['best_checkpoint_sha256'],
            validation_mse=continuation['best_validation_mse'],epoch=continuation['best_epoch'],phase='constant_lr_continuation'))
        extended=True
    best=min(choices,key=lambda r:r['validation_mse']);assert digest(Path(best['checkpoint']))==best['checkpoint_sha256']
    save(folder/'summary.json',dict(status='ARCHITECTURE_MAIN_MODEL_COMPLETE',task=task,selected_recipe=candidate['recipe'],
        selection_sha256=config['selection_sha256'],best=best,phase_candidates=choices,continuation_performed=extended,
        continuation_skipped_after_early_stop=bool(candidate['continuation'] and not extended),
        phase1_summary=str(folder/'training'/candidate['recipe']['id']/'summary.json'),
        continuation_summary=None if continuation is None else str(folder/'continuation/summary.json'),
        wall_seconds=time.monotonic()-start,slurm_job_id=os.environ.get('SLURM_JOB_ID'),test_targets_read=False))
    print('ARCHITECTURE_MAIN_MODEL_COMPLETE',task['id'],best,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','train','phase1','phase2'],required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--index',type=int,default=-1)
    p.add_argument('--group',choices=['cpu','cuda']);a=p.parse_args()
    config=read(SOURCE/'config.json')
    if a.group:
        tasks=[r for r in manifest(config)['tasks'] if r['device']==a.group];a.index=tasks[a.index]['index']
    if a.phase=='prepare':preflight(config,a.root/'prepared')
    elif a.phase=='train':train(config,a.index,a.root)
    else:phase(config,a.index,a.root,a.phase=='phase2')
