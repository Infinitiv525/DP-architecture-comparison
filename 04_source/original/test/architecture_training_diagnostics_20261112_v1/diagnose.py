"""CPU-only re-scoring of frozen TRAIN/VAL checkpoints; no optimizer or TEST."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text())


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main(index,root):
    import numpy as np
    import torch
    config=read(HERE/'config.json');manifest=read(HERE/'models.json')
    assert digest(HERE/'models.json')==config['models_sha256']
    source=Path(config['field_source'])
    for name,sha in config['field_source_sha256'].items():assert digest(source/name)==sha,name
    assert digest(Path(config['fixture_root'])/'fixture/ready.json')==config['fixture_ready_sha256']
    sys.path.insert(0,str(source))
    import common
    import train
    import operator_models
    common.setup('cpu',911118)
    name=config['families'][index];models=[m for m in manifest['models'] if m['model']==name]
    assert len(models)==5 and all(m['model']!='pinn' for m in models)
    folder=root/name;folder.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();fixture=train.Fixture(config['fixture_root'],'cpu');results=[]
    for row in models:
        assert digest(row['checkpoint'])==row['checkpoint_sha256']
        cp=torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
        assert cp['model']==name and cp['task']['seed']==row['seed'] and cp['normalization']==fixture.normalization
        model=operator_models.restore_model(cp).cpu().eval().requires_grad_(False)
        summary=read(row['training_summary']);assert digest(row['training_summary'])==row['training_summary_sha256']
        expected=min(summary['history'],key=lambda r:r['validation']['mean_raw_case_balanced_loss'])['validation']
        phase_metrics={}
        with torch.inference_mode():
            for phase,count in [('train',640),('validation',160)]:
                values=[]
                for x,y,w in fixture.iterator(phase):
                    loss=float(common.patch_loss(common.patch_prediction(model.core,name,x),y,w))
                    assert np.isfinite(loss) and loss>=0
                    values.append(loss)
                assert len(values)==count
                phase_metrics[phase]=dict(cases=count,mean_raw_case_balanced_loss=float(np.mean(values)),
                    median_raw_case_balanced_loss=float(np.median(values)),
                    p95_raw_case_balanced_loss=float(np.percentile(values,95)),case_losses=values)
        actual=phase_metrics['validation']['mean_raw_case_balanced_loss']
        matched=bool(np.isclose(actual,expected['mean_raw_case_balanced_loss'],rtol=.005,atol=1e-7))
        all_cases_matched=bool(np.allclose(phase_metrics['validation']['case_losses'],expected['case_losses'],rtol=.005,atol=1e-7))
        assert digest(row['checkpoint'])==row['checkpoint_sha256']
        result=dict(id=row['id'],model=name,seed=row['seed'],checkpoint_sha256=row['checkpoint_sha256'],
            phase_metrics=phase_metrics,validation_over_training_loss_ratio=actual/max(phase_metrics['train']['mean_raw_case_balanced_loss'],1e-15),
            expected_stored_validation_loss=expected['mean_raw_case_balanced_loss'],
            CPU_validation_mean_matches_stored_GPU=matched,CPU_validation_cases_match_stored_GPU=all_cases_matched,
            CPU_GPU_replay_tolerance=dict(relative=.005,absolute=1e-7,numerical_check_only=True),
            stop_reason=summary['stop_reason'],training_allocated_seconds=summary['allocation_elapsed_seconds'],
            best_checkpoint_selected_at_seconds=summary['best_checkpoint_selected_at_seconds'],
            updates=summary['gradient_updates'],units=summary['final_grown_units'],
            min_native_case_visits=min(summary['complete_native_TRAIN_case_visits']),
            no_weight_updates=True,test_targets_read=False)
        results.append(result);save(folder/(row['id']+'.json'),result)
        print('DIAGNOSED',row['id'],'TRAIN',phase_metrics['train']['mean_raw_case_balanced_loss'],'VAL',actual,'replay',matched,all_cases_matched,flush=True)
        del model,cp;gc.collect()
    save(folder/'complete.json',dict(status='FROZEN_TRAIN_VAL_CPU_DIAGNOSTIC_COMPLETE',models=results,
        all_CPU_validation_means_match_stored_GPU=all(r['CPU_validation_mean_matches_stored_GPU'] for r in results),
        all_CPU_validation_cases_match_stored_GPU=all(r['CPU_validation_cases_match_stored_GPU'] for r in results),
        family=name,no_training=True,test_targets_read=False,
        elapsed_seconds=time.monotonic()-started,threads=torch.get_num_threads(),
        host_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    try:main(a.index,a.root)
    except Exception:
        save(a.root/('failure_'+str(a.index)+'.json'),dict(error=traceback.format_exc(),test_targets_read=False));raise
