"""One common H200 evaluates all calibration checkpoints on full VAL, never TEST."""
import argparse
import csv
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
import train


def timing(model,known,config):
    with torch.inference_mode():
        for _ in range(config['latency_warmup']):model(known)
        torch.cuda.synchronize()
        pairs=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(config['latency_repetitions'])]
        for start,end in pairs:start.record();model(known);end.record()
        torch.cuda.synchronize();values=np.asarray([a.elapsed_time(b) for a,b in pairs],dtype=np.float64)
    return dict(p50_ms=float(np.percentile(values,50)),p95_ms=float(np.percentile(values,95)),
                p99_ms=float(np.percentile(values,99)),max_ms=float(values.max())),values


def main(config,run_root,output):
    from checkpoint_io import restore_model
    from data_metrics import plot_comparison,field_metrics
    output.mkdir(parents=True,exist_ok=False);store=train.Store(config,'cuda');records=[]
    for recipe in config['recipes']:
        folder=run_root/'training'/recipe['id'];result=output/recipe['id'];result.mkdir()
        summary=folder/'summary.json'
        if not summary.exists():
            records.append(dict(id=recipe['id'],model=recipe['model'],status='TRAINING_FAILED_OR_INCOMPLETE'));continue
        s=train.read(summary)
        if s['status']!='CALIBRATION_TRAINING_COMPLETE':raise ValueError('Unexpected training status')
        if train.digest(folder/'best.pt')!=s['best_checkpoint_sha256']:raise ValueError('Best checkpoint hash mismatch')
        checkpoint=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        for name,sha in checkpoint['source_sha256'].items():
            if train.digest(Path(checkpoint['source'])/name)!=sha:raise ValueError('Checkpoint source mismatch')
        assert checkpoint['model']==recipe['model'] and checkpoint['config']==train.recipe_config(config,recipe)
        assert checkpoint['normalization']==store.normalization and checkpoint['test_targets_read'] is False
        model=restore_model(checkpoint).cuda().eval();score=train.validate(model,store,store.val_ids)
        train.save(result/'validation.json',score);times=[];traces=[]
        for position,index in enumerate(store.val_ids[:config['latency_screen_validation_cases']]):
            known,target=store.get(index);measured,raw=timing(model,known,config)
            times.append(dict(index=index,**measured));traces.append(raw)
            if position<4:
                with torch.inference_mode():pred=model(known).cpu().numpy()
                ref=target.cpu().numpy();fluid=known['fluid'].cpu().numpy()
                np.savez_compressed(result/('prediction_%04d.npz'%index),prediction=pred)
                train.save(result/('field_metrics_%04d.json'%index),field_metrics(pred,ref,fluid,known['reynolds']))
                if position==0:plot_comparison(result/'comparison.png',pred,ref,fluid,recipe['id']+' / VAL '+str(index))
        np.savez_compressed(result/'raw_timings.npz',ids=np.asarray(store.val_ids[:len(times)]),gpu_ms=np.stack(traces))
        train.save(result/'latency.json',times);worst=max(r['p99_ms'] for r in times)
        history=s['training'].get('history',[]);v=[r for r in history if 'validation_mse' in r]
        trend=None if len(v)<2 else (v[-2]['validation_mse']-v[-1]['validation_mse'])/max(v[-2]['validation_mse'],1e-15)
        needs_extension=bool(s['training']['last_epoch']==config['max_epochs'] and s['training']['best_epoch']==config['max_epochs'] and trend is not None and trend>.01)
        row=dict(id=recipe['id'],model=recipe['model'],status='EVALUATED',validation_mse=score['mse'],velocity_l2=score['velocity_l2'],pressure_l2=score['pressure_l2'],
            worst_screen_gpu_p99_ms=worst,latency_screen_pass=worst<=config['latency_p99_target_ms'],screen_cases=len(times),
            best_epoch=s['training']['best_epoch'],last_epoch=s['training']['last_epoch'],
            initial_validation_mse=s['training']['initial_validation_mse'],last_relative_validation_gain=trend,
            needs_longer_training_review=needs_extension,checkpoint=str(folder/'best.pt'),checkpoint_sha256=s['best_checkpoint_sha256'],
            training_device=s['device'],fitted_or_selected_core_scalars=s['fitted_or_selected_core_scalars'])
        train.save(result/'summary.json',row);records.append(row);train.save(output/'progress.json',records)
        print('EVALUATED',json.dumps(row),flush=True);del model,checkpoint;gc.collect();torch.cuda.empty_cache()
    recommendations=[]
    for name in dict.fromkeys(r['model'] for r in config['recipes']):
        available=[r for r in records if r['model']==name and r['status']=='EVALUATED']
        eligible=[r for r in available if r['latency_screen_pass'] and r['best_epoch']>0]
        best=min(eligible or available,key=lambda r:r['validation_mse']) if available else None
        recommendations.append(dict(model=name,recipe=None if best is None else best['id'],
            status='NO_SUCCESSFUL_CALIBRATION' if best is None else 'CANDIDATE_FOR_REVIEW' if eligible else 'BEST_AVAILABLE_WITH_UNMET_CRITERIA',
            needs_longer_training_review=False if best is None else best['needs_longer_training_review']))
    with (output/'comparison.csv').open('w',newline='') as f:
        columns=list(dict.fromkeys(k for row in records for k in row));w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(records)
    train.save(output/'summary.json',dict(status='CALIBRATION_EVALUATION_FINISHED',models=records,recommendations=recommendations,
               full_validation_cases=160,test_targets_read=False,final_five_repetitions_started=False,
               latency_is_screening_not_final_tail_estimate=True,automatic_final_training=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();c=train.read(train.SOURCE/'config.json');train.dependencies(c);train.configure('cuda',4,c['calibration_seed']);main(c,a.run_root,a.output)
