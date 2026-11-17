"""Independent numerical artifact audit and longer common-H200 latency measurements."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import torch
from common import SOURCE,read,save,digest,engine


def finite_tree(value):
    if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(finite_tree(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(finite_tree(v) for v in value)
    if isinstance(value,float):return bool(np.isfinite(value))
    return True


def independent(pred,reference,fluid):
    pred=pred.astype('float64');reference=reference.astype('float64');error=pred-reference
    mse=np.mean(error*error,axis=(0,1))
    return dict(mse=float(mse.mean()),mse_fields=mse.tolist(),
        velocity_l2=float(np.linalg.norm(error[fluid,:2])/max(np.linalg.norm(reference[fluid,:2]),1e-12)),
        pressure_l2=float(np.linalg.norm(error[fluid,2])/max(np.linalg.norm(reference[fluid,2]),1e-12)))


def hardware():
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]
    try:
        r=subprocess.run(['nvidia-smi','-i',visible,'--query-gpu=name,uuid,driver_version,temperature.gpu,clocks.sm,power.draw,pstate','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
        return dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
    except Exception as error:return dict(error=str(error))


def latency(model,known,config):
    with torch.inference_mode():
        for _ in range(config['latency_warmup']):model(known)
        torch.cuda.synchronize();pairs=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(config['latency_repetitions'])]
        for start,end in pairs:start.record();model(known);end.record()
        torch.cuda.synchronize();values=np.asarray([a.elapsed_time(b) for a,b in pairs],dtype='float64')
    return dict(p50_ms=float(np.percentile(values,50)),p95_ms=float(np.percentile(values,95)),p99_ms=float(np.percentile(values,99)),max_ms=float(values.max())),values


def audit_model(e,c,config,store,model,checkpoint,folder,recorded):
    from data_metrics import field_metrics
    assert finite_tree(checkpoint['state']) and checkpoint['normalization']==store.normalization
    for name,sha in checkpoint['source_sha256'].items():assert digest(Path(checkpoint['source'])/name)==sha
    rows=[];max_difference=0.
    with torch.inference_mode():
        for old in recorded['cases']:
            known,target=store.get(old['index']);pred=model(known).cpu().numpy();ref=target.cpu().numpy();fluid=known['fluid'].cpu().numpy()
            assert pred.shape==(512,512,3) and np.isfinite(pred).all() and np.count_nonzero(pred[~fluid])==0
            assert abs(pred[:,:,2][fluid].mean(dtype='float64'))<1e-5
            metrics=independent(pred,ref,fluid)
            for key in ['mse','mse_fields','velocity_l2','pressure_l2']:
                assert np.allclose(metrics[key],old[key],rtol=5e-5,atol=1e-11),(old['index'],key,metrics[key],old[key])
            max_difference=max(max_difference,abs(metrics['mse']-old['mse']))
            rows.append(dict(index=old['index'],**metrics))
    # Stored raw fields are checked more strictly; no model/backend roundoff enters here.
    raw_count=0
    for index in store.val_ids[:4]:
        path=folder/('prediction_%04d.npz'%index)
        if not path.exists():continue
        known,target=store.get(index)
        with np.load(path,allow_pickle=False) as raw:pred=raw['prediction']
        again=field_metrics(pred,target.cpu().numpy(),known['fluid'].cpu().numpy(),known['reynolds'])
        old=read(folder/('field_metrics_%04d.json'%index))
        for key in ['all_cell_mse','fluid_velocity_relative_l2','fluid_pressure_relative_l2']:
            assert np.isclose(again[key],old[key],rtol=1e-12,atol=1e-14)
        raw_count+=1
    core=model.core
    counts=dict(core_parameter_elements=sum(p.numel() for p in core.parameters()),
        core_parameter_real_scalar_equivalents=sum(p.numel()*(2 if p.is_complex() else 1) for p in core.parameters()),
        core_buffer_elements=sum(p.numel() for p in core.buffers()),
        core_buffer_real_scalar_equivalents=sum(p.numel()*(2 if p.is_complex() else 1) for p in core.buffers()),
        registered_pipeline_buffer_bytes=sum(b.numel()*b.element_size() for b in model.buffers()))
    return dict(status='MODEL_ARTIFACT_AUDIT_PASS',validation_cases_recomputed=len(rows),stored_fields_recomputed=raw_count,
        max_absolute_mse_difference=max_difference,mean_validation_mse=float(np.mean([r['mse'] for r in rows])),counts=counts,cases=rows)


def main(config,output,extended=False):
    e,c=engine(config);e.configure('cuda',4,c['calibration_seed'])
    from checkpoint_io import restore_model
    output.mkdir(parents=True,exist_ok=False);store=e.Store(c,'cuda')
    for index in store.val_ids:store.get(index)
    records=[];calibration=Path(config['calibration_run'])
    recipes=[r for r in c['recipes'] if not extended or r['id'] in config['continuation_recipes']]
    for recipe in recipes:
        rid=recipe['id'];folder=output/rid;folder.mkdir()
        if extended:
            training=Path(config['review_run'])/'extensions'/rid
            s=read(training/'summary.json');checkpoint_path=Path(s['best_checkpoint'])
            assert s['status']=='KAN_CONTINUATION_COMPLETE' and digest(checkpoint_path)==s['best_checkpoint_sha256']
            recorded=None
        else:
            training=calibration/'training'/rid;s=read(training/'summary.json');checkpoint_path=training/'best.pt'
            assert s['status']=='CALIBRATION_TRAINING_COMPLETE' and digest(checkpoint_path)==s['best_checkpoint_sha256']
            recorded=read(calibration/'evaluation'/rid/'validation.json')
        data=torch.load(checkpoint_path,map_location='cpu',weights_only=True);assert data['model']==recipe['model']
        model=restore_model(data).cuda().eval();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        if recorded is None:recorded=e.validate(model,store,store.val_ids)
        assert np.isclose(recorded['mse'],data['validation_mse'],rtol=5e-5,atol=1e-11)
        audit=audit_model(e,c,config,store,model,data,folder if extended else calibration/'evaluation'/rid,recorded)
        save(folder/'artifact_audit.json',audit);blocks=[];traces=[];before=hardware()
        repetitions=config['repeat_blocks'] if rid in config['repeat_models'] else 1
        for block in range(repetitions):
            block_rows=[];raw=[]
            for index in store.val_ids[:config['latency_cases']]:
                known,_=store.get(index);stat,values=latency(model,known,config)
                block_rows.append(dict(index=index,**stat));raw.append(values)
            blocks.append(dict(block=block,cases=block_rows));traces.append(np.stack(raw))
            save(folder/'latency_blocks.json',blocks)
        np.savez_compressed(folder/'raw_timings.npz',gpu_ms=np.stack(traces),ids=np.asarray(store.val_ids[:config['latency_cases']]))
        worst=max(row['p99_ms'] for block in blocks for row in block['cases'])
        summary=dict(id=rid,status='REVIEW_COMPLETE',model=recipe['model'],checkpoint=str(checkpoint_path),checkpoint_sha256=digest(checkpoint_path),
            validation_mse=audit['mean_validation_mse'],worst_gpu_p99_ms=worst,latency_pass=worst<=config['latency_target_ms'],
            timing_blocks=repetitions,repetitions_per_case=config['latency_repetitions'],warmup=config['latency_warmup'],
            counts=audit['counts'],hardware_before=before,hardware_after=hardware(),
            additional_peak_inference_bytes=torch.cuda.max_memory_allocated()-baseline,test_targets_read=False)
        save(folder/'summary.json',summary);records.append(summary);save(output/'progress.json',records)
        print('REVIEWED',json.dumps(summary),flush=True);del model,data;gc.collect();torch.cuda.empty_cache()
    save(output/'summary.json',dict(status='CALIBRATION_REVIEW_COMPLETE',extended=extended,models=records,
        regenerated_validation_fields=160*len(records),test_targets_read=False,final_repetitions_started=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--extended',action='store_true')
    a=p.parse_args();config=read(SOURCE/'config.json');config['review_run']=str(a.run_root);main(config,a.output,a.extended)
