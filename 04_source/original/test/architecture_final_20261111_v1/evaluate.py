"""Full-field accuracy and serialized CUDA/host latency for50 frozen models."""
import argparse
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import traceback
import numpy as np
import torch
from common import SOURCE,read,save,digest,dependencies,legacy_paths


class Data:
    def __init__(self,config,root,role,device='cuda'):
        assert role in ['validation','test'];prepared=read(config['prepared_path']);cache=Path(prepared['cache'])
        self.split=read(config['split_path'])['indices'];self.normalization=read(cache/'normalization.json');self.inputs=read(cache/'inputs.json')
        assert [len(self.split[k]) for k in ['train','validation','test']]==[640,160,200]
        if role=='test':
            ready=read(root/'preflight/ready.json');assert ready['status']=='FINAL50_VALIDATION_PREFLIGHT_PASS'
            assert ready['models_sha256']==config['models_sha256'] and not ready['test_targets_read']
            self.ids=self.split['test'];targets=np.load(prepared['rawtest_targets_sealed_in'],mmap_mode='r',allow_pickle=False)
        else:
            self.ids=self.split['validation'][:config['preflight_validation_cases']]
            targets=np.load(cache/'Y_validation.npy',mmap_mode='r',allow_pickle=False)
        from ldc_data_v25 import canonical_target
        fluids=np.load(cache/'fluid.npy',mmap_mode='r',allow_pickle=False);raw=np.load(cache.parent/'raw/X_original.npy',mmap_mode='r',allow_pickle=False)
        if role=='test':
            # Verify the frozen canonicalizer against existing VAL cache only
            # after the TEST file-access gate has passed.
            check=np.load(cache/'Y_validation.npy',mmap_mode='r',allow_pickle=False)
            for index in self.split['validation'][:2]:
                assert np.array_equal(canonical_target(targets[index],fluids[index]),check[self.split['validation'].index(index)])
            save(root/('test_access_'+device+'.json'),dict(role='TEST',opened_unix=time.time(),models_sha256=config['models_sha256'],
                test_ids=self.ids,validation_canonicalization_checks=2,test_previously_opened_in_old_study=True))
        self.known={};self.reference={};self.fluid={}
        for index in self.ids:
            fluid=np.array(fluids[index],dtype=bool,copy=True);s=self.inputs['schema']
            sdf=np.array(raw[index,s['sdf_channel']],dtype=np.float32,copy=True)*s['sdf_orientation']
            geometry=torch.tensor(np.stack((fluid.astype('float32'),sdf)),device=device)
            re=float(self.inputs['reynolds'][index])
            self.known[index]=dict(geometry=geometry,geometry_native=geometry[None],fluid=torch.tensor(fluid,device=device),
                re=torch.tensor(re,device=device,dtype=torch.float32),reynolds=re,index=index)
            target=canonical_target(targets[index],fluid) if role=='test' else np.array(targets[self.split['validation'].index(index)],copy=True)
            target=np.moveaxis(target,0,2).copy();assert target.shape==(512,512,3) and np.isfinite(target).all()
            assert np.count_nonzero(target[~fluid])==0 and abs(target[:,:,2][fluid].mean(dtype='float64'))<1e-5
            self.reference[index]=target;self.fluid[index]=fluid


class Loaded:
    def __init__(self,row,normalization):
        assert digest(row['checkpoint'])==row['checkpoint_sha256'];cp=torch.load(row['checkpoint'],map_location='cpu',weights_only=True)
        assert cp['normalization']==normalization and cp['config']['seed']==row['seed']
        self.pinn=row['model']=='pinn'
        if self.pinn:
            from ldc_shared_pinn_v26 import SharedSteadyPINN,regular_grid
            assert cp['config']==row['config'];self.model=SharedSteadyPINN(cp['config'],normalization)
            self.model.load_state_dict(cp['state'],strict=True);self.qgrid=regular_grid()
            assert sum(p.numel() for p in self.model.parameters())==134922
        else:
            from checkpoint_io import restore_model
            assert cp['model']==row['model'] and not cp['test_targets_read']
            for name,sha in cp['source_sha256'].items():assert digest(Path(cp['source'])/name)==sha
            self.model=restore_model(cp)
        self.model=self.model.cuda().eval().requires_grad_(False)
        for value in self.model.state_dict().values():
            if isinstance(value,torch.Tensor):assert torch.isfinite(value).all()

    def __call__(self,known):
        if self.pinn:
            from ldc_store_v26 import query
            return query(self.model,known,self.qgrid)
        return self.model(known)


def environment():
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]
    h=subprocess.run(['nvidia-smi','-i',visible,'--query-gpu=name,uuid,driver_version,temperature.gpu,clocks.sm,power.draw,pstate','--format=csv,noheader'],capture_output=True,text=True)
    return dict(torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        hardware=h.stdout.strip(),hardware_error=h.stderr.strip(),slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        cpu_affinity=sorted(os.sched_getaffinity(0)),threads=torch.get_num_threads(),matmul_tf32=False,cudnn_tf32=False)


def stats(values,target):
    return {**{'p%d_ms'%p:float(np.percentile(values,p)) for p in [50,95,99]},
            'maximum_ms':float(values.max()),'above_target':int(np.sum(values>target))}


def measure(loaded,known,warmup,repetitions,target):
    with torch.inference_mode():
        for _ in range(warmup):loaded(known)
        torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        a.record();b.record();b.synchronize()  # initialize both events before measured calls
        gpu=np.empty(repetitions,dtype='float64');wall=np.empty_like(gpu)
        for i in range(repetitions):
            start=time.perf_counter_ns();a.record();value=loaded(known);b.record();b.synchronize()
            wall[i]=(time.perf_counter_ns()-start)/1e6;gpu[i]=a.elapsed_time(b)
    assert np.isfinite(gpu).all() and np.isfinite(wall).all() and np.all(gpu>0) and np.all(wall>0)
    return value,dict(gpu=stats(gpu,target),host_instrumented=stats(wall,target)),gpu,wall


def check_field(prediction,fluid):
    assert prediction.shape==(512,512,3) and np.isfinite(prediction).all()
    assert np.count_nonzero(prediction[~fluid])==0
    assert abs(prediction[:,:,2][fluid].mean(dtype='float64'))<1e-5


def instrumentation_overhead(target):
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);gpu=[];wall=[]
    a.record();b.record();b.synchronize()
    for _ in range(1000):
        start=time.perf_counter_ns();a.record();b.record();b.synchronize()
        wall.append((time.perf_counter_ns()-start)/1e6);gpu.append(a.elapsed_time(b))
    return dict(gpu=stats(np.asarray(gpu),target),host_instrumented=stats(np.asarray(wall),target),subtracted_from_measurements=False)


def preflight(config,root,models):
    from data_metrics import field_metrics
    out=root/'preflight';out.mkdir(parents=True,exist_ok=False);data=Data(config,root,'validation');rows=[];start=time.monotonic()
    for row in models:
        loaded=Loaded(row,data.normalization);cases=[]
        for index in data.ids:
            known=data.known[index]
            with torch.inference_mode():before=loaded(known).cpu().numpy()
            value,timing,_,_=measure(loaded,known,config['preflight_warmup'],config['preflight_repetitions'],config['latency_target_ms'])
            prediction=value.cpu().numpy();check_field(prediction,data.fluid[index]);assert np.allclose(before,prediction,rtol=1e-6,atol=1e-7)
            score=field_metrics(prediction,data.reference[index],data.fluid[index],known['reynolds'])
            expected=next(r for r in row['expected_validation'] if r['index']==index)
            actual=dict(mse=score['all_cell_mse'],velocity_l2=score['fluid_velocity_relative_l2'],pressure_l2=score['fluid_pressure_relative_l2'])
            tolerance=1e-3 if row['model']=='pinn' else 5e-5
            for key in actual:assert np.isclose(actual[key],expected[key],rtol=tolerance,atol=1e-9),(row['id'],index,key,actual[key],expected[key])
            cases.append(dict(index=index,metrics=actual,timing=timing))
        assert digest(row['checkpoint'])==row['checkpoint_sha256']
        result=dict(id=row['id'],status='MODEL_FINAL_PREFLIGHT_PASS',cases=cases);rows.append(result)
        save(out/'progress.json',rows);print('PREFLIGHT',row['id'],flush=True)
        del loaded;gc.collect();torch.cuda.empty_cache()
    save(out/'ready.json',dict(status='FINAL50_VALIDATION_PREFLIGHT_PASS',models=rows,models_sha256=config['models_sha256'],
        validation_fields=2*len(rows),test_targets_read=False,wall_seconds=time.monotonic()-start,environment=environment(),
        instrumentation_overhead=instrumentation_overhead(config['latency_target_ms'])))


def evaluate(config,root,models):
    from data_metrics import field_metrics,plot_comparison
    output=root/'test';output.mkdir(parents=True,exist_ok=False);data=Data(config,root,'test');records=[]
    for row in models:
        folder=output/'models'/row['id'];folder.mkdir(parents=True,exist_ok=False);started=time.monotonic();cpu=time.process_time()
        loaded=Loaded(row,data.normalization);before=environment();torch.cuda.synchronize()
        baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        predicted=np.lib.format.open_memmap(folder/'predictions.npy',mode='w+',dtype='float32',shape=tuple(config['prediction_shape']))
        cases=[];gpu_traces=[];wall_traces=[]
        for position,index in enumerate(data.ids):
            known=data.known[index];value,timing,gpu,wall=measure(loaded,known,config['latency_warmup'],config['latency_repetitions'],config['latency_target_ms'])
            prediction=value.cpu().numpy();check_field(prediction,data.fluid[index]);predicted[position]=prediction
            score=field_metrics(prediction,data.reference[index],data.fluid[index],known['reynolds'])
            cases.append(dict(index=index,reynolds=known['reynolds'],geometry_hash=data.inputs['geometry_hashes'][index],**score,timing=timing))
            gpu_traces.append(gpu);wall_traces.append(wall)
            if position==0:plot_comparison(folder/'comparison.png',prediction,data.reference[index],data.fluid[index],row['id']+' / TEST '+str(index))
            if (position+1)%10==0:
                save(folder/'progress.json',dict(completed_cases=position+1,total_cases=200,last_index=index,elapsed_seconds=time.monotonic()-started))
                print('TEST_PROGRESS',row['id'],position+1,'of200',flush=True)
        predicted.flush();del predicted
        gpu=np.stack(gpu_traces);wall=np.stack(wall_traces);assert gpu.shape==wall.shape==(200,config['latency_repetitions'])
        np.savez_compressed(folder/'raw_timings.npz',ids=np.asarray(data.ids),gpu_ms=gpu,host_instrumented_ms=wall)
        assert digest(row['checkpoint'])==row['checkpoint_sha256']
        summary=dict(status='FINAL50_MODEL_TEST_COMPLETE',model=row,cases=cases,test_cases=200,no_weight_updates=True,no_test_selection=True,
            means={key:float(np.mean([r[key] for r in cases])) for key in ['all_cell_mse','fluid_velocity_relative_l2','fluid_pressure_relative_l2']},
            worst_case_gpu_p99_ms=max(r['timing']['gpu']['p99_ms'] for r in cases),
            worst_case_host_p99_ms=max(r['timing']['host_instrumented']['p99_ms'] for r in cases),
            gpu_5ms_pass=all(r['timing']['gpu']['p99_ms']<=config['latency_target_ms'] for r in cases),
            host_instrumented_5ms_pass=all(r['timing']['host_instrumented']['p99_ms']<=config['latency_target_ms'] for r in cases),
            pooled_gpu_timing=stats(gpu.flatten(),config['latency_target_ms']),pooled_host_timing=stats(wall.flatten(),config['latency_target_ms']),
            predictions_sha256=digest(folder/'predictions.npy'),timings_sha256=digest(folder/'raw_timings.npz'),
            prediction_shape=config['prediction_shape'],timing_shape=list(gpu.shape),elapsed_seconds=time.monotonic()-started,
            cpu_process_seconds=time.process_time()-cpu,peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),additional_peak_cuda_bytes=torch.cuda.max_memory_allocated()-baseline,
            environment_before=before,environment_after=environment(),models_sha256=config['models_sha256'])
        save(folder/'summary.json',summary);records.append({k:summary[k] for k in ['status','model','means','worst_case_gpu_p99_ms','worst_case_host_p99_ms','gpu_5ms_pass','host_instrumented_5ms_pass','elapsed_seconds']})
        save(output/'progress.json',records);print('MODEL_TEST_COMPLETE',row['id'],summary['means'],flush=True)
        del loaded,value;gc.collect();torch.cuda.empty_cache()
    save(output/'summary.json',dict(status='FINAL50_TEST_COMPLETE',models=records,total_models=50,total_case_predictions=10000,
        test_ids=data.ids,models_sha256=config['models_sha256'],no_weight_updates=True,no_test_selection=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['preflight','test'],required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    config=read(SOURCE/'config.json');e,c,manifest=dependencies(config);legacy_paths(config);e.configure('cuda',4,911114)
    assert torch.__version__=='2.12.0+cu130' and np.__version__=='2.5.4'
    try:
        if a.mode=='preflight':preflight(config,a.root,manifest['models'])
        else:evaluate(config,a.root,manifest['models'])
    except Exception:
        save(a.root/(a.mode+'_failure.json'),dict(error=traceback.format_exc(),mode=a.mode,models_sha256=config['models_sha256']));raise
