"""Single known-scenario matched pilot, never independent generalization evidence."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from fpo_stable_v6 import (StablePINN,DT,grid,features,conditioning,trajectory,load_cases,
                            fingerprint,errors,residual,stable_step,grid_physics)
from fpo_common_v6 import WORK,save,digest,verify
from fpo_spatial_v10 import SpatialPINN as StablePINN,stable_step,features,conditioning


def tensor(a):return torch.as_tensor(a,dtype=torch.float32,device='cuda')


def continuous_physics(model,raw,c,case,rng,active,config):
    pde=torch.zeros((),device='cuda')
    if active:
        yy,xx=case['interior'][rng.integers(len(case['interior']),size=config['collocation'])].T
        q=tensor(np.column_stack((4+(xx+.5+rng.uniform(-.4,.4,len(xx)))/64,
                                  6+(yy+.5+rng.uniform(-.4,.4,len(yy)))/64,rng.uniform(0,DT,len(xx))))).requires_grad_(True)
        pde=residual(q,model(q,c,raw,case['weight']),case['record']['reynolds']).square().sum(1).mean()
    b=case['boundary'][rng.integers(len(case['boundary']),size=128)]
    q=tensor(np.column_stack((b,rng.uniform(0,DT,len(b))))).requires_grad_(True)
    bc=(model(q,c,raw,case['weight'])[:,:2]/model.std[:2]).square().mean()
    return pde,bc


def assess(model,case,out,qgrid,config):
    out.mkdir(parents=True,exist_ok=False);model.eval()
    if fingerprint(case['record']['field'])!=case['record']['field_fingerprint']:raise ValueError('Reference changed')
    previous,current=case['prefix'];initial=current.cpu().numpy()[case['fluid_np']]
    rows=[];step_times=[];checked_times=[];failure=None;scale=model.std[:2].mean()
    yy,xx=case['interior'].T
    with torch.inference_mode():
        initial_div=float(grid_physics(current,current,case,scale)[1].sqrt()*scale)
        for offset in range(1,239):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
            a.record();pred=stable_step(model,previous,current,case,qgrid);b.record()
            mom,div=grid_physics(current,pred,case,scale);e.record();e.synchronize()
            step_times.append(a.elapsed_time(b));checked_times.append(a.elapsed_time(e))
            p=pred.cpu().numpy()
            if not np.isfinite(p).all() or np.max(np.abs(p))>config['divergence_limit']:
                failure=dict(offset=offset,reason='nonfinite or abs signal above1000');break
            reference=np.asarray(case['reference'][offset+3]);ref=reference[case['fluid_np']]
            refdiv=(reference[yy,xx+1,0]-reference[yy,xx-1,0]+reference[yy+1,xx,1]-reference[yy-1,xx,1])*32
            rows.append(dict(offset=offset,frame=offset+3,time=offset*DT,model=errors(p[case['fluid_np']],ref),
                             persistence=errors(initial,ref),own_fd_momentum_loss=float(mom),own_fd_divergence_loss=float(div),
                             own_divergence_rms=float(div.sqrt()*scale),reference_divergence_rms=float(np.sqrt(np.mean(refdiv**2)))))
            previous,current=current,pred
    scores={};mass_pass=False
    if failure is None:
        groups={'all_future':rows,'steps_1_10':rows[:10],'steps_11_40':rows[10:40],'steps_41_120':rows[40:120],'steps_121_238':rows[120:]}
        scores={scope:{kind:{metric:float(np.mean([r[kind][metric] for r in group])) for metric in group[0][kind]}
                       for kind in ['model','persistence']} for scope,group in groups.items()}
        mean_div=float(np.mean([r['own_divergence_rms'] for r in rows]));ref_div=float(np.mean([r['reference_divergence_rms'] for r in rows]))
        mass_limit=max(3*initial_div,3*ref_div,1e-3);mass_pass=mean_div<=mass_limit
        score=(scores['all_future']['model']['velocity_relative_l2']+scores['all_future']['model']['pressure_relative_l2'])/2
        mass=dict(initial_rms=initial_div,mean_model_rms=mean_div,mean_reference_rms=ref_div,prospective_limit=mass_limit,pass_gate=mass_pass)
    else:score=config['divergence_score_penalty'];mass=dict(initial_rms=initial_div,pass_gate=False)
    result=dict(score=score,scores=scores,divergence=failure,predicted_steps=len(rows),mass=mass,
                gpu_step_ms={k:float(np.percentile(step_times,p)) for k,p in [('p50',50),('p95',95),('p99',99)]},
                gpu_step_plus_fd_monitor_ms={k:float(np.percentile(checked_times,p)) for k,p in [('p50',50),('p95',95),('p99',99)]},
                reference_future_as_input=False,scope='KNOWN_TRAINING_SCENARIO_POSITIVE_CONTROL')
    save(out/'per_frame.json',rows);save(out/'summary.json',result);model.train();return result


def benchmark(model,case,qgrid):
    model.eval();previous,current=case['prefix'];rounds=[];monitored=[]
    with torch.inference_mode():
        for _ in range(30):stable_step(model,previous,current,case,qgrid)
        for _ in range(3):
            times=[];full=[]
            for i in range(200):
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);c=torch.cuda.Event(enable_timing=True)
                a.record();value=stable_step(model,previous,current,case,qgrid);b.record()
                grid_physics(current,value,case,model.std[:2].mean());c.record();c.synchronize()
                times.append(a.elapsed_time(b));full.append(a.elapsed_time(c))
            rounds.append({k:float(np.percentile(times,p)) for k,p in [('p50',50),('p95',95),('p99',99)]})
            monitored.append({k:float(np.percentile(full,p)) for k,p in [('p50',50),('p95',95),('p99',99)]})
    return dict(grid=[1024,256],whole_gpu_step_rounds_ms=rounds,step_plus_fd_monitor_rounds_ms=monitored,
                samples_per_round=200,rendering_included=False,initial_tensors_gpu=True)


def train(config,case,out,qgrid,normal):
    out.mkdir(parents=True,exist_ok=False);save(out/'config.json',config)
    torch.manual_seed(config['seed']);rng=np.random.default_rng(config['seed'])
    model=StablePINN(config,normal['mean'],normal['std']).cuda();best_key=None;history=[]
    start=time.monotonic();coverage=np.zeros(238,dtype=bool);torch.cuda.reset_peak_memory_stats()
    for phase,updates,length,lr in [('one_step',config['one_step_updates'],1,config['lr']),
                                    ('own_rollout',config['rollout_updates'],config['rollout_length'],config['rollout_lr'])]:
        if phase=='own_rollout':
            chosen=torch.load(out/'best.pt',weights_only=True);torch.save(chosen,out/'one_step_best.pt');model.load_state_dict(chosen['state'])
        optimizer=torch.optim.Adam(model.parameters(),lr=lr)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=updates,eta_min=lr*.05)
        windows=list(range(1,240-length));order=rng.permutation(windows);cursor=0
        for update in range(1,updates+1):
            if cursor==len(order):order=rng.permutation(windows);cursor=0
            k=int(order[cursor]);cursor+=1;previous,current=case['data'][k-1],case['data'][k]
            raw=features(previous,current);c=conditioning(model,raw,case['geometry'],case['record']['reynolds'])
            optimizer.zero_grad(set_to_none=True);own_mom=torch.zeros((),device='cuda');own_div=own_mom
            if length==1:
                yy,xx=case['fluid_yx'][rng.integers(len(case['fluid_yx']),size=config['batch'])].T
                y=torch.as_tensor(yy,device='cuda');x=torch.as_tensor(xx,device='cuda')
                q=tensor(np.column_stack((4+(xx+.5)/64,6+(yy+.5)/64,np.full(len(xx),DT))))
                pred=model.sparse_prediction(q,c,raw,case['weight'],y,x)
                data=((pred-case['data'][k+1,y,x])/(model.std*config['data_loss_scale'])).square().mean();coverage[k-1]=True
            else:
                losses=[];ms=[];ds=[];old=current
                for offset,pred in enumerate(trajectory(previous,current,lambda a,b:stable_step(model,a,b,case,qgrid),length),1):
                    losses.append(((pred[case['fluid']]-case['data'][k+offset][case['fluid']])/(model.std*config['data_loss_scale'])).square().mean())
                    if config['own_momentum_weight'] or config['own_divergence_weight']:
                        m,d=grid_physics(old,pred,case,model.std[:2].mean());ms.append(m);ds.append(d)
                    old=pred
                data=torch.stack(losses).mean()
                if ms:own_mom=torch.stack(ms).mean();own_div=torch.stack(ds).mean()
            ramp=1. if phase=='own_rollout' else min(1.,max(0.,(update-config['warmup'])/config['ramp']))
            pde,bc=continuous_physics(model,raw,c,case,rng,ramp>0,config)
            loss=data+ramp*config['pde_weight']*pde+config['bc_weight']*bc+config['own_momentum_weight']*own_mom+config['own_divergence_weight']*own_div
            if not torch.isfinite(loss):
                save(out/'failure.json',dict(phase=phase,update=update,reason='nonfinite loss'));raise RuntimeError('Nonfinite loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();scheduler.step()
            if update==1 or update%config['validate_every']==0 or update==updates:
                label=f'{phase}_{update:06d}';result=assess(model,case,out/'assessments'/label,qgrid,config)
                row=dict(phase=phase,update=update,data=float(data.detach()),pde=float(pde.detach()),boundary=float(bc.detach()),
                         own_momentum=float(own_mom.detach()),own_divergence=float(own_div.detach()),
                         full238_score=result['score'],mass=result['mass'],elapsed_seconds=time.monotonic()-start)
                history.append(row);save(out/'history.json',history)
                selection_key=(0 if result['divergence'] is None and result['mass']['pass_gate'] else 1,result['score'])
                state=dict(config=config,state=model.state_dict(),phase=phase,update=update,score=result['score'],selection_key=selection_key,source_freeze=digest(WORK/'freeze.json'))
                torch.save(state,out/'last.pt')
                if best_key is None or selection_key<best_key:best_key=selection_key;torch.save(state,out/'best.pt')
                print(config['name'],label,'SCORE',result['score'],'MASS',result['mass'],flush=True)
    elapsed=time.monotonic()-start;chosen=torch.load(out/'best.pt',weights_only=True);model.load_state_dict(chosen['state'])
    selected=assess(model,case,out/'selected_checkpoint',qgrid,config);timing=benchmark(model,case,qgrid)
    accurate=selected['divergence'] is None and selected['scores']['all_future']['model']['velocity_relative_l2']<=.1 and selected['scores']['all_future']['model']['pressure_relative_l2']<=.2
    fast=all(r['p99']<=5 for r in timing['whole_gpu_step_rounds_ms']) and selected['gpu_step_ms']['p99']<=5
    result=dict(status='KNOWN_CASE_STABILIZATION_PILOT_EVALUATED',config=config,selected_phase=chosen['phase'],selected_update=chosen['update'],
                accuracy=selected,latency=timing,training_seconds=elapsed,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                parameter_count=sum(p.numel() for p in model.parameters()),gpu=torch.cuda.get_device_name(),
                working_gate=dict(accuracy_pass=bool(accurate),latency_pass=fast,mass_growth_pass=selected['mass']['pass_gate']),
                independent_validation=False,main_test_opened=False,llm_comparison_ready=False,
                all_training_transitions_visited=bool(coverage.all()),checkpoint_sha256=digest(out/'best.pt'),
                limitations=['Same known training trajectory is used for positive-control selection; not heldout accuracy',
                             'Curl divergence guarantee applies to increments away from solid-mask stencils, not all obstacle-boundary cells',
                             'Continuous PDE derivatives and grid data derivatives use documented distinct discretizations'])
    save(out/'summary.json',result);return result


def run():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    verify();torch.set_num_threads(4)
    if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name():raise RuntimeError('H200 required')
    receipt=json.loads((WORK/'prepared.json').read_text());record=receipt['records'][0]
    if record['proposed_split']!='train':raise ValueError('Positive control must be original train case')
    case=load_cases([record])[0]
    if digest(receipt['weight_file'])!=receipt['weight_sha256']:raise ValueError('Geometry weight changed')
    with np.load(receipt['weight_file'],allow_pickle=False) as z:case['weight']=tensor(z['weight'])
    case['interior_gpu']=torch.zeros_like(case['fluid']);y,x=case['interior'].T;case['interior_gpu'][y,x]=True
    with np.load(record['state_cache'],allow_pickle=False) as z:raw=tensor(z['features'])
    case['prefix']=(raw[:,:,:3]-raw[:,:,3:],raw[:,:,:3]);case['reference']=np.load(record['field'],mmap_mode='r',allow_pickle=False)
    normal=json.loads(Path(receipt['normalization']).read_text());qgrid=grid()
    configs=json.loads((WORK/'protocol.json').read_text())['candidates']
    out=WORK/('smoke' if args.smoke else 'training')/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=False)
    save(out/'normalization.json',normal);results=[]
    for config in configs:
        if args.smoke:config=dict(config,one_step_updates=3,rollout_updates=2,validate_every=3,warmup=0,ramp=1)
        results.append(train(config,case,out/config['name'],qgrid,normal));torch.cuda.empty_cache()
    save(out/'summary.json',dict(status='SMOKE_COMPLETE' if args.smoke else 'MATCHED_KNOWN_CASE_PILOT_COMPLETE',
                                 results=results,independent_validation=False))


if __name__=='__main__':run()
