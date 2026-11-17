"""Full TRAIN/VAL capacity calibration; no TEST target access or final repetitions."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback
import numpy as np
import torch

SOURCE=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def dependencies(config):
    for folder,files in config['dependencies'].items():
        for name,sha in files.items():
            if digest(Path(folder)/name)!=sha:raise ValueError('Dependency hash mismatch: '+name)
        sys.path.insert(0,folder)


def configure(device,threads,seed):
    if len(os.sched_getaffinity(0))<threads:raise RuntimeError('CPU affinity smaller than requested thread count')
    torch.set_num_threads(threads);torch.set_num_interop_threads(1);torch.manual_seed(seed);np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    if device=='cuda' and (not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'H200' not in torch.cuda.get_device_name()):
        raise RuntimeError('Exactly one allocated H200 required')
    if device=='cpu' and torch.cuda.is_available():raise RuntimeError('CPU task unexpectedly sees GPU')


def sync(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize()


class Store:
    def __init__(self,config,device,cache_train=False):
        root=Path(config['root']);prepared=read(root/config['prepared_path']);assert prepared['status']=='LDC_DATA_READY'
        self.split=read(root/config['split_path'])['indices'];self.train_ids=self.split['train'];self.val_ids=self.split['validation']
        assert len(self.train_ids)==640 and len(self.val_ids)==160 and len(self.split['test'])==200
        assert len(set(self.train_ids+self.val_ids+self.split['test']))==1000
        self.cache=Path(prepared['cache']);self.normalization=read(self.cache/'normalization.json');self.inputs=read(self.cache/'inputs.json')
        self.raw=np.load(self.cache.parent/'raw/X_original.npy',mmap_mode='r',allow_pickle=False)
        self.fluid=np.load(self.cache/'fluid.npy',mmap_mode='r',allow_pickle=False)
        self.y={'train':np.load(self.cache/'Y_train.npy',mmap_mode='r',allow_pickle=False),
                'validation':np.load(self.cache/'Y_validation.npy',mmap_mode='r',allow_pickle=False)}
        self.slots={i:('train',p) for p,i in enumerate(self.train_ids)};self.slots.update({i:('validation',p) for p,i in enumerate(self.val_ids)})
        self.device=device;self.known_cache={};self.target_cache={}
        self.plan=dict(train_ids=self.train_ids,validation_ids=self.val_ids,revision=prepared['revision'],
            split_sha256=digest(root/config['split_path']),prepared_sha256=digest(root/config['prepared_path']),
            normalization=self.normalization,test_targets_read=False,known_geometry_split=True)
        if cache_train:
            for i in self.train_ids:self.get(i)

    def get(self,index):
        if index not in self.slots:raise ValueError('Only frozen TRAIN/VAL indices may be loaded')
        if index not in self.known_cache:
            phase,slot=self.slots[index];s=self.inputs['schema'];fluid=np.array(self.fluid[index],dtype=bool,copy=True)
            sdf=np.array(self.raw[index,s['sdf_channel']],dtype=np.float32,copy=True)*s['sdf_orientation']
            geometry=np.stack((fluid.astype('float32'),sdf));target=np.array(self.y[phase][slot],dtype=np.float32,copy=True).transpose(1,2,0).copy()
            assert geometry.shape==(2,512,512) and target.shape==(512,512,3)
            if not np.isfinite(geometry).all() or not np.isfinite(target).all():raise FloatingPointError('Nonfinite native input/target')
            if np.count_nonzero(target[~fluid]) or abs(target[:,:,2][fluid].mean(dtype=np.float64))>1e-5:raise ValueError('Reference gauge/mask mismatch')
            re=float(self.inputs['reynolds'][index]);assert re>0
            self.known_cache[index]=dict(geometry=torch.tensor(geometry,device=self.device),fluid=torch.tensor(fluid,device=self.device),
                                          re=torch.tensor(re,device=self.device),reynolds=re,index=index)
            self.target_cache[index]=torch.tensor(target,device=self.device)
        return self.known_cache[index],self.target_cache[index]


def point_data(config,device):
    folder=Path(config['point_fixture']);meta=read(folder/'fixture.json')
    assert meta['status']=='MATCHED_FIXTURE_READY' and meta['train_cases']==640 and meta['point_samples']==1310720 and meta['point_features']==57
    arrays=[]
    for name in ['x.npy','y.npy']:
        if digest(folder/name)!=meta['files'][name]['sha256']:raise ValueError('Training point fixture mismatch')
        value=np.load(folder/name,allow_pickle=False)
        if not np.isfinite(value).all():raise FloatingPointError('Nonfinite training samples')
        arrays.append(torch.from_numpy(value).to(device))
    return *arrays,meta


def recipe_config(config,recipe):
    c=dict(config['model_defaults']);c.update(recipe['overrides']);c['seed']=config['calibration_seed'];c['tfno']=dict(c['tfno'])
    return c


def make_model(name,c,normalization):
    from models import PointPipeline,AuthorTFNOPipeline
    return AuthorTFNOPipeline(normalization,c['tfno']) if name=='tfno' else PointPipeline(name,normalization,c)


def rbf_initialize(model,x,seed):
    rng=np.random.default_rng(seed+3300);indices=rng.choice(len(x),len(model.core.centres),replace=False)
    centres=x[torch.tensor(indices,device=x.device)]
    with torch.no_grad():
        distances=torch.cdist(centres.double(),centres.double()).square()/x.shape[1]
        radius=(distances[distances>1e-12].median()/2).sqrt().to(x.dtype)
        if not torch.isfinite(radius) or radius<=0:raise FloatingPointError('Invalid RBF initial radius')
        model.core.centres.copy_(centres);model.core.log_width.fill_(float(torch.log(torch.expm1(radius))))
    return dict(rule='centres sampled without targets from TRAIN point inputs; radius from median pairwise squared distance',
                indices_sha256=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),initial_radius=float(radius))


def validate(model,store,ids):
    was_training=model.training;model.eval();rows=[]
    with torch.inference_mode():
        for index in ids:
            known,target=store.get(index);pred=model(known)
            if pred.shape!=(512,512,3) or not torch.isfinite(pred).all():raise FloatingPointError('Invalid validation field')
            error=pred.double()-target.double();fluid=known['fluid'];mse=error.square().mean((0,1))
            rows.append(dict(index=index,mse=float(mse.mean()),mse_fields=mse.cpu().tolist(),
                velocity_l2=float(error[fluid,:2].norm()/target.double()[fluid,:2].norm().clamp_min(1e-12)),
                pressure_l2=float(error[fluid,2].norm()/target.double()[fluid,2].norm().clamp_min(1e-12))))
    model.train(was_training)
    return dict(mse=float(np.mean([r['mse'] for r in rows])),velocity_l2=float(np.mean([r['velocity_l2'] for r in rows])),
                pressure_l2=float(np.mean([r['pressure_l2'] for r in rows])),cases=rows)


def checkpoint(model,name,c,store,path,epoch,score,dynamic=None,optimizer=None,rng=None,scheduler=None):
    value=dict(model=name,state=model.state_dict(),config=c,normalization=store.normalization,dynamic=dynamic or {},
        epoch=epoch,validation_mse=score,source=str(SOURCE),source_sha256={p.name:digest(p) for p in SOURCE.iterdir() if p.is_file()},
        train_ids_sha256=hashlib.sha256(json.dumps(store.train_ids).encode()).hexdigest(),test_targets_read=False)
    temporary=path.with_suffix('.tmp');torch.save(value,temporary);temporary.replace(path)
    if optimizer is not None:
        state=dict(optimizer=optimizer.state_dict(),epoch=epoch,torch_rng=torch.get_rng_state(),
                   numpy_rng=rng.bit_generator.state if rng is not None else None,
                   scheduler=None if scheduler is None else scheduler.state_dict())
        if next(model.parameters()).device.type=='cuda':state['cuda_rng']=torch.cuda.get_rng_state()
        temporary=path.with_name(path.stem+'_optimizer.tmp');torch.save(state,temporary);temporary.replace(path.with_name(path.stem+'_optimizer.pt'))


def train_gradient(model,name,c,config,store,x,y,output,device):
    rng=np.random.default_rng(c['seed']+1);optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['max_epochs'],eta_min=config['min_learning_rate'])
    history=[];initial=validate(model,store,store.val_ids);best=initial['mse'];best_epoch=0;bad=0
    checkpoint(model,name,c,store,output/'best.pt',0,best,optimizer=optimizer,rng=rng,scheduler=scheduler)
    save(output/'validation_epoch_0000.json',initial)
    started=time.monotonic();last_epoch=0
    for epoch in range(1,config['max_epochs']+1):
        model.train();loss_sum=0.;count=0
        order=rng.permutation(len(store.train_ids) if name=='tfno' else len(x))
        batches=(order if name=='tfno' else [order[k:k+c['gradient_batch_points']] for k in range(0,len(order),c['gradient_batch_points'])])
        for batch in batches:
            if name=='tfno':
                index=store.train_ids[int(batch)];known,target=store.get(index);pred=model(known)
                loss=((pred-target)/model.std).square()[known['fluid']].mean()
            else:
                indices=torch.from_numpy(batch).to(device);loss=(model.core(x[indices])-y[indices]).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite gradient loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            if hasattr(getattr(model,'core',None),'project_parameters'):model.core.project_parameters()
            loss_sum+=float(loss.detach());count+=1
        scheduler.step();sync(device);last_epoch=epoch
        row=dict(epoch=epoch,train_normalized_loss=loss_sum/count,updates=count,lr=optimizer.param_groups[0]['lr'],wall_seconds=time.monotonic()-started)
        if epoch%config['assess_every']==0 or epoch==config['max_epochs']:
            score=validate(model,store,store.val_ids);row.update(validation_mse=score['mse'],velocity_l2=score['velocity_l2'],pressure_l2=score['pressure_l2'])
            save(output/('validation_epoch_%04d.json'%epoch),score)
            if score['mse']<best:
                best=score['mse'];best_epoch=epoch;bad=0;checkpoint(model,name,c,store,output/'best.pt',epoch,best,optimizer=optimizer,rng=rng,scheduler=scheduler)
            else:bad+=1
        history.append(row);save(output/'history.json',history)
        print('EPOCH',json.dumps(row),flush=True)
        if epoch>=config['minimum_epochs'] and bad>=config['patience_assessments']:break
    checkpoint(model,name,c,store,output/'last.pt',last_epoch,history[-1].get('validation_mse',best),optimizer=optimizer,rng=rng,scheduler=scheduler)
    return dict(algorithm='Adam, clip1, cosine learning rate, normalized fluid data loss',history=history,
                initial_validation_mse=initial['mse'],best_validation_mse=best,best_epoch=best_epoch,last_epoch=last_epoch,
                optimizer_updates=sum(h['updates'] for h in history),training_wall_seconds=time.monotonic()-started)


def train_constructive(model,name,c,config,store,x,y,output):
    from models import fit_elm,fit_cascade
    initial=validate(model,store,store.val_ids);started=time.monotonic();ridge=config['ridge_mean']*len(x)
    if name=='elm':history=fit_elm(model.core,x,y,ridge=ridge)
    elif name=='cascor':history=fit_cascade(model.core,x,y,width=c['constructive_width'],pool=c['cascade_pool'],candidate_steps=c['cascade_candidate_steps'],ridge=ridge)
    else:
        path=Path(config['hardware_source'])/'bench.py';spec=importlib.util.spec_from_file_location('hardware_reference',path);reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
        generator=torch.Generator(device='cpu').manual_seed(c['seed']+23);weights=[];biases=[]
        for _ in range(c['constructive_width']):
            weights.append(torch.rand(4,c['constructive_pool'],x.shape[1],generator=generator)*2-1)
            biases.append(torch.rand(4,c['constructive_pool'],generator=generator)*2-1)
        bank=dict(weight=torch.stack(weights).to(x.device),bias=torch.stack(biases).to(x.device))
        torch.save({k:v.cpu() for k,v in bank.items()},output/'candidate_bank.pt')
        history=reference.fit_scn_canonical(model.core,x,y,bank,dict(c,ridge=ridge))
    sync(str(x.device));fit_seconds=time.monotonic()-started;save(output/'constructive_history.json',history)
    score=validate(model,store,store.val_ids)
    dynamic=dict(scn_units=len(model.core.weight)) if name=='scn' else (dict(cascor_units=len(model.core.units)) if name=='cascor' else {})
    checkpoint(model,name,c,store,output/'best.pt',1,score['mse'],dynamic=dynamic)
    return dict(algorithm=name,history=history,initial_validation_mse=initial['mse'],best_validation_mse=score['mse'],best_epoch=1,last_epoch=1,
                ridge_mean=config['ridge_mean'],ridge_sum_equivalent=ridge,fit_wall_seconds=fit_seconds,validation=score,dynamic=dynamic)


def prepare(config,output):
    from checks import run
    from models import PointPipeline
    output.mkdir(parents=True,exist_ok=False);core=run();torch.set_num_threads(4)
    x,y,meta=point_data(config,'cpu');store=Store(config,'cpu')
    assert meta['data_plan']['train_ids']==store.train_ids and meta['data_plan']['normalization']==store.normalization
    # Verify native targets and shared point labels refer to the same physical scale.
    known,target=store.get(store.train_ids[0]);assert known['index'] not in store.val_ids
    for recipe in config['recipes']:
        c=recipe_config(config,recipe);model=make_model(recipe['model'],c,store.normalization)
        if recipe['model']=='rbf':rbf_initialize(model,x,c['seed'])
        if recipe['model']!='tfno':
            z=model.core(x[:17]);assert z.shape==(17,3) and torch.isfinite(z).all()
        del model
    # Exercise the new native scorer/checkpoint/scheduler path before dependent jobs allocate GPUs.
    from checkpoint_io import restore_model
    from data_metrics import field_metrics
    tiny=Store(config,'cpu');tiny.train_ids=tiny.train_ids[:2];tiny.val_ids=tiny.val_ids[:2]
    recipe=next(r for r in config['recipes'] if r['id']=='pinet_a');c=recipe_config(config,recipe)
    torch.manual_seed(c['seed']);model=make_model('pinet',c,store.normalization)
    small=output/'preflight';small.mkdir()
    check=train_gradient(model,'pinet',dict(c,gradient_batch_points=16),dict(config,max_epochs=2,assess_every=1,minimum_epochs=1),tiny,x[:32],y[:32],small,'cpu')
    loaded=torch.load(small/'best.pt',map_location='cpu',weights_only=True);restored=restore_model(loaded)
    score=validate(restored,tiny,tiny.val_ids)
    assert np.isclose(score['mse'],loaded['validation_mse'],rtol=1e-12,atol=1e-14)
    opt=torch.load(small/'last_optimizer.pt',map_location='cpu',weights_only=True)
    assert opt['scheduler']['T_max']==2 and opt['epoch']==2
    for index in tiny.val_ids:
        known,target=tiny.get(index)
        with torch.inference_mode():pred=restored(known).numpy()
        ref=field_metrics(pred,target.numpy(),known['fluid'].numpy(),known['reynolds'])
        actual=next(r['mse'] for r in score['cases'] if r['index']==index)
        assert np.isclose(ref['all_cell_mse'],actual,rtol=1e-12,atol=1e-14)
    save(output/'preflight_result.json',dict(status='CALIBRATION_PREFLIGHT_PASS',gradient_updates=check['optimizer_updates'],native_validation_cases=2,strict_restore=True,scheduler_state_restored=True))
    save(output/'ready.json',dict(status='FULL_TRAIN_VAL_CALIBRATION_READY',core_checks=core,point_samples=len(x),
        point_sample_sha256=meta['point_sample_sha256'],fixture_sha256=digest(Path(config['point_fixture'])/'fixture.json'),
        data_plan=store.plan,config=config,test_targets_read=False))
    print('FULL_TRAIN_VAL_CALIBRATION_READY',flush=True)


def trial(config,recipe,output,device):
    name=recipe['model'];c=recipe_config(config,recipe);output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();cpu_start=time.process_time();store=Store(config,device,cache_train=name=='tfno')
    x=y=meta=None
    if name!='tfno':x,y,meta=point_data(config,device)
    torch.manual_seed(c['seed']);model=make_model(name,c,store.normalization).to(device)
    initialization=rbf_initialize(model,x,c['seed']) if name=='rbf' else dict(rule='seeded model constructor')
    save(output/'data_plan.json',store.plan);save(output/'recipe.json',recipe);save(output/'model_config.json',c)
    save(output/'source_freeze.json',dict(files={p.name:digest(p) for p in SOURCE.iterdir() if p.is_file()},source=str(SOURCE),slurm_job_id=os.environ.get('SLURM_JOB_ID')))
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    if name in ['elm','scn','cascor']:training=train_constructive(model,name,c,config,store,x,y,output)
    else:training=train_gradient(model,name,c,config,store,x,y,output,device)
    save(output/'training_summary.json',training)
    core=model.core;parameters=sum(p.numel() for p in core.parameters());buffers=sum(p.numel() for p in core.buffers())
    fitted=core.beta.numel() if name=='elm' else buffers if name=='scn' else parameters+core.beta.numel() if name=='cascor' else parameters
    summary=dict(status='CALIBRATION_TRAINING_COMPLETE',recipe=recipe,model=name,device=device,initialization=initialization,training=training,
        best_checkpoint=str(output/'best.pt'),best_checkpoint_sha256=digest(output/'best.pt'),
        core_parameters=parameters,core_buffers=buffers,fitted_or_selected_core_scalars=fitted,
        wall_seconds=time.monotonic()-start,cpu_process_seconds=time.process_time()-cpu_start,
        peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
        torch_threads=torch.get_num_threads(),cpu_affinity=sorted(os.sched_getaffinity(0)),torch=torch.__version__,
        point_sample_sha256=None if meta is None else meta['point_sample_sha256'],test_targets_read=False,
        final_five_repetitions_started=False)
    save(output/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='training'},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','train']);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--recipe');p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,default=4)
    a=p.parse_args()
    try:
        config=read(SOURCE/'config.json');dependencies(config);configure(a.device,a.threads,config['calibration_seed'])
        if a.phase=='prepare':prepare(config,a.output)
        else:
            recipe=next(r for r in config['recipes'] if r['id']==a.recipe)
            if recipe['device']!=a.device:raise ValueError('Frozen recipe/backend mismatch')
            trial(config,recipe,a.output,a.device)
    except Exception:
        traceback.print_exc();raise
