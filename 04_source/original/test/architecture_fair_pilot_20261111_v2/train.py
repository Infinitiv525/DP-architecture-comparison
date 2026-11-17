"""Shared full-field training; gradient, direct ELM, SCN and CasCor algorithms.

No shared accuracy cutoff. Checkpoints and full-size fields remain on Perun.
Pilot is distinct from the later five-seed cohort and never reads TEST targets.
"""
import argparse
import copy
import gc
import hashlib
import os
from pathlib import Path
import time
import numpy as np
import torch
from common import HERE, patch_loss, patch_prediction, read, recipe, save, setup, sync
from operator_models import PackedFieldOperator, restore_model
from weighted_fit import append_scn, cascade_candidate, fit_output, scn_candidate


def atomic_checkpoint(path,value):
    path=Path(path); temp=path.with_suffix('.tmp'); torch.save(value,temp); temp.replace(path)


class Fixture:
    def __init__(self,root,device):
        folder=Path(root)/'fixture';self.metadata=read(folder/'ready.json')
        assert self.metadata['status']=='FULL_NATIVE_TRAIN_VAL_FIXTURE_READY'
        assert not self.metadata['test_targets_read']
        self.normalization=self.metadata['normalization']; self.tables={}
        start=time.perf_counter()
        for phase in ['train','validation']:
            self.tables[phase]=tuple(torch.from_numpy(np.load(folder/(phase+'_'+label+'.npy'),allow_pickle=False)).to(device)
                for label in ['x','y','weights'])
        self.load_seconds=time.perf_counter()-start
    def iterator(self,phase='train',order=None):
        x,y,w=self.tables[phase]
        for i in (range(len(x)) if order is None else order):yield x[i],y[i],w[i]


def validation(model,fixture):
    model.eval();scores=[]
    with torch.no_grad():
        for x,y,w in fixture.iterator('validation'):
            value=patch_prediction(model.core,model.name,x)
            score=patch_loss(value,y,w)
            if not torch.isfinite(score):raise FloatingPointError('Nonfinite complete VAL loss')
            scores.append(float(score))
    return dict(mean_raw_case_balanced_loss=float(np.mean(scores)),case_losses=scores,cases=len(scores))


def physical_validation_metrics(model,fixture):
    records=[]; adapter=model.adapter
    with torch.no_grad():
        for x,y,w in fixture.iterator('validation'):
            patch=patch_prediction(model.core,model.name,x)
            field=adapter.decode_raw(patch.T.reshape(1,192,64,64))
            target=adapter.decode_raw(y.T.reshape(1,192,64,64))
            geometry=torch.nn.functional.pixel_shuffle(x[:,:128].T.reshape(1,128,64,64),8)
            geometry=geometry*adapter.geometry_std+adapter.geometry_mean
            fluid=geometry[:,0]>.5
            projected=adapter.project_known_constraints(field,fluid)
            error=(projected-target)*fluid[:,None]
            uv=float(error[:,:2].norm()/target[:,:2].norm().clamp_min(1e-12))
            pressure=float(error[:,2].norm()/target[:,2].norm().clamp_min(1e-12))
            records.append(dict(velocity_relative_L2=uv,pressure_relative_L2=pressure))
    return dict(case_metrics=records,mean_velocity_relative_L2=float(np.mean([r['velocity_relative_L2'] for r in records])),
        mean_pressure_relative_L2=float(np.mean([r['pressure_relative_L2'] for r in records])),
        processing='Shared solid mask and zero-mean fluid pressure; complete native fields')


def dynamic(model):
    if model.name=='scn':return dict(scn_units=len(model.core.weight))
    if model.name=='cascor':return dict(cascor_units=len(model.core.units))
    return {}


def main(root,task_index):
    root=Path(root); manifest=read(HERE/'pilot_tasks.json');task=manifest['tasks'][task_index]
    name=task['model']; device=task['device'];seed=task['seed'];setup(device,seed)
    folder=root/'pilot'/task['id'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():print('Already complete',task['id'],flush=True);return
    begin=time.perf_counter();fixture=Fixture(root,device)
    model=PackedFieldOperator(name,recipe(name),fixture.normalization).to(device)
    gradients=name not in ['elm','scn','cascor']
    optimizer=torch.optim.Adam(model.core.parameters(),lr=task['learning_rate']) if gradients else None
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=task['epochs'],eta_min=1e-5) if gradients else None
    start_unit=0;history=[]; best=float('inf');case_counts=np.zeros(640,dtype='int64')
    previous_seconds=0; candidate_history=[]
    def checkpoint(unit,score,is_best=False):
        value=dict(model=name,recipe=recipe(name),normalization=fixture.normalization,
            state=model.state_dict(),dynamic=dynamic(model),task=task,unit=unit,history=history,
            best_validation=best,case_counts=case_counts.tolist(),candidate_history=candidate_history,
            optimizer=optimizer.state_dict() if optimizer else None,scheduler=scheduler.state_dict() if scheduler else None,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else None,
            elapsed_training_seconds=previous_seconds+time.perf_counter()-begin,
            test_targets_read=False,score=score)
        atomic_checkpoint(folder/'last.pt',value)
        if is_best:atomic_checkpoint(folder/'best.pt',value)
    if (folder/'last.pt').exists():
        saved=torch.load(folder/'last.pt',map_location=device,weights_only=False)
        assert saved['task']==task
        model=restore_model(saved).to(device);start_unit=saved['unit'];history=saved['history'];best=saved['best_validation']
        case_counts=np.array(saved['case_counts']);candidate_history=saved['candidate_history'];previous_seconds=saved['elapsed_training_seconds']
        if gradients:
            optimizer=torch.optim.Adam(model.core.parameters(),lr=task['learning_rate']);optimizer.load_state_dict(saved['optimizer'])
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=task['epochs'],eta_min=1e-5);scheduler.load_state_dict(saved['scheduler'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device=='cuda':torch.cuda.set_rng_state_all([r.cpu() for r in saved['cuda_rng']])
    stop_reason=None
    limit=task['epochs'] if gradients else 1 if name=='elm' else task['constructive_units']
    for unit in range(start_unit+1,limit+1):
        model.train();start=time.perf_counter();losses=[];fit_meta=None
        if gradients:
            generator=torch.Generator(device='cpu').manual_seed(seed+unit)
            order=torch.randperm(640,generator=generator).tolist()
            for index in order:
                x,y,w=(a[index] for a in fixture.tables['train'])
                optimizer.zero_grad(set_to_none=True);prediction=patch_prediction(model.core,name,x)
                loss=patch_loss(prediction,y,w)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite full-field TRAIN loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.core.parameters(),1.);optimizer.step()
                if hasattr(model.core,'project_parameters'):model.core.project_parameters()
                losses.append(float(loss.detach()));case_counts[index]+=1
            scheduler.step()
        elif name=='elm':
            fit_meta=fit_output(model.core,lambda:fixture.iterator(),ridge=task['ridge']);case_counts+=1
        else:
            if unit==1:
                fit_output(model.core,lambda:fixture.iterator(),ridge=task['ridge']);case_counts+=1
            if name=='scn':
                node,meta=scn_candidate(model.core,lambda:fixture.iterator(),pool=task['candidate_pool'],seed=seed+unit)
                candidate_history.append(dict(unit=unit,**meta))
                if node is None:stop_reason=meta['stop']
                else:append_scn(model.core,node)
            else:
                node,meta=cascade_candidate(model.core,lambda:fixture.iterator(),pool=task['candidate_pool'],epochs=task['candidate_epochs'])
                candidate_history.append(dict(unit=unit,**meta));model.core.add(node)
            if stop_reason is None:
                fit_meta=fit_output(model.core,lambda:fixture.iterator(),ridge=task['ridge']);case_counts+=1
        sync(device); duration=time.perf_counter()-start
        evaluate=unit==1 or unit==limit or unit%task['validation_every']==0 or stop_reason is not None
        score=validation(model,fixture) if evaluate else None
        record=dict(unit=unit,unit_seconds=duration,train_online_loss=float(np.mean(losses)) if losses else None,
            validation=score,fit=fit_meta,stop_reason=stop_reason)
        history.append(record)
        improved=score is not None and score['mean_raw_case_balanced_loss']<best
        if improved:best=score['mean_raw_case_balanced_loss']
        if evaluate:checkpoint(unit,score,improved)
        save(folder/'progress.json',dict(status='RUNNING',task=task,unit=unit,limit=limit,
            best_validation=best,latest=record,total_seconds=previous_seconds+time.perf_counter()-begin,
            complete_native_TRAIN_case_visits=case_counts.tolist(),test_targets_read=False))
        print(task['id'],unit,limit,'VAL',score['mean_raw_case_balanced_loss'] if score else '-',flush=True)
        if stop_reason is not None:break
        # A bounded continuation uses last.pt; no silent Slurm requeue or budget expansion.
        if time.perf_counter()-begin>task.get('soft_time_limit_seconds',6800):
            if not evaluate:checkpoint(unit,None,False)
            save(folder/'continuation_required.json',dict(status='CHECKPOINT_SAVED_CONTINUATION_REQUIRED',unit=unit,limit=limit))
            return
    # Evaluate complete TRAIN and VAL for the actual selected checkpoint. Never
    # call the final unit the selected model when an earlier unit won on VAL.
    # Keep the final grown capacity for constructive methods. Selecting an early
    # smaller model on VAL would silently undo the parameter-budget control.
    selected_path=folder/('best.pt' if gradients else 'last.pt')
    selected=torch.load(selected_path,map_location=device,weights_only=False)
    best_model=restore_model(selected).to(device).eval()
    with torch.no_grad():
        train_scores=[float(patch_loss(patch_prediction(best_model.core,name,x),y,w)) for x,y,w in fixture.iterator()]
    val=validation(best_model,fixture)
    physical=physical_validation_metrics(best_model,fixture)
    save(folder/'history.json',history)
    ledger=best_model.parameter_ledger();ledger['accounting_only']=False
    sha=hashlib.sha256(selected_path.read_bytes()).hexdigest()
    save(folder/'complete.json',dict(status='FULL_TRAIN_VAL_PILOT_COMPLETE',task=task,
        selected_unit=selected['unit'],completed_units=history[-1]['unit'],stop_reason=stop_reason,
        train_mean_raw_case_balanced_loss=float(np.mean(train_scores)),train_case_losses=train_scores,
        validation=val,physical_validation_metrics=physical,selected_parameter_ledger=ledger,checkpoint_sha256=sha,
        selected_checkpoint=str(selected_path),fixture_load_seconds=fixture.load_seconds,
        selection_policy='best VAL epoch at unchanged capacity' if gradients else 'last grown capacity; best earlier VAL remains recorded separately',
        capacity_stage='partial-growth pilot, not final parameter-matched cohort' if name in ['scn','cascor'] else 'full planned core capacity',
        total_seconds=previous_seconds+time.perf_counter()-begin,
        complete_native_TRAIN_case_visits=case_counts.tolist(),
        training_case_visits_note='Counts output-fit passes/gradient epochs; constructive candidate passes are additional and recorded separately.',
        test_targets_read=False,accuracy_threshold=None))
    print(task['id'],'FULL_TRAIN_VAL_PILOT_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--task',type=int,required=True)
    a=p.parse_args();main(a.root,a.task)
