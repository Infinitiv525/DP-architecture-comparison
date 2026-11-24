"""Equal allocated-time learning, with the same H200 for all nine families."""
import argparse
import hashlib
import importlib
import json
import math
import os
import resource
from pathlib import Path
import sys
import time
import numpy as np
import torch
from budget import allocation_clock

HERE=Path(__file__).resolve().parent
NAMES=['fno']


def dependencies():
    from runtime import training_dependencies
    return training_dependencies()


def run(mode,index,root):
    policy,common,training,models,fit=dependencies()
    if mode=='smoke':
        name=NAMES[index];seconds=60;seed=934001;config_index=0
        task_id=name+'_smoke';reserve=10.;period=10.
    elif mode=='tuning':
        name=NAMES[index//2];config_index=index%2;seed=policy['tuning_seed'];seconds=policy['tuning_budget_seconds_per_candidate']
        task_id=name+'_t'+str(config_index);reserve=180.;period=180.
    else:
        manifest=json.loads((Path(root)/'selection.json').read_text())
        row=manifest['tasks'][index];name=row['model'];seed=row['seed'];config_index=row['configuration_index']
        seconds=policy['main_budget_seconds_per_run'];task_id=row['id'];reserve=300.;period=policy.get('final_main_evaluation_period_seconds',600.)
        if mode=='preflight':seconds=60;reserve=10.;period=10.
    common.setup('cuda',seed)
    if 'H200' not in torch.cuda.get_device_name():raise RuntimeError('Cost-matched learning requires H200')
    budget,allocation=allocation_clock(seconds)
    folder=Path(root)/mode/task_id;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():print('Already complete',task_id,flush=True);return
    started=folder/'started.json'
    if started.exists():raise RuntimeError('Previous attempt exists: audit its entire allocation cost before any resume')
    task=dict(model=name,seed=seed,configuration_index=config_index,learning_rate=[.001,.0003][config_index],
              ridge=[1e-8,1e-6][config_index],time_budget_seconds=seconds,mode=mode)
    common.save(started,dict(task=task,allocation=allocation,policy_sha256=hashlib.sha256((HERE/'policy.json').read_bytes()).hexdigest()))
    fixture=training.Fixture(policy['fixture_root'],'cuda')
    model=models.PackedFieldOperator(name,common.recipe(name),fixture.normalization).cuda()
    gradient=name not in ['elm','scn','cascor']
    optimizer=torch.optim.Adam(model.core.parameters(),lr=task['learning_rate']) if gradient else None
    counts=np.zeros(640,dtype='int64');history=[];candidate_history=[]
    best=float('inf');updates=epochs=units=0;next_evaluation=budget.used
    last_unit_seconds=0.;stop=None;last_score=None;selected_time=None

    def evaluate(event):
        nonlocal best,next_evaluation,last_score,selected_time
        torch.cuda.synchronize(); score=training.validation(model,fixture);last_score=score
        current=dict(event=event,allocation_elapsed_seconds=budget.used,epochs=epochs,updates=updates,units=units,
                     validation=score,parameter_ledger=model.parameter_ledger())
        history.append(current)
        checkpoint=dict(model=name,recipe=common.recipe(name),normalization=fixture.normalization,
            state=model.state_dict(),dynamic=training.dynamic(model),task=task,history=history,
            case_counts=counts.tolist(),candidate_history=candidate_history,
            allocation_elapsed_seconds=budget.used,allocation=allocation,optimizer=optimizer.state_dict() if optimizer else None)
        training.atomic_checkpoint(folder/'last.pt',checkpoint)
        if score['mean_raw_case_balanced_loss']<best:
            best=score['mean_raw_case_balanced_loss'];selected_time=budget.used
            training.atomic_checkpoint(folder/'best.pt',checkpoint)
        common.save(folder/'history.json',history)
        next_evaluation=budget.used+period
        print(task_id,event,'seconds',round(budget.used,1),'epoch',epochs,'units',units,'VAL',score['mean_raw_case_balanced_loss'],flush=True)

    evaluate('initial')
    if gradient:
        while budget.can_start(reserve,max(1.,last_unit_seconds*2)):
            epochs+=1
            generator=torch.Generator(device='cpu').manual_seed(seed+epochs)
            order=torch.randperm(640,generator=generator).tolist();epoch_complete=True
            for index in order:
                if not budget.can_start(reserve,max(.05,last_unit_seconds*2)):
                    epoch_complete=False;stop='training_time_budget';break
                fraction=min(1.,budget.used/max(1.,seconds-reserve))
                lr=1e-5+.5*(task['learning_rate']-1e-5)*(1+math.cos(math.pi*fraction))
                for group in optimizer.param_groups:group['lr']=lr
                x,y,w=(a[index] for a in fixture.tables['train']);begin=time.monotonic()
                model.train();optimizer.zero_grad(set_to_none=True)
                loss=common.patch_loss(common.patch_prediction(model.core,name,x),y,w)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite common native TRAIN objective')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.core.parameters(),1.);optimizer.step()
                if hasattr(model.core,'project_parameters'):model.core.project_parameters()
                _=float(loss.detach());last_unit_seconds=time.monotonic()-begin
                counts[index]+=1;updates+=1
                if budget.used>=next_evaluation:evaluate('during_training')
            if not epoch_complete:break
        stop=stop or 'training_time_budget'
    else:
        if budget.can_start(reserve):
            begin=time.monotonic();fit.fit_output(model.core,lambda:fixture.iterator(),ridge=task['ridge']);counts+=1
            last_unit_seconds=time.monotonic()-begin;evaluate('first_complete_output_fit')
        else:stop='training_time_budget_before_first_fit'
        if name=='elm' and stop is None:stop='algorithm_completed'
        maximum=common.recipe(name)['width']
        while stop is None and units<maximum:
            if not budget.can_start(reserve,max(1.,last_unit_seconds*2)):
                stop='training_time_budget';break
            begin=time.monotonic()
            if name=='scn':
                node,meta=fit.scn_candidate(model.core,lambda:fixture.iterator(),pool=16,seed=seed+units+1)
                if node is None:stop=meta['stop'];candidate_history.append(dict(unit=units+1,**meta));break
                fit.append_scn(model.core,node)
            else:
                node,meta=fit.cascade_candidate(model.core,lambda:fixture.iterator(),pool=8,epochs=1)
                model.core.add(node)
            units+=1;candidate_history.append(dict(unit=units,**meta))
            output_fit=fit.fit_output(model.core,lambda:fixture.iterator(),ridge=task['ridge']);counts+=1
            torch.cuda.synchronize();last_unit_seconds=time.monotonic()-begin
            if budget.used>=next_evaluation:evaluate('during_growth')
        if stop is None:stop='planned_capacity_reached'
    evaluate('training_finished')
    selected=torch.load(folder/'best.pt',map_location='cuda',weights_only=False)
    selected_model=models.restore_model(selected).cuda().eval()
    physical=training.physical_validation_metrics(selected_model,fixture)
    torch.cuda.synchronize()
    status='MAIN_PREFLIGHT_COMPLETE' if mode=='preflight' else ('BUDGETED_TRAINING_COMPLETE' if mode!='smoke' else 'BUDGET_WORKER_SMOKE_COMPLETE')
    selected_ledger=selected_model.parameter_ledger();selected_ledger['accounting_only']=False
    common.save(folder/'complete.json',dict(status=status,task=task,stop_reason=stop,
        limit_seconds=seconds,allocation_elapsed_seconds=budget.used,within_time_limit=budget.used<=seconds,
        best_complete_validation_loss=best,best_checkpoint_selected_at_seconds=selected_time,
        physical_validation_metrics=physical,selected_parameter_ledger=selected_ledger,
        last_parameter_ledger=model.parameter_ledger(),selected_checkpoint=str(folder/'best.pt'),
        selected_checkpoint_sha256=hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest(),
        complete_native_TRAIN_case_visits=counts.tolist(),gradient_updates=updates,epochs_started=epochs,
        final_grown_units=units,candidate_history=candidate_history,history=history,
        accuracy_threshold=None,test_targets_read=False,hardware=torch.cuda.get_device_name(),
        peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),
        allocated_CPU_cores=4,allocated_H200_GPUs=1,
        host_CPU_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,
        host_CPU_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,
        host_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        scope='Timing/control integration only, not final scientific evidence' if mode in ['smoke','preflight'] else 'Complete budgeted TRAIN/VAL fit',
        accounting_note='Entire allocation runtime is the cost, not the earlier selected checkpoint time. Final scheduler runtime must also be collected.'))
    print(task_id,status,'used_seconds',round(budget.used,2),'limit',seconds,'reason',stop,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','tuning','main','preflight'],required=True)
    p.add_argument('--index',type=int,required=True);p.add_argument('--root',required=True)
    a=p.parse_args();run(a.mode,a.index,a.root)
