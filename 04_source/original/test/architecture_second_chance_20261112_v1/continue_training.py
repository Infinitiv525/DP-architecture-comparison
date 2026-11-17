"""Exploratory extra4h from last weights+Adam+data cursor; never a main run."""
import argparse
import gc
from pathlib import Path
import resource
import time
import numpy as np
import torch
from budget import allocation_clock
from runtime import HERE,read,digest,dependencies,recovered_cursor,optimizer_steps,check_parents


def main(index,root):
    policy,parents,common,training,models=dependencies();row=parents['tasks'][index]
    name=row['model'];seed=row['seed'];limit=policy['additional_GPU_seconds_limit']
    common.setup('cuda',seed)
    assert 'H200' in torch.cuda.get_device_name() and torch.cuda.device_count()==1
    budget,allocation=allocation_clock(limit)
    folder=root/'continuation'/row['id'];folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'started.json').exists(),'Preserve an existing attempt, never reset its budget silently'
    check_parents(row)
    cp=torch.load(row['original_last'],map_location='cpu',weights_only=False)
    assert cp['model']==name and cp['normalization']==read(Path(policy['fixture_root'])/'fixture/ready.json')['normalization']
    model=models.restore_model(cp).cuda()
    optimizer=torch.optim.Adam(model.core.parameters(),lr=cp['task']['learning_rate'])
    optimizer.load_state_dict(cp['optimizer'])
    fixed_lrs=[g['lr'] for g in optimizer.param_groups]
    assert fixed_lrs==row['constant_learning_rates']
    counts=np.asarray(cp['case_counts'],dtype='int64');base_counts=counts.copy()
    updates=int(counts.sum());base_updates=updates
    epoch,position,order=recovered_cursor(counts,seed,updates)
    assert dict(epoch=epoch,position=position,global_updates=updates)==row['resume_cursor']
    assert all(int(v['step'])==updates for v in optimizer_steps(optimizer))
    task=dict(id=row['id'],model=name,seed=seed,mode='exploratory_continuation',
        excluded_from_primary_experiment=True,additional_GPU_seconds_limit=limit,
        original_nominal_GPU_seconds=14400,cumulative_nominal_GPU_seconds=28800,
        constant_saved_learning_rates=fixed_lrs,no_optimizer_reset=True,no_architecture_change=True,
        no_TEST_targets=True,original_last_sha256=row['original_last_sha256'],
        parent_inspection_sha256=policy['parents_sha256'])
    common.save(folder/'started.json',dict(task=task,allocation=allocation,
        recovered_cursor=row['resume_cursor'],initial_optimizer_steps=optimizer_steps(optimizer),
        policy_sha256=digest(HERE/'policy.json'),original_actual_GPU_seconds=row['original_allocated_GPU_seconds']))
    fixture=training.Fixture(policy['fixture_root'],'cuda')
    assert fixture.normalization==cp['normalization']
    history=[];best_new=float('inf');best_new_score=None;selected_at=None
    initial_verified=False;next_eval=budget.used;last_step_seconds=0.;reserve=300.
    del cp;gc.collect()

    def evaluate(event):
        nonlocal best_new,best_new_score,selected_at,next_eval,initial_verified
        torch.cuda.synchronize();score=training.validation(model,fixture)
        if not initial_verified:
            assert np.allclose(score['case_losses'],row['last_validation_GPU']['case_losses'],rtol=.001,atol=1e-7)
            initial_verified=True
        current=dict(event=event,additional_allocation_seconds=budget.used,
            cumulative_allocated_GPU_seconds=row['original_allocated_GPU_seconds']+budget.used,
            epochs=epoch,position_in_epoch=position,updates=updates,new_updates=updates-base_updates,
            validation=score,optimizer_steps=optimizer_steps(optimizer))
        history.append(current)
        state=dict(model=name,recipe=common.recipe(name),normalization=fixture.normalization,
            state=model.state_dict(),dynamic=training.dynamic(model),task=task,optimizer=optimizer.state_dict(),
            history=history,case_counts=counts.tolist(),
            cursor=dict(epoch=epoch,position=position,global_updates=updates),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            parent=dict(last_path=row['original_last'],last_sha256=row['original_last_sha256'],
                        best_path=row['original_best'],best_sha256=row['original_best_sha256']),
            additional_allocation_seconds=budget.used,excluded_from_primary_experiment=True,
            test_targets_read=False)
        training.atomic_checkpoint(folder/'last.pt',state)
        if score['mean_raw_case_balanced_loss']<best_new:
            best_new=score['mean_raw_case_balanced_loss'];best_new_score=score;selected_at=budget.used
            training.atomic_checkpoint(folder/'best.pt',state)
        common.save(folder/'history.json',history)
        next_eval=budget.used+300.
        print(row['id'],event,'extra_seconds',round(budget.used,1),'updates',updates,'VAL',score['mean_raw_case_balanced_loss'],flush=True)

    evaluate('restored_original_last_no_updates')
    while budget.can_start(reserve,max(.05,last_step_seconds*2)):
        index=order[position];x,y,w=(a[index] for a in fixture.tables['train']);start=time.monotonic()
        model.train();optimizer.zero_grad(set_to_none=True)
        loss=common.patch_loss(common.patch_prediction(model.core,name,x),y,w)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite unchanged native TRAIN objective')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.core.parameters(),1.);optimizer.step()
        _=float(loss.detach());last_step_seconds=time.monotonic()-start
        counts[index]+=1;updates+=1;position+=1
        if position==640:
            epoch+=1;position=0
            order=torch.randperm(640,generator=torch.Generator(device='cpu').manual_seed(seed+epoch)).tolist()
        if budget.used>=next_eval:evaluate('during_exploratory_continuation')
    evaluate('continuation_finished')
    recovered_cursor(counts,seed,updates)
    assert [g['lr'] for g in optimizer.param_groups]==fixed_lrs
    assert min(counts-base_counts)>0 and updates-base_updates==int((counts-base_counts).sum())
    best_new_sha=digest(folder/'best.pt');last_sha=digest(folder/'last.pt')
    improved=best_new<row['original_best_VAL_loss']
    chosen=folder/'best.pt' if improved else Path(row['original_best'])
    selected_score=best_new_score if improved else min(read(row['original_complete'])['history'],key=lambda v:v['validation']['mean_raw_case_balanced_loss'])['validation']
    check_parents(row)
    common.save(folder/'complete.json',dict(status='EXPLORATORY_CONTINUATION_COMPLETE',task=task,
        additional_allocation_seconds=budget.used,within_additional_limit=budget.used<=limit,
        original_allocated_GPU_seconds=row['original_allocated_GPU_seconds'],
        cumulative_observed_GPU_seconds=row['original_allocated_GPU_seconds']+budget.used,
        original_best_VAL_loss=row['original_best_VAL_loss'],best_new_VAL_loss=best_new,
        last_VAL_loss=history[-1]['validation']['mean_raw_case_balanced_loss'],
        best_new_validation=best_new_score,last_validation=history[-1]['validation'],selected_validation=selected_score,
        selected_checkpoint=str(chosen),selected_checkpoint_sha256=digest(chosen),
        selected_is_original_baseline=not improved,best_new_selected_at_additional_seconds=selected_at,
        new_best_checkpoint=str(folder/'best.pt'),new_best_sha256=best_new_sha,
        new_last_checkpoint=str(folder/'last.pt'),new_last_sha256=last_sha,
        global_updates=updates,additional_updates=updates-base_updates,
        initial_case_counts=base_counts.tolist(),final_case_counts=counts.tolist(),
        final_cursor=dict(epoch=epoch,position=position,global_updates=updates),
        optimizer_steps=optimizer_steps(optimizer),history=history,
        actual_GPU_model=torch.cuda.get_device_name(),allocated_CPU_cores=4,allocated_H200_GPUs=1,
        peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),
        host_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        original_files_unchanged=True,test_targets_read=False,excluded_from_primary_experiment=True,
        note='Best-so-far can retain original baseline; new-phase best and final values remain reported, including any deterioration. Integer update and visit counters are authoritative, not floating Adam step tensors.'))
    print(row['id'],'EXPLORATORY_CONTINUATION_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();main(a.index,a.root)
