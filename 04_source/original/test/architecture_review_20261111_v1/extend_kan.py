"""Continue the two flagged ChebKAN calibrations at the saved minimum learning rate."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from common import SOURCE,read,save,digest,engine


def main(config,recipe_id,output):
    e,c=engine(config);e.configure('cpu',4,c['calibration_seed']);e.SOURCE=SOURCE
    from checkpoint_io import restore_model
    assert recipe_id in ['chebkan_a','chebkan_b']
    parent=Path(config['calibration_run'])/'training'/recipe_id
    previous=read(parent/'summary.json');assert previous['status']=='CALIBRATION_TRAINING_COMPLETE'
    checkpoint=torch.load(parent/'last.pt',map_location='cpu',weights_only=True)
    state=torch.load(parent/'last_optimizer.pt',map_location='cpu',weights_only=True)
    assert checkpoint['epoch']==state['epoch']==c['max_epochs']==200
    model=restore_model(checkpoint);model_config=checkpoint['config'];store=e.Store(c,'cpu')
    x,y,meta=e.point_data(c,'cpu');assert store.normalization==checkpoint['normalization']
    optimizer=torch.optim.Adam(model.parameters(),lr=model_config['learning_rate']);optimizer.load_state_dict(state['optimizer'])
    assert all(abs(g['lr']-c['min_learning_rate'])<1e-14 for g in optimizer.param_groups)
    scheduler=torch.optim.lr_scheduler.ConstantLR(optimizer,factor=1.0,total_iters=config['extra_epochs'])
    rng=np.random.default_rng();rng.bit_generator.state=state['numpy_rng'];torch.set_rng_state(state['torch_rng'])
    output.mkdir(parents=True,exist_ok=False);initial=e.validate(model,store,store.val_ids)
    assert np.isclose(initial['mse'],checkpoint['validation_mse'],rtol=1e-6,atol=1e-12)
    lineage=dict(parent_checkpoint=str(parent/'last.pt'),parent_sha256=digest(parent/'last.pt'),parent_optimizer_sha256=digest(parent/'last_optimizer.pt'),
        rule='resume all Adam moments and RNG; extra epochs at constant saved minimum LR',constant_lr=c['min_learning_rate'])
    save(output/'lineage.json',lineage);save(output/'initial_validation.json',initial)
    best=initial['mse'];best_epoch=200;history=[];bad=0;started=time.monotonic()
    e.checkpoint(model,'chebkan',model_config,store,output/'best.pt',200,best,optimizer=optimizer,rng=rng,scheduler=scheduler)
    for offset in range(1,config['extra_epochs']+1):
        epoch=200+offset;model.train();order=rng.permutation(len(x));loss_sum=0.;updates=0
        for start in range(0,len(order),model_config['gradient_batch_points']):
            indices=torch.from_numpy(order[start:start+model_config['gradient_batch_points']])
            loss=(model.core(x[indices])-y[indices]).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite continuation loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            loss_sum+=float(loss.detach());updates+=1
        scheduler.step();row=dict(epoch=epoch,updates=updates,normalized_train_loss=loss_sum/updates,lr=optimizer.param_groups[0]['lr'],wall_seconds=time.monotonic()-started)
        if offset%c['assess_every']==0 or offset==config['extra_epochs']:
            score=e.validate(model,store,store.val_ids);row.update(validation_mse=score['mse'],velocity_l2=score['velocity_l2'],pressure_l2=score['pressure_l2'])
            save(output/('validation_epoch_%04d.json'%epoch),score)
            if score['mse']<best:
                best=score['mse'];best_epoch=epoch;bad=0;e.checkpoint(model,'chebkan',model_config,store,output/'best.pt',epoch,best,optimizer=optimizer,rng=rng,scheduler=scheduler)
            else:bad+=1
        history.append(row);save(output/'history.json',history);print('EPOCH',json.dumps(row),flush=True)
        if offset>=100 and bad>=5:break
    e.checkpoint(model,'chebkan',model_config,store,output/'last.pt',epoch,row.get('validation_mse',best),optimizer=optimizer,rng=rng,scheduler=scheduler)
    save(output/'summary.json',dict(status='KAN_CONTINUATION_COMPLETE',recipe=recipe_id,lineage=lineage,
        initial_validation_mse=initial['mse'],best_validation_mse=best,best_epoch=best_epoch,last_epoch=epoch,
        wall_seconds=time.monotonic()-started,history=history,best_checkpoint=str(output/'best.pt'),best_checkpoint_sha256=digest(output/'best.pt'),
        cpu_affinity=sorted(__import__('os').sched_getaffinity(0)),torch_threads=torch.get_num_threads(),
        point_sample_sha256=meta['point_sample_sha256'],test_targets_read=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recipe',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(read(SOURCE/'config.json'),a.recipe,a.output)
