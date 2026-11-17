"""One full TRAIN simulation: paired CPU/GPU time and native-loss checks."""
import argparse
import gc
import platform
import time
from pathlib import Path
import numpy as np
import torch
from common import (HERE, case_on_device, load_data, patch_loss, patch_prediction,
                    recipe, rows, save, setup, sync, timer)
from field_adapter import case_balanced_field_loss
from operator_models import PackedFieldOperator, restore_model
from weighted_fit import (WeightedNormalEquations, append_scn, cascade_candidate,
                          fit_output, scn_candidate)

NAMES=['tfno','chebkan','pinet','dendrite','realnpu','rbf','elm','scn','cascor']


def verify_interface(model, case, target, x,y,w):
    a=model.adapter
    with torch.no_grad():
        packed=a.target_patches(target)
        recovered=a.decode_raw(packed)[0].permute(1,2,0)
        assert torch.allclose(recovered,target,atol=2e-7,rtol=2e-6)
        encoded=a.encode(case['geometry'],case['re'])
        native=torch.nn.functional.pixel_shuffle(encoded[:,:128],8)
        native=native*a.geometry_std+a.geometry_mean
        assert torch.allclose(native[0],case['geometry'],atol=2e-7,rtol=2e-6)
        generator=torch.Generator(device=x.device).manual_seed(294101)
        prediction=y+.01*torch.randn(y.shape,device=y.device,generator=generator)
        field=a.decode_raw(prediction.reshape(1,64,64,192).permute(0,3,1,2))[0].permute(1,2,0)
        native_loss=case_balanced_field_loss(field,target)
        packed_loss=patch_loss(prediction,y,w)
        assert torch.allclose(native_loss,packed_loss,rtol=5e-5,atol=1e-10), (native_loss,packed_loss)
    # This checks training gradients as well as scalar losses for the adapters.
    pred=prediction.detach().clone().requires_grad_(True)
    field=a.decode_raw(pred.reshape(1,64,64,192).permute(0,3,1,2))[0].permute(1,2,0)
    g1=torch.autograd.grad(case_balanced_field_loss(field,target),pred)[0]
    g2=torch.autograd.grad(patch_loss(pred,y,w),pred)[0]
    assert torch.allclose(g1,g2,rtol=1e-4,atol=1e-10)
    return dict(loss_native=float(native_loss),loss_packed=float(packed_loss),
                gradient_max_difference=float((g1-g2).abs().max()),
                geometry_roundtrip_max_error=float((native[0]-case['geometry']).abs().max()),
                target_roundtrip_max_error=float((recovered-target).abs().max()))


def weighted_solver_check(device):
    """Compare streamed equations with a small direct weighted reference."""
    generator=torch.Generator(device=device).manual_seed(99181)
    x=torch.randn(91,7,generator=generator,device=device); x[:,0]=1
    y=torch.randn(91,192,generator=generator,device=device)
    weights=[torch.tensor([.2,.7,1.3],device=device),torch.tensor([.5,.3,.8],device=device)]
    equations=WeightedNormalEquations(7,device=device)
    equations.add(x[:43],y[:43],weights[0]); equations.add(x[43:],y[43:],weights[1])
    beta,residuals=equations.solve(1e-8)
    expected=[]
    for c in range(3):
        scale=torch.cat((weights[0][c].expand(43),weights[1][c].expand(48))).double().sqrt()
        a=x.double()*scale[:,None]; b=y[:,c*64:(c+1)*64].double()*scale[:,None]
        penalty=torch.eye(7,device=device,dtype=torch.float64)*2e-8; penalty[0,0]=0
        expected.append(torch.linalg.solve(a.T@a+penalty,a.T@b))
    reference=torch.cat(expected,1).float()
    assert torch.allclose(reference,beta,atol=2e-6,rtol=2e-5)
    return dict(max_beta_difference=float((beta-reference).abs().max()),normal_equation_residual=residuals)


def benchmark_model(name,data,device,folder):
    setup(device,931101)
    case=case_on_device(data.cases[data.train_ids[0]],device)
    target=data.targets[data.train_ids[0]].to(device)
    model=PackedFieldOperator(name,recipe(name),data.normalization).to(device)
    x,y,w=rows(model,case,target)
    assert x.shape==(4096,665) and y.shape==(4096,192)
    result=dict(model=name,device=device,case_id=data.train_ids[0],native_target_values=786432,
                encoded_rows=4096,input_channels=665,output_channels=192,
                scope='One complete TRAIN simulation. No TEST targets. Timing estimates are not guarantees.',
                adapter_checks=verify_interface(model,case,target,x,y,w))
    if device=='cuda': torch.cuda.reset_peak_memory_stats()
    begin=time.perf_counter()
    if name not in ['elm','scn','cascor']:
        optimizer=torch.optim.Adam(model.core.parameters(),lr=1e-3)
        def step():
            optimizer.zero_grad(set_to_none=True)
            value=patch_prediction(model.core,name,x)
            loss=patch_loss(value,y,w)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite native gradient objective')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.core.parameters(),1.)
            optimizer.step()
            if hasattr(model.core,'project_parameters'): model.core.project_parameters()
            return float(loss.detach())
        history=[step() for _ in range(3)]
        result['training_step']=timer(lambda:history.append(step()),device,repeats=5)
        result['one_case_learning_losses']=history
        result['training_updates']=len(history)
    elif name=='elm':
        iterator=lambda:iter([(x,y,w)])
        with torch.no_grad():
            result['basis_forward']=timer(lambda:model.core.basis(x),device)
            h=model.core.basis(x)
            result['gram_one_case']=timer(lambda:WeightedNormalEquations(h.shape[1],device=device).add(h,y,w),device)
        sync(device); started=time.perf_counter()
        result['fit']=fit_output(model.core,iterator)
        sync(device); result['fit_total_seconds']=time.perf_counter()-started
    else:
        iterator=lambda:iter([(x,y,w)])
        result['initial_output_fit']=fit_output(model.core,iterator)
        sync(device); started=time.perf_counter()
        if name=='scn':
            node,meta=scn_candidate(model.core,iterator,pool=16,seed=82101)
            result['first_candidate']=meta
            if node is not None: append_scn(model.core,node)
        else:
            node,meta=cascade_candidate(model.core,iterator,pool=8,epochs=1)
            result['first_candidate']=meta; model.core.add(node)
        sync(device); result['first_candidate_seconds']=time.perf_counter()-started
        result['one_node_output_refit']=fit_output(model.core,iterator)
        # A separate UNTRAINED full-capacity object bounds late-growth costs.
        # Its random units are for hardware timing only; never saved as trained.
        late=PackedFieldOperator(name,recipe(name),data.normalization,capacity_construction=True).to(device)
        if name=='scn':
            with torch.no_grad():
                late.core.weight.normal_(0,1/665**.5); late.core.bias.uniform_(-1,1)
        with torch.no_grad():
            result['late_growth_basis']=timer(lambda:late.core.basis(x),device,repeats=2)
            h=late.core.basis(x)
            result['late_growth_gram_case']=timer(lambda:WeightedNormalEquations(h.shape[1],device=device).add(h,y,w),device,repeats=2)
            equations=WeightedNormalEquations(h.shape[1],device=device); equations.add(h,y,w)
            result['late_growth_solve']=timer(lambda:equations.solve(1e-8),device,repeats=2)
        del late,h,equations
    sync(device); result['measured_training_work_seconds']=time.perf_counter()-begin
    model.eval()
    with torch.no_grad():
        prediction=patch_prediction(model.core,name,x)
        result['one_case_raw_loss_after']=float(patch_loss(prediction,y,w))
        full=model(case)
        assert full.shape==(512,512,3) and torch.isfinite(full).all()
        result['inference_full_path']=timer(lambda:model(case),device,repeats=10)
        state=model.state_dict()
        checkpoint=dict(model=name,recipe=recipe(name),normalization=data.normalization,state=state,
                        dynamic=dict(scn_units=len(model.core.weight)) if name=='scn' else
                                dict(cascor_units=len(model.core.units)) if name=='cascor' else {})
        restored=restore_model(checkpoint).to(device)
        restored.eval(); full2=restored(case)
        assert torch.allclose(full,full2,atol=1e-6,rtol=1e-5)
        result['checkpoint_restore_max_difference']=float((full-full2).abs().max())
    result['parameter_ledger_current']=model.parameter_ledger()
    result['peak_GPU_allocated_bytes']=torch.cuda.max_memory_allocated() if device=='cuda' else 0
    result['status']='ONE_CASE_TIMING_AND_TECHNICAL_CHECKS_COMPLETE'
    save(folder/(name+'.json'),result)
    print(name,device,result['status'],flush=True)
    del model,x,y,case,target
    gc.collect()
    if device=='cuda': torch.cuda.empty_cache()
    return result


def main(device,root):
    folder=Path(root)/('benchmark_'+device); folder.mkdir(parents=True,exist_ok=True)
    setup(device,931101); data=load_data()
    save(folder/'data_plan.json',data.plan)
    save(folder/'environment.json',dict(device=device,torch=torch.__version__,numpy=np.__version__,
        host=platform.node(),CPU_threads=torch.get_num_threads(),
        GPU=torch.cuda.get_device_name() if device=='cuda' else None,
        weighted_solver_check=weighted_solver_check(device)))
    results=[]
    for name in NAMES:
        try: results.append(benchmark_model(name,data,device,folder))
        except Exception as error:
            import traceback
            save(folder/(name+'_failure.json'),dict(model=name,error=str(error),traceback=traceback.format_exc()))
            print(traceback.format_exc(),flush=True)
            raise
    save(folder/'complete.json',dict(status='ALL9_ONE_CASE_BENCHMARK_COMPLETE',device=device,
        models=[x['model'] for x in results],test_targets_read=False,
        note='Technical checks and cost timing only; this is not full-cohort accuracy evidence.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--device',choices=['cpu','cuda'],required=True)
    parser.add_argument('--root',required=True); args=parser.parse_args(); main(args.device,args.root)
