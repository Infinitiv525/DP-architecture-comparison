"""Frozen final-evaluation utilities. No optimizer and no parameter selection."""
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path('/mnt/project/perun2601449/dp_llm_cfd')
for folder in ['analysis','llm','training']:sys.path.insert(0,str(ROOT/'test'/folder))
from ldc_main_common_v33 import WORK,read,save,digest,fingerprint,verify as verify_main
from ldc_main_scheduler_v33 import locked
from ldc_store_v26 import query,conditioning
from ldc_shared_pinn_v26 import SharedSteadyPINN,regular_grid
from ldc_data_v25 import canonical_target,reference_physics
from fpo_state_v4 import errors
from audit_ldc_quality_v31 import boundary_audit

FINAL=ROOT/'results/ldc_final_test_20261108_v49'


def verify():
    verify_main()
    freeze=read(FINAL/'freeze.json')
    for path,expected in freeze['files'].items():
        if digest(path)!=expected:raise ValueError('Final evaluation source/metadata changed: '+path)
    for path,expected in freeze['input_fingerprints'].items():
        if fingerprint(path)!=expected:raise ValueError('Sealed TEST source changed: '+path)
    if read(FINAL/'training_closeout/summary.json')['status']!='TRAINING_AUDITED_MODEL_MANIFEST_READY':
        raise ValueError('Completed training audit required')
    return read(FINAL/'protocol.json')


def aggregate(rows):
    if not rows:raise ValueError('Empty final evaluation')
    keys=['mse_all','velocity_relative_l2','pressure_relative_l2','relative_l2_all']
    return {k:float(np.mean([r[k] for r in rows])) for k in keys}


def metrics(prediction,reference,fluid,reynolds):
    if prediction.shape!=reference.shape or prediction.shape[-1]!=3:
        raise ValueError('Field shape mismatch')
    if not np.isfinite(prediction).all() or np.abs(prediction).max()>1000:
        raise ValueError('Nonfinite/overflow prediction; no fabricated score or clipping')
    difference=prediction.astype(np.float64)-reference
    result=errors(prediction[fluid],reference[fluid])
    result.update(mse_all=float(np.mean(difference**2)),
        relative_l2_all=float(np.sqrt(np.sum(difference**2)/max(np.sum(reference.astype(np.float64)**2),1e-24))))
    solid=np.column_stack(np.where(~fluid));lo=solid.min(0);hi=solid.max(0)
    margin=np.maximum(np.ceil((hi-lo+1)*.1).astype(int),1)
    lo=np.maximum(lo-margin,0);hi=np.minimum(hi+margin,fluid.shape[0]-1)
    region=np.zeros_like(fluid);region[lo[0]:hi[0]+1,lo[1]:hi[1]+1]=True;region &= fluid
    result['near_obstacle_errors']=errors(prediction[region],reference[region])
    result['physics']=reference_physics(np.moveaxis(prediction,2,0),fluid,reynolds)
    return result


class EvaluationInputs:
    def __init__(self,protocol,role):
        if role not in ['validation','test']:raise ValueError('Explicit evaluation role required')
        self.role=role;prepared=read(protocol['prepared_path']);self.cache=Path(prepared['cache'])
        split=read(protocol['split_path'])['indices']
        self.ids=split['validation'][:1] if role=='validation' else protocol['test_ids']
        if role=='test':
            verify()
            if self.ids!=split['test']:raise ValueError('Exactly the sealed200TEST cases required')
            if read(FINAL/'preflight/review.json')['status']!='FOUR_GPU_VALIDATION_PREFLIGHT_PASS':
                raise ValueError('All GPU preflight checks must pass before TEST access')
        self.fluid=np.load(self.cache/'fluid.npy',mmap_mode='r',allow_pickle=False)
        self.inputs=read(self.cache/'inputs.json');self.normalization=read(self.cache/'normalization.json')
        x=np.load(self.cache.parent/'raw/X_original.npy',mmap_mode='r',allow_pickle=False)
        # The TEST reference file is opened only in the explicitly gated branch.
        if role=='test':raw=np.load(prepared['rawtest_targets_sealed_in'],mmap_mode='r',allow_pickle=False)
        else:raw=np.load(self.cache/'Y_validation.npy',mmap_mode='r',allow_pickle=False)
        self.known={};self.reference={};self.reference_physics={}
        for index in self.ids:
            fluid=np.array(self.fluid[index],copy=True)
            sdf=np.asarray(x[index,self.inputs['schema']['sdf_channel']],dtype=np.float32)*self.inputs['schema']['sdf_orientation']
            native=np.stack((fluid.astype('float32'),sdf))[None]
            self.known[index]=dict(geometry_native=torch.tensor(native,device='cuda'),
                fluid=torch.tensor(fluid,device='cuda'),reynolds=float(self.inputs['reynolds'][index]))
            if role=='test':target=np.moveaxis(canonical_target(np.asarray(raw[index]),fluid),0,2)
            else:target=np.moveaxis(np.array(raw[split['validation'].index(index)],copy=True),0,2)
            self.reference[index]=target
            self.reference_physics[index]=reference_physics(np.moveaxis(target,2,0),fluid,self.known[index]['reynolds'])
        self.qgrid=regular_grid()


def load_model(row,normalization):
    if digest(row['checkpoint'])!=row['checkpoint_sha256']:raise ValueError('Frozen checkpoint changed')
    checkpoint=torch.load(row['checkpoint'],map_location='cpu',weights_only=True)
    if checkpoint['config']!=row['config'] or checkpoint['normalization']!=normalization:
        raise ValueError('Checkpoint config/normalization mismatch')
    model=SharedSteadyPINN(row['config'],normalization).cuda()
    model.load_state_dict(checkpoint['state'],strict=True)
    if sum(p.numel() for p in model.parameters())!=134922:raise ValueError('Wrong architecture')
    model.eval().requires_grad_(False)
    return model


def predict_case(model,store,index,timing):
    known=store.known[index];fluid=np.asarray(store.fluid[index]);reference=store.reference[index]
    times=[]
    with torch.inference_mode():
        if timing:
            for _ in range(30):query(model,known,store.qgrid)
            torch.cuda.synchronize()
            for _ in range(200):
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                a.record();value=query(model,known,store.qgrid);b.record();b.synchronize()
                times.append(a.elapsed_time(b))
        else:value=query(model,known,store.qgrid)
        prediction=value.cpu().numpy()
    row=dict(sample=index,reynolds=known['reynolds'],geometry=store.inputs['geometry_hashes'][index],
             **metrics(prediction,reference,fluid,known['reynolds']))
    row['reference_physics']=store.reference_physics[index]
    if timing:
        row['timing']={f'p{p}':float(np.percentile(times,p)) for p in [50,95,99]}
        row['timing'].update(maximum_ms=float(max(times)),above_5ms=int(np.sum(np.asarray(times)>5)))
        with torch.no_grad():context=conditioning(model,known)
        row['boundary']=boundary_audit(model,context,fluid,prediction,reference,
             dict(boundary=dict(outer_points_per_side=256,corner_exclusion=1/512)))
        row['boundary_mean']=float(np.mean([row['boundary'][k]['velocity_error_rms'] for k in ['bottom','top','left','right','obstacle']]))
    return row,times


def claim(rank,job):
    directory=FINAL/'evaluation'
    with locked(directory):
        state=read(directory/'state.json')
        if state['attention_required']:return None
        for identifier,row in state['tasks'].items():
            if row['status']=='PENDING':
                row.update(status='RUNNING',rank=rank,job=job,claimed_unix=time.time())
                save(directory/'state.json',state)
                return row['model']
    return None


def finish(identifier,rank,summary=None,error=None):
    directory=FINAL/'evaluation'
    with locked(directory):
        state=read(directory/'state.json');row=state['tasks'][identifier]
        if row['status']!='RUNNING' or row['rank']!=rank:raise ValueError('Evaluation ownership mismatch')
        row.update(status='ATTENTION' if error else 'DONE',finished_unix=time.time())
        if error:row['error']=error;state['attention_required']=True
        else:row['summary_sha256']=digest(summary)
        save(directory/'state.json',state)
