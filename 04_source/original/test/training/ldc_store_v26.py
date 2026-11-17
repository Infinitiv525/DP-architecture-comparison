"""Only prepared TRAIN/VAL caches; final-test targets are absent from this API."""
import json
from pathlib import Path
import numpy as np
import torch
from fpo_step_v5a import project_known_constraints
from ldc_shared_pinn_v26 import regular_grid


def interior_mask(fluid):
    mask=fluid.copy()
    for offset in [1,2,3]:
        for axis in [0,1]:mask &= np.roll(fluid,offset,axis)&np.roll(fluid,-offset,axis)
    mask[:3]=False;mask[-3:]=False;mask[:,:3]=False;mask[:,-3:]=False
    return mask


def obstacle_faces(fluid):
    n=fluid.shape[0];parts=[]
    y,x=np.where(fluid[:,:-1]!=fluid[:,1:]);parts.append(np.column_stack(((x+1)/n,(y+.5)/n)))
    y,x=np.where(fluid[:-1,:]!=fluid[1:,:]);parts.append(np.column_stack(((x+.5)/n,(y+1)/n)))
    points=np.concatenate(parts).astype('float32')
    if not len(points):raise ValueError('Obstacle boundary missing')
    return points


class Store:
    def __init__(self,prepared,split,smoke=False):
        self.cache=Path(prepared['cache']);self.split=split
        self.train_ids=split['train'][:8] if smoke else split['train']
        self.validation_ids=split['validation'][:2] if smoke else split['validation']
        self.train_slots={i:j for j,i in enumerate(split['train'])}
        self.validation_slots={i:j for j,i in enumerate(split['validation'])}
        self.allowed=set(self.train_ids)|set(self.validation_ids)
        self.geometry_np=np.load(self.cache/'geometry_low.npy',mmap_mode='r',allow_pickle=False)
        self.fluid_np=np.load(self.cache/'fluid.npy',mmap_mode='r',allow_pickle=False)
        self.y_train=np.load(self.cache/'Y_train.npy',mmap_mode='r',allow_pickle=False)
        self.y_validation=np.load(self.cache/'Y_validation.npy',mmap_mode='r',allow_pickle=False)
        if self.y_train.shape!=(640,3,512,512) or self.y_validation.shape!=(160,3,512,512):raise ValueError('Unexpected native cache shapes')
        self.inputs=json.loads((self.cache/'inputs.json').read_text());self.normalization=json.loads((self.cache/'normalization.json').read_text())
        original_x=np.load(self.cache.parent/'raw/X_original.npy',mmap_mode='r',allow_pickle=False)
        schema=self.inputs['schema']
        self.known_cache={};self.targets={};self.interiors={};self.fluid_cells={};self.body={}
        for i in self.train_ids+self.validation_ids:
            fluid=np.array(self.fluid_np[i],dtype=bool,copy=True)
            sdf=np.asarray(original_x[i,schema['sdf_channel']],dtype=np.float32)*schema['sdf_orientation']
            native=np.stack((fluid.astype('float32'),sdf))[None]
            self.known_cache[i]=dict(geometry_native=torch.tensor(native,device='cuda'),
                fluid=torch.tensor(fluid,device='cuda'),reynolds=float(self.inputs['reynolds'][i]))
            if i in self.train_ids:
                label=np.moveaxis(np.array(self.y_train[self.train_slots[i]],copy=True),0,2)
                self.targets[i]=torch.tensor(label,device='cuda').contiguous()
                self.interiors[i]=np.column_stack(np.where(interior_mask(fluid))).astype('int16')
                self.fluid_cells[i]=np.column_stack(np.where(fluid)).astype('int16')
                self.body[i]=obstacle_faces(fluid)
        self.qgrid=regular_grid()

    def known(self,index):
        if index not in self.allowed:raise ValueError('Sealed or undeclared case')
        return self.known_cache[index]

    def training_target(self,index,y,x):
        if index not in self.train_ids:raise ValueError('OnlyTRAIN targets enter optimization')
        return self.targets[index][y,x]

    def validation_reference(self,index):
        if index not in self.validation_ids:raise ValueError('ValidationCPUscorer only')
        return np.asarray(self.y_validation[self.validation_slots[index]])


def query(model,known,qgrid):
    context=conditioning(model,known)
    return project_known_constraints(model.grid_prediction(qgrid,context),known['fluid'])


def conditioning(model,known):
    # This input-dependent pooling is recomputed inside every timed query.
    geometry=torch.nn.functional.avg_pool2d(known['geometry_native'],8)
    return model.context(geometry,known['reynolds'])
