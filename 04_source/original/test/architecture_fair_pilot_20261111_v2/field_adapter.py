"""Shared native-field interface; runtime evidence is recorded separately.

Pixel unshuffle/shuffle only permutes values: native geometry and native targets
remain complete. The optional global descriptor is identical for all families
and supplements the full geometry; it never replaces it.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class FullFieldAdapter(nn.Module):
    native_size=512
    packing_factor=8
    packed_size=64
    geometry_channels=128
    global_channels=512
    physical_channels=3
    coordinate_channels=22
    inputs=665
    outputs=192

    def __init__(self,normalization,projection_seed=174019):
        super().__init__()
        self.register_buffer('geometry_mean',torch.tensor(normalization['geometry_mean']).float()[None,:,None,None])
        self.register_buffer('geometry_std',torch.tensor(normalization['geometry_std']).float()[None,:,None,None])
        self.register_buffer('target_mean',torch.tensor(normalization['target_mean']).float()[None,:,None,None])
        self.register_buffer('target_std',torch.tensor(normalization['target_std']).float()[None,:,None,None])
        generator=torch.Generator(device='cpu').manual_seed(projection_seed)
        size=2*self.native_size*self.native_size
        signs=(torch.randint(0,2,(1,size),generator=generator,dtype=torch.int8).float()*2-1)
        frequencies=torch.randperm(size//2-1,generator=generator)[:self.global_channels//2]+1
        self.register_buffer('projection_signs',signs)
        self.register_buffer('projection_frequencies',frequencies)
        self.projection_scale=math.sqrt(2/size)
        self.projection_seed=projection_seed
        v=(torch.arange(self.packed_size).float()+.5)/self.packed_size
        yy,xx=torch.meshgrid(v,v,indexing='ij');q=torch.stack((xx,yy))
        phase=2*math.pi*q[:,None]*torch.tensor([1.,2.,4.,8.,16.])[None,:,None,None]
        coordinates=torch.cat((2*q-1,phase.sin().flatten(0,1),phase.cos().flatten(0,1)))[None]
        self.register_buffer('coordinates',coordinates,persistent=False)

    def encode(self,geometry,reynolds):
        if geometry.ndim==3:geometry=geometry[None]
        if geometry.shape[1:]!=(2,512,512):raise ValueError('Every model requires both complete native geometry fields')
        if geometry.dtype!=torch.float32:raise ValueError('Shared FP32 input contract')
        batch=geometry.shape[0];reynolds=reynolds.reshape(batch,1).to(dtype=geometry.dtype)
        normalized=(geometry-self.geometry_mean)/self.geometry_std
        packed=F.pixel_unshuffle(normalized,self.packing_factor)
        spectrum=torch.fft.rfft(normalized.flatten(1)*self.projection_signs,dim=1)
        selected=spectrum[:,self.projection_frequencies]
        global_geometry=torch.cat((selected.real,selected.imag),1)*self.projection_scale
        physical=torch.cat((reynolds/100.,reynolds.log()/8.,100./reynolds),1)
        shape=(batch,-1,self.packed_size,self.packed_size)
        values=torch.cat((packed,global_geometry[:,:,None,None].expand(shape),
                          physical[:,:,None,None].expand(shape),self.coordinates.expand(batch,-1,-1,-1)),1)
        if values.shape!=(batch,self.inputs,64,64):raise ValueError('Shared encoded-input contract differs')
        return values

    def decode_raw(self,normalized_patches):
        if normalized_patches.shape[1:]!=(self.outputs,64,64):raise ValueError('Every core must return complete native-field patches')
        return F.pixel_shuffle(normalized_patches,self.packing_factor)*self.target_std+self.target_mean

    def target_patches(self,target_hwc):
        if target_hwc.ndim==3:target_hwc=target_hwc[None]
        target=target_hwc.permute(0,3,1,2)
        return F.pixel_unshuffle((target-self.target_mean)/self.target_std,self.packing_factor)

    @staticmethod
    def project_known_constraints(field,fluid):
        if fluid.ndim==2:fluid=fluid[None]
        mask=fluid.to(field.dtype);pressure=field[:,2]
        mean=(pressure*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1)
        return torch.cat((field[:,:2],(pressure-mean[:,None,None])[:,None]),1)*mask[:,None]


def case_balanced_field_loss(raw_prediction_hwc,target_hwc,epsilon=1e-12):
    """Proposed common objective; no acceptance threshold or TEST information.

Each TRAIN case contributes squared relative velocity and pressure errors.
All native cells participate; canonical references have zero values in solids.
Constructive fitting must implement the same positive per-case/channel weights.
"""
    if raw_prediction_hwc.ndim==3:raw_prediction_hwc=raw_prediction_hwc[None];target_hwc=target_hwc[None]
    if raw_prediction_hwc.shape!=target_hwc.shape:raise ValueError('Native target/prediction shape mismatch')
    error=raw_prediction_hwc-target_hwc
    velocity=error[...,:2].square().sum((1,2,3))/target_hwc[...,:2].square().sum((1,2,3)).clamp_min(epsilon)
    pressure=error[...,2].square().sum((1,2))/target_hwc[...,2].square().sum((1,2)).clamp_min(epsilon)
    return .5*(velocity+pressure).mean()
