"""State-conditioned, data-assisted PINN. Prefix frames 2,3; future frames 4..241.

Known initial flow is a problem input, never a later target. Bicubic sampling
is implemented with polynomial weights so physical second derivatives include
the spatial dependence of the initial-state features (no detached grid_sample).
"""
import numpy as np
import torch
from torch import nn
from fpo_common_v4 import ROOT, WORK, save, digest, verify, coords
from fpo_conditioned_v3 import residual, errors


def bicubic(field, q):
    # Cell centres of the original cropped grid. Clamp extends only the input
    # interpolation stencil; it does not impose a fluid wall at the crop edges.
    xy=(q[:,:2]-q.new_tensor([4.,6.]))*64-.5
    index=torch.floor(xy).detach().long();f=xy-index.to(xy.dtype)
    def weights(z):
        return torch.stack((-.5*z+z*z-.5*z**3,1-2.5*z*z+1.5*z**3,
                            .5*z+2*z*z-1.5*z**3,-.5*z*z+.5*z**3),1)
    wx,wy=weights(f[:,0]),weights(f[:,1])
    offsets=torch.arange(-1,3,device=q.device)
    ix=(index[:,0,None]+offsets).clamp(0,field.shape[1]-1)
    iy=(index[:,1,None]+offsets).clamp(0,field.shape[0]-1)
    values=field[iy[:,:,None],ix[:,None,:]]
    return (values*(wy[:,:,None]*wx[:,None,:])[:,:,:,None]).sum((1,2))


class StatePINN(nn.Module):
    def __init__(self, config, mean, std):
        super().__init__();self.state_conditioning=config['state_conditioning']
        self.register_buffer('lo',torch.tensor([4.,6.,0.]))
        self.register_buffer('span',torch.tensor([16.,4.,11.9]))
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32))
        self.register_buffer('frequencies',torch.tensor(config['frequencies'],dtype=torch.float32))
        latent=config['latent']
        self.encoder=nn.Sequential(nn.Conv2d(8,8,3,padding=1),nn.Tanh(),nn.AvgPool2d(2),
                                   nn.Conv2d(8,16,3,padding=1),nn.Tanh(),nn.AvgPool2d(2),
                                   nn.AdaptiveAvgPool2d((4,16)),nn.Flatten(),
                                   nn.Linear(1024,latent),nn.Tanh())
        last=3+6*len(config['frequencies'])+latent+1+6;layers=[]
        for _ in range(config['depth']):
            layers += [nn.Linear(last,config['width']),nn.Tanh()];last=config['width']
        self.net=nn.Sequential(*layers,nn.Linear(last,3))

    def context(self, geometry, state_low, reynolds):
        if self.state_conditioning:
            obs=(state_low[:,:3]-self.mean[None,:,None,None])/self.std[None,:,None,None]
            change=state_low[:,3:]/self.std[None,:,None,None]
            state=torch.cat((obs,change),1)
        else:
            state=torch.zeros_like(state_low)
        z=self.encoder(torch.cat((geometry,state),1))
        re=z.new_tensor(float(reynolds)).log().reshape(1,1)/8
        return torch.cat((z,re),1)

    def forward(self, q, context, initial_field=None, cached_features=None):
        if self.state_conditioning:
            raw=bicubic(initial_field,q) if cached_features is None else cached_features
            local=torch.cat(((raw[:,:3]-self.mean)/self.std,raw[:,3:]/self.std),1)
        else:
            raw=None;local=q.new_zeros((len(q),6))
        s=(q-self.lo)/self.span
        phase=2*torch.pi*s.unsqueeze(-1)*self.frequencies
        inputs=torch.cat((2*s-1,torch.sin(phase).flatten(1),torch.cos(phase).flatten(1),
                          context.expand(len(q),-1),local),1)
        delta=self.net(inputs)*self.std
        if self.state_conditioning:
            # Enforce the last observed prefix state at relative t=0. This is
            # a known problem condition, not future ground-truth feedback.
            return raw[:,:3]+(1-torch.exp(-q[:,2:3]))*delta
        return delta+self.mean


def load(record):
    with np.load(record['state_cache'],allow_pickle=False) as z:
        initial=z['features'].copy();low=z['low'].copy()
    with np.load(str(record['cache'])+'/geometry.npz',allow_pickle=False) as z:
        geometry=z['encoded'].copy();fluid=z['fluid'].copy()
        boundary=z['boundary'].copy();interior=z['interior_yx'].copy()
    with np.load(str(record['cache'])+'/samples.npz',allow_pickle=False) as z:
        pools={}
        for prefix in ['all','validation']:
            frames=z[prefix+'_frame'];keep=frames>=4
            q=z[prefix+'_q'][keep].copy();q[:,2]-=.05
            pools[prefix]=(q,z[prefix+'_y'][keep].copy(),frames[keep].copy())
    return dict(record=record,geometry=geometry,fluid=fluid,boundary=boundary,interior=interior,
                initial=initial,low=low,pools=pools)
