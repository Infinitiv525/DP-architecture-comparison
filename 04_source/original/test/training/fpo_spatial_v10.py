"""Explicit local FD features; zero-input control has identical expanded trunk."""
import torch
from torch import nn
from fpo_stable_v6 import StablePINN as Parent,features as base_features,conditioning as base_conditioning,curl_grid
from fpo_step_v5a import project_known_constraints


def expanded_features(previous,current,mode):
    raw=base_features(previous,current)
    dy,dx=torch.gradient(current,spacing=(1/64,1/64),dim=(0,1),edge_order=2)
    dyy=torch.gradient(dy,spacing=(1/64,),dim=(0,),edge_order=2)[0]
    dxx=torch.gradient(dx,spacing=(1/64,),dim=(1,),edge_order=2)[0]
    extra=torch.cat((.1*dx,.1*dy,.01*(dxx+dyy)),2)
    if mode=='zeros':extra=torch.zeros_like(extra)
    elif mode!='fd':raise ValueError('Unknown spatial mode')
    return torch.cat((raw,extra),2)


class SpatialPINN(Parent):
    def __init__(self,config,mean,std):
        super().__init__(config,mean,std);self.spatial_mode=config['spatial_mode']
        old=self.net[0];self.net[0]=nn.Linear(old.in_features+9,old.out_features)
        with torch.no_grad():
            self.net[0].weight[:,:old.in_features].copy_(old.weight);self.net[0].weight[:,old.in_features:].zero_();self.net[0].bias.copy_(old.bias)

    def load_state_dict(self,state_dict,strict=True,assign=False):
        state=dict(state_dict);w=state['net.0.weight']
        if w.shape[1]==self.net[0].in_features-9:
            state['net.0.weight']=torch.cat((w,torch.zeros((len(w),9),device=w.device,dtype=w.dtype)),1)
        return super().load_state_dict(state,strict=strict,assign=assign)

    def heads(self,q,c,raw,weight):
        s=(q-self.lo)/self.span;phase=2*torch.pi*s[:,:2,None]*self.frequencies
        extra=raw[:,6:] if self.spatial_mode=='fd' else torch.zeros_like(raw[:,6:])
        local=torch.cat(((raw[:,:3]-self.mean)/self.std,raw[:,3:6]/self.std,extra/self.std.repeat(3)),1)
        h=self.net(torch.cat((2*s-1,torch.sin(phase).flatten(1),torch.cos(phase).flatten(1),c.expand(len(q),-1),local),1))
        return torch.cat((h[:,:1]*self.psi_scale*weight,h[:,1:2]*self.std[2]),1)

    def grid_prediction(self,qgrid,c,raw,weight):
        h=self.heads(qgrid,c,raw.reshape(-1,15),weight.reshape(-1,1))
        correction=torch.cat((curl_grid(h[:,0].reshape(raw.shape[:2])),h[:,1].reshape(raw.shape[:2]+(1,))),2)
        return raw[:,:,:3]+qgrid[:,2].reshape(raw.shape[:2]+(1,))/.05*correction

    def sparse_prediction(self,*args,**kwargs):raise NotImplementedError('This pilot uses full-grid objectives only')


def features(previous,current):
    # The trainer uses this function for continuous PDE query fields.
    # Mode is applied by the model, below, consistently with grid inference.
    return expanded_features(previous,current,'fd')


def conditioning(model,raw,geometry,reynolds):return base_conditioning(model,raw[:,:,:6],geometry,reynolds)


def stable_step(model,previous,current,case,qgrid):
    raw=expanded_features(previous,current,model.spatial_mode)
    c=conditioning(model,raw[:,:,:6],case['geometry'],case['record']['reynolds'])
    return project_known_constraints(model.grid_prediction(qgrid,c,raw,case['weight']),case['fluid'])
