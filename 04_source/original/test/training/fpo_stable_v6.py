"""Matched stabilization pilot: inertia, own-state physics, solenoidal increments.

The curl arm preserves the discrete divergence of the velocity increment in
fluid interior stencils. Solid masking can disturb this at the obstacle edge;
neither that edge nor the external crop is claimed to have an exact global BC.
"""
import numpy as np
import torch
from torch import nn
from fpo_state_v4 import StatePINN,bicubic,residual,errors
from fpo_step_v5a import DT,grid,features,conditioning,trajectory,load_cases,fingerprint,project_known_constraints


def derivatives(field):
    ny,nx=field.shape[:2]
    return torch.gradient(field,spacing=(4/ny,16/nx),dim=(0,1),edge_order=2)


def curl_grid(psi):
    dy,dx=derivatives(psi)
    return torch.stack((dy,-dx),-1)


def divergence_grid(velocity):
    _,ux=derivatives(velocity[:,:,0]);vy,_=derivatives(velocity[:,:,1])
    return ux+vy


class StablePINN(StatePINN):
    def __init__(self,config,mean,std):
        if not config['state_conditioning']:raise ValueError('Known prefix is mandatory')
        super().__init__(config,mean,std);self.kind=config['state_update'];self.span[2]=DT
        if self.kind not in ['ar2','first_order','curl']:raise ValueError('Unknown state update')
        last=3+4*len(config['frequencies'])+config['latent']+1+6;layers=[]
        for _ in range(config['depth']):
            layers += [nn.Linear(last,config['width']),nn.Tanh()];last=config['width']
        self.net=nn.Sequential(*layers,nn.Linear(last,2 if self.kind=='curl' else 3))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('psi_scale',self.std[:2].mean().clone()/(2*torch.pi))

    def heads(self,q,c,raw,weight):
        s=(q-self.lo)/self.span;phase=2*torch.pi*s[:,:2,None]*self.frequencies
        local=torch.cat(((raw[:,:3]-self.mean)/self.std,raw[:,3:]/self.std),1)
        inputs=torch.cat((2*s-1,torch.sin(phase).flatten(1),torch.cos(phase).flatten(1),c.expand(len(q),-1),local),1)
        h=self.net(inputs)
        if self.kind=='curl':return torch.cat((h[:,:1]*self.psi_scale*weight,h[:,1:2]*self.std[2]),1)
        return h*self.std

    def forward(self,q,c,initial_field,weight_field):
        raw=bicubic(initial_field,q)
        w=bicubic(weight_field[:,:,None],q) if self.kind=='curl' else None
        h=self.heads(q,c,raw,w)
        if self.kind=='curl':
            if not q.requires_grad:raise ValueError('Continuous curl query needs physical coordinate gradients')
            g=torch.autograd.grad(h[:,0],q,torch.ones_like(h[:,0]),create_graph=True,retain_graph=True)[0]
            correction=torch.stack((g[:,1],-g[:,0],h[:,1]),1)
        else:correction=h
        inertia=raw[:,3:] if self.kind=='ar2' else 0.
        return raw[:,:3]+q[:,2:3]/DT*(inertia+correction)

    def grid_prediction(self,qgrid,c,raw,weight):
        h=self.heads(qgrid,c,raw.reshape(-1,6),weight.reshape(-1,1) if self.kind=='curl' else None)
        if self.kind=='curl':
            psi=h[:,0].reshape(raw.shape[:2]);vel=curl_grid(psi)
            correction=torch.cat((vel,h[:,1].reshape(raw.shape[:2]+(1,))),2)
        else:correction=h.reshape(raw.shape[:2]+(3,))
        inertia=raw[:,:,3:] if self.kind=='ar2' else 0.
        return raw[:,:,:3]+qgrid[:,2].reshape(raw.shape[:2]+(1,))/DT*(inertia+correction)

    def sparse_prediction(self,q,c,raw,weight,y,x):
        """Match the finite-difference grid velocity used at inference exactly."""
        centre=raw[y,x];h=self.heads(q,c,centre,weight[y,x,None] if self.kind=='curl' else None)
        if self.kind!='curl':
            return centre[:,:3]+q[:,2:3]/DT*((centre[:,3:] if self.kind=='ar2' else 0.)+h)
        ny,nx=raw.shape[:2]
        def axis(index,size):
            offsets=torch.tensor([-1,0,1],device=q.device).expand(len(q),-1).clone()
            coefficients=q.new_tensor([-1.,0.,1.]).expand(len(q),-1).clone()
            first=index==0;last=index==size-1
            offsets[first]=torch.tensor([0,1,2],device=q.device)
            coefficients[first]=q.new_tensor([-3.,4.,-1.])
            offsets[last]=torch.tensor([-2,-1,0],device=q.device)
            coefficients[last]=q.new_tensor([1.,-4.,3.])
            return index[:,None]+offsets,coefficients
        ys,cy=axis(y,ny);xs,cx=axis(x,nx)
        qy=q[:,None,:].expand(-1,3,-1).clone();qy[:,:,1]=6+(ys+.5)*4/ny
        qx=q[:,None,:].expand(-1,3,-1).clone();qx[:,:,0]=4+(xs+.5)*16/nx
        hy=self.heads(qy.reshape(-1,3),c,raw[ys,x[:,None]].reshape(-1,6),weight[ys,x[:,None]].reshape(-1,1))[:,0].reshape(-1,3)
        hx=self.heads(qx.reshape(-1,3),c,raw[y[:,None],xs].reshape(-1,6),weight[y[:,None],xs].reshape(-1,1))[:,0].reshape(-1,3)
        correction=torch.stack(((hy*cy).sum(1)/(2*4/ny),-(hx*cx).sum(1)/(2*16/nx),h[:,1]),1)
        return centre[:,:3]+q[:,2:3]/DT*correction


def stable_step(model,previous,current,case,qgrid):
    raw=features(previous,current);c=conditioning(model,raw,case['geometry'],case['record']['reynolds'])
    pred=model.grid_prediction(qgrid,c,raw,case['weight'])
    return project_known_constraints(pred,case['fluid'])


def grid_physics(current,predicted,case,velocity_scale):
    """Midpoint FD momentum + next-state divergence on fluid interior only.

Roll indexing is used to form stencils, but ALL crop-edge and obstacle-crossing
stencils are excluded. This does not impose periodic physical crop boundaries.
"""
    h=1/64;mid=(current+predicted)/2
    def dx(a):return (torch.roll(a,-1,1)-torch.roll(a,1,1))/(2*h)
    def dy(a):return (torch.roll(a,-1,0)-torch.roll(a,1,0))/(2*h)
    def lap(a):return (torch.roll(a,-1,1)+torch.roll(a,1,1)+torch.roll(a,-1,0)+torch.roll(a,1,0)-4*a)/(h*h)
    u,v,p=mid.unbind(2);re=case['record']['reynolds']
    ru=(predicted[:,:,0]-current[:,:,0])/DT+u*dx(u)+v*dy(u)+dx(p)-lap(u)/re
    rv=(predicted[:,:,1]-current[:,:,1])/DT+u*dx(v)+v*dy(v)+dy(p)-lap(v)/re
    d=dx(predicted[:,:,0])+dy(predicted[:,:,1])
    mask=case['interior_gpu'];scale=velocity_scale.clamp_min(1e-4)
    momentum=(ru[mask].square()+rv[mask].square()).mean()/scale.square()
    continuity=d[mask].square().mean()/scale.square()
    return momentum,continuity
