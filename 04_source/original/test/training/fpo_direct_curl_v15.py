"""Direct absolute-time PINN: primitive versus divergence-free increments.

Grid curl preserves initial FD divergence only on fluid interior after masking.
There is no claim of exact whole-boundary/momentum physics or zero initial div.
"""
import torch
from torch import nn
from fpo_state_v4 import StatePINN,bicubic
from fpo_stable_v6 import curl_grid


class DirectPINN(StatePINN):
    def __init__(self,config,mean,std):
        if not config['state_conditioning']:raise ValueError('Known prefix mandatory')
        super().__init__(config,mean,std);self.kind=config['representation']
        if self.kind not in ['primitive','curl']:raise ValueError('Unknown representation')
        if self.kind=='curl':self.net[-1]=nn.Linear(config['width'],2)
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('psi_scale',self.std[:2].mean().clone()/(2*torch.pi))

    def heads(self,q,context,raw):
        s=(q-self.lo)/self.span;phase=2*torch.pi*s.unsqueeze(-1)*self.frequencies
        local=torch.cat(((raw[:,:3]-self.mean)/self.std,raw[:,3:]/self.std),1)
        h=self.net(torch.cat((2*s-1,torch.sin(phase).flatten(1),torch.cos(phase).flatten(1),context.expand(len(q),-1),local),1))
        if self.kind=='curl':return torch.cat((h[:,:1]*self.psi_scale,h[:,1:]*self.std[2]),1)
        return h*self.std

    def forward(self,q,context,initial_field=None,cached_features=None):
        raw=bicubic(initial_field,q) if cached_features is None else cached_features
        h=self.heads(q,context,raw)
        if self.kind=='curl':
            if not q.requires_grad:raise ValueError('Continuous curl needs differentiable physical query')
            g=torch.autograd.grad(h[:,0].sum(),q,create_graph=True,retain_graph=True)[0]
            correction=torch.stack((g[:,1],-g[:,0],h[:,1]),1)
        else:correction=h
        return raw[:,:3]+(1-torch.exp(-q[:,2:3]))*correction

    def grid_prediction(self,q,context,raw):
        # q must be a full physical grid at ONE time, same contract as query().
        h=self.heads(q,context,raw.reshape(-1,6))
        if self.kind=='curl':correction=torch.cat((curl_grid(h[:,0].reshape(raw.shape[:2])),h[:,1].reshape(raw.shape[:2]+(1,))),2)
        else:correction=h.reshape(raw.shape[:2]+(3,))
        return raw[:,:,:3]+(1-torch.exp(-q[:,2])).reshape(raw.shape[:2]+(1,))*correction

    def sampled_prediction(self,q,context,raw,y,x):
        centre=raw[y,x];h=self.heads(q,context,centre)
        if self.kind=='primitive':return centre[:,:3]+(1-torch.exp(-q[:,2:3]))*h
        ny,nx=raw.shape[:2]
        def axis(index,size):
            offsets=torch.tensor([-1,0,1],device=q.device).expand(len(q),-1).clone()
            coeff=q.new_tensor([-1.,0.,1.]).expand(len(q),-1).clone()
            first=index==0;last=index==size-1
            offsets[first]=torch.tensor([0,1,2],device=q.device);coeff[first]=q.new_tensor([-3.,4.,-1.])
            offsets[last]=torch.tensor([-2,-1,0],device=q.device);coeff[last]=q.new_tensor([1.,-4.,3.])
            return index[:,None]+offsets,coeff
        ys,cy=axis(y,ny);xs,cx=axis(x,nx)
        qy=q[:,None,:].expand(-1,3,-1).clone();qy[:,:,1]=6+(ys+.5)*4/ny
        qx=q[:,None,:].expand(-1,3,-1).clone();qx[:,:,0]=4+(xs+.5)*16/nx
        hy=self.heads(qy.reshape(-1,3),context,raw[ys,x[:,None]].reshape(-1,6))[:,0].reshape(-1,3)
        hx=self.heads(qx.reshape(-1,3),context,raw[y[:,None],xs].reshape(-1,6))[:,0].reshape(-1,3)
        correction=torch.stack(((hy*cy).sum(1)/(2*4/ny),-(hx*cx).sum(1)/(2*16/nx),h[:,1]),1)
        return centre[:,:3]+(1-torch.exp(-q[:,2:3]))*correction
