"""Time-local PINN with known current/previous states and physical dt=0.05.

Validation rollout accepts only the two initial fields. Every later input is
its own prediction. Reference future arrays are held by the scorer separately.
"""
import numpy as np
import torch
from torch import nn
from fpo_state_v4 import StatePINN,bicubic,residual,errors
from fpo_common_v5a import ROOT,WORK,save,digest,verify
DT=.05


class StepPINN(StatePINN):
    def __init__(self,config,mean,std):
        if not config['state_conditioning']:raise ValueError('Known state is mandatory')
        super().__init__(config,mean,std)
        self.span[2]=DT
        # Spatial Fourier features; local time is linear over a single short
        # interval instead of many time oscillations within one snapshot step.
        last=3+4*len(config['frequencies'])+config['latent']+1+6;layers=[]
        for _ in range(config['depth']):
            layers += [nn.Linear(last,config['width']),nn.Tanh()];last=config['width']
        self.net=nn.Sequential(*layers,nn.Linear(last,3))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,q,context,initial_field=None,cached_features=None):
        raw=bicubic(initial_field,q) if cached_features is None else cached_features
        local=torch.cat(((raw[:,:3]-self.mean)/self.std,raw[:,3:]/self.std),1)
        s=(q-self.lo)/self.span
        phase=2*torch.pi*s[:,:2,None]*self.frequencies
        inputs=torch.cat((2*s-1,torch.sin(phase).flatten(1),torch.cos(phase).flatten(1),
                          context.expand(len(q),-1),local),1)
        correction=self.net(inputs)*self.std
        # Initial zero correction is the two-frame linear extrapolator.
        # The predicted correction is learned from data and NS residuals.
        return raw[:,:3]+(q[:,2:3]/DT)*(raw[:,3:]+correction)


def grid(device='cuda',ny=256,nx=1024):
    y,x=torch.meshgrid(6+(torch.arange(ny,device=device)+.5)*4/ny,
                       4+(torch.arange(nx,device=device)+.5)*16/nx,indexing='ij')
    return torch.stack((x.flatten(),y.flatten(),torch.full_like(x.flatten(),DT)),1)


def features(previous,current):return torch.cat((current,current-previous),2)


def conditioning(model,raw,geometry,reynolds):
    ny,nx=raw.shape[:2]
    low=raw.reshape(ny//8,8,nx//8,8,6).mean((1,3)).permute(2,0,1)[None]
    return model.context(geometry,low,reynolds)


def project_known_constraints(value,fluid):
    # Gauge uses prediction only, not reference future pressure. Zero values
    # inside the known solid mask are geometry constraints, not crop walls.
    mask=fluid.to(value.dtype)
    pressure=value[:,:,2]-(value[:,:,2]*mask).sum()/mask.sum()
    return torch.cat((value[:,:,:2],pressure[:,:,None]),2)*mask[:,:,None]


def step(model,previous,current,geometry,reynolds,fluid,qgrid):
    raw=features(previous,current);c=conditioning(model,raw,geometry,reynolds)
    value=model(qgrid,c,cached_features=raw.reshape(-1,6)).reshape(current.shape)
    return project_known_constraints(value,fluid)


def trajectory(previous,current,step_function,count):
    """No reference reader is accepted by this causal state machine."""
    for _ in range(count):
        predicted=step_function(previous,current)
        previous,current=current,predicted
        yield predicted


def fingerprint(path):
    s=__import__('pathlib').Path(path).stat()
    # st_dev identifies the CLIENT'S mount, not the shared file across hosts.
    # The original full SHA attestation is retained in the prepared receipt.
    return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino)


def load_cases(receipts,smoke=False):
    if smoke:receipts=[next(r for r in receipts if r['proposed_split']==role) for role in ['train','validation']]
    train_count=sum(r['proposed_split']=='train' for r in receipts)
    required=train_count*240*256*1024*3*4+12*1024**3
    if torch.cuda.mem_get_info()[0]<required:raise RuntimeError('Insufficient GPU memory for float32 training trajectories and headroom')
    cases=[]
    for r in receipts:
        if r['proposed_split']=='test':raise ValueError('Sealed field')
        if fingerprint(r['field'])!=r['field_fingerprint']:
            raise ValueError('Field fingerprint mismatch: '+r['field']+' expected='+str(r['field_fingerprint'])+' actual='+str(fingerprint(r['field'])))
        with np.load(str(r['cache'])+'/geometry.npz',allow_pickle=False) as z:
            geometry=z['encoded'].copy();fluid=z['fluid'].copy();interior=z['interior_yx'].copy();boundary=z['boundary'].copy()
        case=dict(record=r,geometry=torch.tensor(geometry[None],device='cuda'),fluid_np=fluid,
                  fluid=torch.tensor(fluid,device='cuda'),interior=interior,boundary=boundary,
                  fluid_yx=np.column_stack(np.where(fluid)))
        if r['proposed_split']=='train':
            a=np.load(r['field'],mmap_mode='r',allow_pickle=False)
            data=torch.empty((240,256,1024,3),dtype=torch.float32,device='cuda')
            for start in range(0,240,8):
                b=torch.tensor(np.asarray(a[start+2:start+10],dtype=np.float32),device='cuda')
                if not torch.isfinite(b).all():raise ValueError('Reference not finite after float32 conversion')
                p=b[:,:,:,2];p-=p[:,case['fluid']].mean(1)[:,None,None]
                b*=case['fluid'][None,:,:,None];data[start:start+len(b)].copy_(b)
            if fingerprint(r['field'])!=r['field_fingerprint']:raise ValueError('Reference changed while loading')
            case['data']=data
        else:
            # The only initial GPU states for validation come from the same
            # previously frozen prefix cache. No future reference is copied.
            if digest(r['state_cache'])!=r['state_cache_sha256']:raise ValueError('Changed prefix')
            with np.load(r['state_cache'],allow_pickle=False) as z:raw=torch.tensor(z['features'],device='cuda')
            case['prefix']=(raw[:,:,:3]-raw[:,:,3:],raw[:,:,:3])
            case['reference']=np.load(r['field'],mmap_mode='r',allow_pickle=False)
        cases.append(case);print('LOADED_TRAJECTORY',r['path'],r['proposed_split'],flush=True)
    return cases
