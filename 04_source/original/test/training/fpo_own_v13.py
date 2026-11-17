"""Own conditional residual PINN; no external template code or dependencies."""
import torch
from torch import nn
from fpo_state_v4 import StatePINN,bicubic


class ConditionalBlock(nn.Module):
    def __init__(self,width,context_size):
        super().__init__();self.first=nn.Linear(width,width);self.second=nn.Linear(width,width)
        self.condition=nn.Linear(context_size,2*width)

    def forward(self,x,context):
        scale,shift=self.condition(context).chunk(2,dim=-1)
        hidden=torch.tanh((1+.1*torch.tanh(scale))*self.first(x)+.1*torch.tanh(shift))
        return x+.5*torch.tanh(self.second(hidden))


class OwnPINN(StatePINN):
    def __init__(self,config,mean,std):
        if not config['state_conditioning']:raise ValueError('Initial states mandatory')
        super().__init__(config,mean,std)
        del self.net
        self.entry=nn.Linear(3+6*len(config['frequencies'])+6,config['width'])
        self.blocks=nn.ModuleList([ConditionalBlock(config['width'],config['latent']+1) for _ in range(config['depth'])])
        self.output=nn.Linear(config['width'],3)

    def forward(self,q,context,initial_field=None,cached_features=None):
        raw=bicubic(initial_field,q) if cached_features is None else cached_features
        local=torch.cat(((raw[:,:3]-self.mean)/self.std,raw[:,3:]/self.std),1)
        s=(q-self.lo)/self.span;phase=2*torch.pi*s.unsqueeze(-1)*self.frequencies
        x=torch.tanh(self.entry(torch.cat((2*s-1,torch.sin(phase).flatten(1),torch.cos(phase).flatten(1),local),1)))
        for block in self.blocks:x=block(x,context)
        delta=self.output(x)*self.std
        return raw[:,:3]+(1-torch.exp(-q[:,2:3]))*delta


def make_model(config,mean,std):
    if config['model_kind']=='baseline':return StatePINN(config,mean,std)
    if config['model_kind']=='own':return OwnPINN(config,mean,std)
    raise ValueError('Unknown model kind')
