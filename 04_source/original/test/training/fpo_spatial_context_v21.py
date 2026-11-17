"""Own coordinate PINN with smooth spatial features from the known-input CNN.

The new feature interpolant is C4; the old raw-field bicubic path is unchanged.
No claim that this removes every continuous/discrete physics discrepancy.
"""
import math
import torch
from torch import nn
from fpo_direct_curl_v15 import DirectPINN
from fpo_stable_v6 import curl_grid


def spline_numerators():
    # B5(z)=sum_k (-1)^k*C(6,k)*(z+3-k)_+^5 / 120.
    # For fraction f in[0,1), expand exactly into small integer coefficients.
    return [[sum((-1)**k*math.comb(6,k)*math.comb(5,n)*(3-offset-k)**(5-n)
                 for k in range(4-offset)) for n in range(6)] for offset in range(-2,4)]


def axis_weights(coordinate,lo,span,size,numerators):
    s=(coordinate-lo)*(size/span)-.5;index=torch.floor(s).detach();f=s-index
    c=numerators.to(dtype=coordinate.dtype)
    w=c[:,5][None].expand(len(f),-1)
    for n in range(4,-1,-1):w=w*f[:,None]+c[:,n][None]
    ids=(index.long()[:,None]+torch.arange(-2,4,device=coordinate.device)).clamp(0,size-1)
    return ids,w/120


def spline_points(feature,q,numerators):
    channels,ny,nx=feature.shape
    ix,wx=axis_weights(q[:,0],4.,16.,nx,numerators)
    iy,wy=axis_weights(q[:,1],6.,4.,ny,numerators)
    value=feature.permute(1,2,0)[iy[:,:,None],ix[:,None,:]]
    return (value*(wy[:,:,None]*wx[:,None,:])[:,:,:,None]).sum((1,2))


def axis_matrix(count,source_size,lo,span,numerators,reference):
    coordinate=lo+(torch.arange(count,device=reference.device,dtype=reference.dtype)+.5)*(span/count)
    ids,w=axis_weights(coordinate,lo,span,source_size,numerators)
    return reference.new_zeros((count,source_size)).scatter_add(1,ids,w)


class SpatialPINN(DirectPINN):
    def __init__(self,config,mean,std):
        super().__init__(config,mean,std)
        self.local_context=bool(config['local_context']);self.spatial_channels=16
        old=self.net[0];self.original_input_features=old.in_features
        new=nn.Linear(old.in_features+16,old.out_features)
        with torch.no_grad():
            new.weight[:,:old.in_features].copy_(old.weight);new.weight[:,old.in_features:].zero_();new.bias.copy_(old.bias)
        self.net[0]=new
        self.register_buffer('spline_numerators',torch.tensor(spline_numerators(),dtype=torch.int64),persistent=False)
        # Cache only coordinate-stencil matrices, never input-dependent features.
        self._matrix_cache={}

    def load_source(self,state):
        if state['net.0.weight'].shape[1]!=self.original_input_features:raise ValueError('Source architecture mismatch')
        adapted=dict(state);w=state['net.0.weight']
        adapted['net.0.weight']=torch.cat((w,w.new_zeros((w.shape[0],16))),1)
        self.load_state_dict(adapted,strict=True)

    def context(self,geometry,state_low,reynolds):
        obs=(state_low[:,:3]-self.mean[None,:,None,None])/self.std[None,:,None,None]
        change=state_low[:,3:]/self.std[None,:,None,None]
        value=torch.cat((geometry,obs,change),1)
        for i in range(6):value=self.encoder[i](value)
        spatial=value[0];z=value
        for i in range(6,len(self.encoder)):z=self.encoder[i](z)
        re=z.new_tensor(float(reynolds)).log().reshape(1,1)/8
        return dict(global_context=torch.cat((z,re),1),spatial=spatial)

    def grid_features(self,feature,ny,nx):
        sy,sx=feature.shape[1:];key=(ny,nx,sy,sx,str(feature.device),feature.dtype)
        if key not in self._matrix_cache:
            self._matrix_cache[key]=(
                axis_matrix(ny,sy,6.,4.,self.spline_numerators,feature),
                axis_matrix(nx,sx,4.,16.,self.spline_numerators,feature))
        wy,wx=self._matrix_cache[key]
        # Both contractions are recomputed for every query, including benchmarks.
        return ((wy[None]@feature)@wx.T).permute(1,2,0)

    def encoded_heads(self,q,context,raw,spatial):
        s=(q-self.lo)/self.span;phase=2*torch.pi*s.unsqueeze(-1)*self.frequencies
        local=torch.cat(((raw[:,:3]-self.mean)/self.std,raw[:,3:]/self.std),1)
        h=self.net(torch.cat((2*s-1,torch.sin(phase).flatten(1),torch.cos(phase).flatten(1),
            context['global_context'].expand(len(q),-1),local,spatial),1))
        if self.kind=='curl':return torch.cat((h[:,:1]*self.psi_scale,h[:,1:]*self.std[2]),1)
        return h*self.std

    def heads(self,q,context,raw):
        spatial=spline_points(context['spatial'],q,self.spline_numerators) if self.local_context else q.new_zeros((len(q),16))
        return self.encoded_heads(q,context,raw,spatial)

    def grid_prediction(self,q,context,raw):
        ny,nx=raw.shape[:2]
        spatial=self.grid_features(context['spatial'],ny,nx).reshape(-1,16) if self.local_context else q.new_zeros((len(q),16))
        h=self.encoded_heads(q,context,raw.reshape(-1,6),spatial)
        if self.kind=='curl':correction=torch.cat((curl_grid(h[:,0].reshape(ny,nx)),h[:,1].reshape(ny,nx,1)),2)
        else:correction=h.reshape(ny,nx,3)
        return raw[:,:,:3]+(1-torch.exp(-q[:,2])).reshape(ny,nx,1)*correction
