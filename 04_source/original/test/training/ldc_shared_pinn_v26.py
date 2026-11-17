"""Own shared steady2D conditional PINN prototype; no time/prefix/caseID.

Model coordinates[0,1]^2 require reference-scale audit before training.
Uses the generic C4 spline weights already developed in V21, no FNO.
"""
import torch
from torch import nn
from fpo_spatial_context_v21 import spline_numerators,axis_weights,axis_matrix


def regular_grid(ny=512,nx=512,device='cuda',dtype=torch.float32):
    y,x=torch.meshgrid((torch.arange(ny,device=device,dtype=dtype)+.5)/ny,
        (torch.arange(nx,device=device,dtype=dtype)+.5)/nx,indexing='ij')
    return torch.stack((x.flatten(),y.flatten()),1)


def spatial_points(feature,q,numerators):
    channels,ny,nx=feature.shape
    ix,wx=axis_weights(q[:,0],0.,1.,nx,numerators)
    iy,wy=axis_weights(q[:,1],0.,1.,ny,numerators)
    sampled=feature.permute(1,2,0)[iy[:,:,None],ix[:,None,:]]
    return (sampled*(wy[:,:,None]*wx[:,None,:])[:,:,:,None]).sum((1,2))


class SharedSteadyPINN(nn.Module):
    def __init__(self,config,normalization):
        super().__init__()
        self.register_buffer('std',torch.tensor(normalization['target_std'],dtype=torch.float32))
        self.register_buffer('geometry_mean',torch.tensor(normalization['geometry_mean'],dtype=torch.float32))
        self.register_buffer('geometry_std',torch.tensor(normalization['geometry_std'],dtype=torch.float32))
        self.register_buffer('frequencies',torch.tensor(config['frequencies'],dtype=torch.float32))
        self.register_buffer('numerators',torch.tensor(spline_numerators(),dtype=torch.int64),persistent=False)
        self.register_buffer('psi_scale',self.std[:2].mean().clone()/(2*torch.pi))
        self.encoder=nn.Sequential(nn.Conv2d(2,8,3,padding=1),nn.Tanh(),nn.AvgPool2d(2),
            nn.Conv2d(8,16,3,padding=1),nn.Tanh(),nn.AvgPool2d(2))
        self.global_head=nn.Sequential(nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(256,config['latent']),nn.Tanh())
        inputs=2+4*len(config['frequencies'])+config['latent']+1+16
        layers=[]
        for _ in range(config['depth']):layers += [nn.Linear(inputs,config['width']),nn.Tanh()];inputs=config['width']
        layers += [nn.Linear(inputs,2)];self.net=nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        self._coordinate_matrices={}

    def context(self,geometry,reynolds):
        if geometry.ndim!=4 or geometry.shape[:2]!=(1,2):raise ValueError('One geometry, two inputchannels required')
        geometry=(geometry-self.geometry_mean[None,:,None,None])/self.geometry_std[None,:,None,None]
        spatial=self.encoder(geometry);global_feature=self.global_head(spatial)
        re=global_feature.new_tensor(float(reynolds)).log().reshape(1,1)/8
        return dict(spatial=spatial[0],global_feature=torch.cat((global_feature,re),1))

    def heads(self,q,context,spatial=None):
        if q.shape[1]!=2:raise ValueError('Steady model takes x,y, no timecoordinate')
        if spatial is None:spatial=spatial_points(context['spatial'],q,self.numerators)
        phase=2*torch.pi*q[:,:,None]*self.frequencies
        values=torch.cat((2*q-1,torch.sin(phase).flatten(1),torch.cos(phase).flatten(1),
            context['global_feature'].expand(len(q),-1),spatial),1)
        h=self.net(values)
        return torch.cat((h[:,:1]*self.psi_scale,h[:,1:]*self.std[2]),1)

    def forward(self,q,context):
        if not q.requires_grad:raise ValueError('Continuous curl needs coordinategradients')
        h=self.heads(q,context)
        gradient=torch.autograd.grad(h[:,0].sum(),q,create_graph=True,retain_graph=True)[0]
        return torch.stack((gradient[:,1],-gradient[:,0],h[:,1]),1)

    def grid_prediction(self,q,context,ny=512,nx=512):
        if len(q)!=ny*nx:raise ValueError('One complete regulargrid required')
        feature=context['spatial'];sy,sx=feature.shape[1:]
        key=(ny,nx,sy,sx,str(feature.device),feature.dtype)
        if key not in self._coordinate_matrices:
            self._coordinate_matrices[key]=(
                axis_matrix(ny,sy,0.,1.,self.numerators,feature),
                axis_matrix(nx,sx,0.,1.,self.numerators,feature))
        wy,wx=self._coordinate_matrices[key]
        spatial=((wy[None]@feature)@wx.T).permute(1,2,0).reshape(-1,16)
        h=self.heads(q,context,spatial).reshape(ny,nx,2);psi=h[:,:,0]
        uy=torch.gradient(psi,spacing=(1/ny,),dim=(0,),edge_order=2)[0]
        vx=-torch.gradient(psi,spacing=(1/nx,),dim=(1,),edge_order=2)[0]
        return torch.stack((uy,vx,h[:,:,1]),2)

    def sampled_prediction(self,q,context,y,x,ny=512,nx=512):
        centre=self.heads(q,context)
        def stencil(index,size):
            offsets=torch.tensor([-1,0,1],device=q.device).expand(len(q),-1).clone()
            weights=q.new_tensor([-1.,0.,1.]).expand(len(q),-1).clone()
            first=index==0;last=index==size-1
            offsets[first]=torch.tensor([0,1,2],device=q.device);weights[first]=q.new_tensor([-3.,4.,-1.])
            offsets[last]=torch.tensor([-2,-1,0],device=q.device);weights[last]=q.new_tensor([1.,-4.,3.])
            return index[:,None]+offsets,weights
        ys,wy=stencil(y,ny);xs,wx=stencil(x,nx)
        qy=q[:,None,:].expand(-1,3,-1).clone();qy[:,:,1]=(ys+.5)/ny
        qx=q[:,None,:].expand(-1,3,-1).clone();qx[:,:,0]=(xs+.5)/nx
        hy=self.heads(qy.reshape(-1,2),context)[:,0].reshape(-1,3)
        hx=self.heads(qx.reshape(-1,2),context)[:,0].reshape(-1,3)
        return torch.stack(((hy*wy).sum(1)*ny/2,-(hx*wx).sum(1)*nx/2,centre[:,1]),1)


def steady_residual(q,value,reynolds):
    if q.shape[1]!=2 or not q.requires_grad:raise ValueError('Differentiable2D physicalcoordinates required')
    def grad(v):
        if not v.requires_grad:return q*0
        result=torch.autograd.grad(v,q,torch.ones_like(v),create_graph=True,retain_graph=True,allow_unused=True)[0]
        return q*0 if result is None else result
    u,v,p=value.unbind(1);gu,gv,gp=grad(u),grad(v),grad(p)
    lu=grad(gu[:,0])[:,0]+grad(gu[:,1])[:,1]
    lv=grad(gv[:,0])[:,0]+grad(gv[:,1])[:,1]
    return torch.stack((u*gu[:,0]+v*gu[:,1]+gp[:,0]-lu/reynolds,
        u*gv[:,0]+v*gv[:,1]+gp[:,1]-lv/reynolds,gu[:,0]+gv[:,1]),1)
