"""Pure checks for the shared steady2D FlowBench task."""
import numpy as np


def standard_split(n=1000,seed=20261104):
    if n<10:raise ValueError('Need at least10cases')
    order=np.random.default_rng(seed).permutation(n)
    ntest=n//5;remaining=order[ntest:];nval=len(remaining)//5
    return dict(train=remaining[nval:].tolist(),validation=remaining[:nval].tolist(),test=order[:ntest].tolist())


def input_fields(sample):
    if sample.ndim!=3 or sample.shape[0]!=3 or not np.isfinite(sample).all():raise ValueError('Invalid3channel input')
    re=float(sample[0,0,0])
    if not 0<re<=1e5 or not np.all(sample[0]==re):raise ValueError('Reynolds channel not positiveconstant')
    candidates=[i for i in [1,2] if set(np.unique(sample[i]))<=set([0.,1.,255.]) and len(np.unique(sample[i]))==2]
    if len(candidates)!=1:raise ValueError('Cannot unambiguously identify binarygeometrychannel')
    mask_channel=candidates[0];sdf_channel=3-mask_channel
    binary=sample[mask_channel]
    edges=np.concatenate((binary[0],binary[-1],binary[:,0],binary[:,-1]))
    if len(np.unique(edges))!=1:raise ValueError('Centered obstacle must not touch cavitywalls')
    fluid=binary==edges[0]
    if not fluid.any() or fluid.all():raise ValueError('Missing obstacle/fluid')
    sdf=sample[sdf_channel]
    edge_sdf=np.concatenate((sdf[0],sdf[-1],sdf[:,0],sdf[:,-1]))
    sign=1. if np.median(edge_sdf)>0 else -1.
    oriented=sdf*sign
    reliable=np.abs(sdf)>1e-8
    if np.mean(((oriented>0)==fluid)[reliable])<.999:raise ValueError('SDF/binary mask disagree')
    return re,fluid,oriented,dict(reynolds_channel=0,binary_channel=mask_channel,sdf_channel=sdf_channel,sdf_orientation=sign)


def canonical_target(sample,fluid):
    if sample.shape[0]!=4 or sample.shape[1:]!=fluid.shape:raise ValueError('Expected u/v/p/property tensor')
    if not np.isfinite(sample[:3,fluid]).all():raise ValueError('Nonfinite fluidreference')
    value=np.array(sample[:3],dtype=np.float32,copy=True)
    value[2]-=value[2,fluid].mean(dtype=np.float64)
    return np.where(fluid[None],value,0.)


def eroded_fluid(fluid,radius=3):
    mask=fluid.copy()
    for d in range(1,radius+1):
        for axis in [0,1]:mask &= np.roll(fluid,d,axis)&np.roll(fluid,-d,axis)
    mask[:radius]=False;mask[-radius:]=False;mask[:,:radius]=False;mask[:,-radius:]=False
    return mask


def reference_physics(value,fluid,re):
    """Normalized coordinates[0,1]^2; test viscosity1/Re against TRAIN data."""
    u,v,p=np.asarray(value,dtype=np.float64);h=1/u.shape[0];ok=eroded_fluid(fluid)
    if ok.sum()<100:raise ValueError('Insufficient fluidinterior')
    dx=lambda a:(np.roll(a,-1,1)-np.roll(a,1,1))/(2*h)
    dy=lambda a:(np.roll(a,-1,0)-np.roll(a,1,0))/(2*h)
    lap=lambda a:(np.roll(a,-1,1)+np.roll(a,1,1)+np.roll(a,-1,0)+np.roll(a,1,0)-4*a)/(h*h)
    adv=np.stack((u*dx(u)+v*dy(u),u*dx(v)+v*dy(v)),0)[:,ok]
    pressure=np.stack((dx(p),dy(p)),0)[:,ok]
    diffusion=np.stack((lap(u),lap(v)),0)[:,ok]/re
    force=adv+pressure;res=force-diffusion
    norm=lambda a:float(np.linalg.norm(a))
    coefficient=float(np.sum(force*diffusion)/max(np.sum(diffusion**2),1e-24))
    return dict(momentum_rms=float(np.sqrt(np.mean(res**2))),
        momentum_relative=norm(res)/max(norm(adv)+norm(pressure)+norm(diffusion),1e-24),
        inferred_viscosity_multiplier=coefficient,divergence_rms=float(np.sqrt(np.mean((dx(u)+dy(v))[ok]**2))),
        interior_points=int(ok.sum()))
