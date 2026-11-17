"""Instantaneous NS terms, compared at identical physical locations/times."""
import torch


def continuous_terms(q,value,reynolds):
    def grad(a):
        if not a.requires_grad:return q*0
        g=torch.autograd.grad(a,q,torch.ones_like(a),create_graph=True,retain_graph=True,allow_unused=True)[0]
        return q*0 if g is None else g+q*0
    u,v,p=value.unbind(1);gu,gv,gp=grad(u),grad(v),grad(p)
    lapu=grad(gu[:,0])[:,0]+grad(gu[:,1])[:,1]
    lapv=grad(gv[:,0])[:,0]+grad(gv[:,1])[:,1]
    temporal=torch.stack((gu[:,2],gv[:,2]),1)
    advection=torch.stack((u*gu[:,0]+v*gu[:,1],u*gv[:,0]+v*gv[:,1]),1)
    pressure_gradient=gp[:,:2]
    diffusion=-torch.stack((lapu,lapv),1)/reynolds
    return dict(velocity=value[:,:2],temporal=temporal,advection=advection,
        pressure_gradient=pressure_gradient,diffusion=diffusion,
        momentum=temporal+advection+pressure_gradient+diffusion,
        divergence=(gu[:,0]+gv[:,1])[:,None])


def time_stencil(t,step,end=11.9):
    if not 0<=t<=end or step<=0:raise ValueError('Invalid time/step')
    times=[t-step,t+step] if t+step<=end+1e-12 else [t-step,t-2*step]
    if min(times)<-1e-12 or max(times)>end+1e-12:raise ValueError('Stencil outside available horizon')
    return times,t+step>end+1e-12


def temporal_difference(current,minus,other,step,backward=False):
    if backward:return (3*current[...,:2]-4*minus[...,:2]+other[...,:2])/(2*step)
    return (other[...,:2]-minus[...,:2])/(2*step)


def grid_terms(field,temporal,yx,reynolds,h=1/64):
    y,x=yx[:,0],yx[:,1]
    centre=field[y,x];xp=field[y,x+1];xm=field[y,x-1];yp=field[y+1,x];ym=field[y-1,x]
    dx=(xp-xm)/(2*h);dy=(yp-ym)/(2*h)
    lap=(xp+xm+yp+ym-4*centre)/(h*h)
    advection=centre[:,:1]*dx[:,:2]+centre[:,1:2]*dy[:,:2]
    pressure_gradient=torch.stack((dx[:,2],dy[:,2]),1)
    diffusion=-lap[:,:2]/reynolds
    return dict(velocity=centre[:,:2],temporal=temporal[y,x],advection=advection,
        pressure_gradient=pressure_gradient,diffusion=diffusion,
        momentum=temporal[y,x]+advection+pressure_gradient+diffusion,
        divergence=(dx[:,0]+dy[:,1])[:,None])


def describe(terms,scale):
    scale=max(float(scale),1e-4)
    return {k:dict(rms=float(v.detach().square().sum(1).mean().sqrt()),
                   mean_square_over_velocity_scale2=float(v.detach().square().sum(1).mean())/(scale*scale))
            for k,v in terms.items()}


def compare(a,b):
    return {k:dict(difference_rms=float((a[k].detach()-b[k].detach()).square().sum(1).mean().sqrt()),
                   relative_l2=float(((a[k].detach()-b[k].detach()).square().sum()/a[k].detach().square().sum().clamp_min(1e-24)).sqrt()))
            for k in a}
