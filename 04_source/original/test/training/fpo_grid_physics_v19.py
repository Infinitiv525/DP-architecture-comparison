"""Differentiable instantaneous NS residual of the actual emitted FD grid.

Only eroded interior cells are accepted. Pressure gauge cancels in gradients;
all stencil cells must be fluid. No exterior periodic or wall BC is introduced.
"""
import torch


def sampled_grid_terms(model,raw,context,y,x,time,reynolds,dt=.025):
    ny,nx=raw.shape[:2];hx=16/nx;hy=4/ny;n=len(y)
    ys=torch.stack((y,y,y,y+1,y-1));xs=torch.stack((x,x+1,x-1,x,x))
    times=time.expand(5,-1)
    q=torch.stack((4+(xs+.5)*hx,6+(ys+.5)*hy,times),2).reshape(-1,3)
    field=model.sampled_prediction(q,context,raw,ys.reshape(-1),xs.reshape(-1)).reshape(5,n,3)
    centre,xp,xm,yp,ym=field.unbind(0)
    qt=torch.stack((4+(x+.5)*hx,6+(y+.5)*hy,time-dt),1)
    minus=model.sampled_prediction(qt,context,raw,y,x)
    qt=torch.stack((4+(x+.5)*hx,6+(y+.5)*hy,time+dt),1)
    plus=model.sampled_prediction(qt,context,raw,y,x)
    temporal=(plus[:,:2]-minus[:,:2])/(2*dt)
    dx=(xp-xm)/(2*hx);dy=(yp-ym)/(2*hy)
    lap=(xp+xm-2*centre)/(hx*hx)+(yp+ym-2*centre)/(hy*hy)
    advection=centre[:,:1]*dx[:,:2]+centre[:,1:2]*dy[:,:2]
    pressure_gradient=torch.stack((dx[:,2],dy[:,2]),1)
    diffusion=-lap[:,:2]/reynolds
    return dict(velocity=centre[:,:2],temporal=temporal,advection=advection,
        pressure_gradient=pressure_gradient,diffusion=diffusion,
        momentum=temporal+advection+pressure_gradient+diffusion,
        divergence=(dx[:,0]+dy[:,1])[:,None])


def momentum_loss(terms,velocity_scale):
    return terms['momentum'].square().sum(1).mean()/velocity_scale.clamp_min(1e-4).square()
