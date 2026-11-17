"""Regional errors with a single whole-fluid pressure gauge per field."""
import numpy as np

ZONE_NAMES=['near_0_to_0.125','middle_0.125_to_0.5','far_over_0.5']


def zones_from_distance(fluid,distance):
    if fluid.shape!=distance.shape or not np.isfinite(distance).all():raise ValueError('Invalid distance')
    zones=np.stack((fluid&(distance<=.125),fluid&(distance>.125)&(distance<=.5),fluid&(distance>.5)))
    if not np.array_equal(zones.sum(0),fluid.astype(int)):raise ValueError('Regions do not partition fluid')
    return zones


def regional_errors(pred,reference,fluid,zones):
    if not np.array_equal(zones.sum(0),fluid.astype(int)):raise ValueError('Nonpartitioned mask')
    p=np.array(pred[fluid],dtype=np.float64,copy=True);r=np.array(reference[fluid],dtype=np.float64,copy=True)
    p[:,2]-=p[:,2].mean();r[:,2]-=r[:,2].mean()
    delta=(p-r)**2;energy=r*r
    total_error_u=float(delta[:,:2].sum());total_error_p=float(delta[:,2].sum())
    total_energy_u=float(energy[:,:2].sum());total_energy_p=float(energy[:,2].sum())
    result={}
    for name,mask in [('all_fluid',np.ones(len(p),dtype=bool))]+[(name,z[fluid]) for name,z in zip(ZONE_NAMES,zones)]:
        n=int(mask.sum())
        if n==0:result[name]=dict(points=0,area_fraction=0.,empty=True);continue
        eu=float(delta[mask,:2].sum());ep=float(delta[mask,2].sum());ru=float(energy[mask,:2].sum());rp=float(energy[mask,2].sum())
        result[name]=dict(points=n,area_fraction=n/len(p),empty=False,
            velocity_relative_l2=float(np.sqrt(eu/ru)) if ru>1e-24 else None,
            pressure_relative_l2=float(np.sqrt(ep/rp)) if rp>1e-24 else None,
            velocity_rmse=float(np.sqrt(eu/n)),pressure_rmse=float(np.sqrt(ep/n)),
            reference_velocity_rms=float(np.sqrt(ru/n)),reference_pressure_rms=float(np.sqrt(rp/n)),
            velocity_squared_error=eu,pressure_squared_error=ep,
            velocity_error_fraction=eu/total_error_u if total_error_u>1e-24 else 0.,
            pressure_error_fraction=ep/total_error_p if total_error_p>1e-24 else 0.,
            reference_velocity_energy_fraction=ru/total_energy_u if total_energy_u>1e-24 else 0.,
            reference_pressure_energy_fraction=rp/total_energy_p if total_energy_p>1e-24 else 0.)
    return result


def prediction_change(pred,original,fluid):
    p=np.array(pred[fluid],dtype=np.float64,copy=True);r=np.array(original[fluid],dtype=np.float64,copy=True)
    p[:,2]-=p[:,2].mean();r[:,2]-=r[:,2].mean();d=(p-r)**2
    return dict(velocity_rmse=float(np.sqrt(d[:,:2].sum()/len(p))),pressure_rmse=float(np.sqrt(d[:,2].mean())),
        velocity_relative_l2=float(np.sqrt(d[:,:2].sum()/max(float((r[:,:2]**2).sum()),1e-24))))


def contexts(model,known,donor_geometry):
    import torch
    raw=known['raw'];ny,nx=raw.shape[:2];re=known['record']['reynolds']
    low=raw.reshape(ny//8,8,nx//8,8,6).mean((1,3)).permute(2,0,1)[None]
    original=model.context(known['geometry'],low,re)
    neutral=low.clone();neutral[:,:3]=model.mean.reshape(1,3,1,1);neutral[:,3:]=0
    latent_zero=original.clone();latent_zero[:,:-1]=0
    return dict(original=original,geometry_neutral=model.context(torch.zeros_like(known['geometry']),low,re),
        global_state_neutral=model.context(known['geometry'],neutral,re),latent_zero=latent_zero,
        geometry_swap=model.context(donor_geometry,low,re))
