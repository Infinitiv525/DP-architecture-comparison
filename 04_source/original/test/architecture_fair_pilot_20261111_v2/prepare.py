"""Shared complete TRAIN/VAL cache on Perun, without TEST labels or fitting."""
import argparse
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from common import load_data, read, save, setup
from field_adapter import FullFieldAdapter


def main(root):
    setup('cpu',931101); folder=Path(root)/'fixture'; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'ready.json').exists():
        assert read(folder/'ready.json')['status']=='FULL_NATIVE_TRAIN_VAL_FIXTURE_READY'
        print('Existing complete fixture',flush=True); return
    start=time.perf_counter(); data=load_data(640,160)
    adapter=FullFieldAdapter(data.normalization); global_descriptors={}; checks={}
    save(folder/'data_plan.json',data.plan)
    for phase,ids in [('train',data.train_ids),('validation',data.validation_ids)]:
        x=np.lib.format.open_memmap(folder/(phase+'_x.npy'),mode='w+',dtype=np.float32,shape=(len(ids),4096,665))
        y=np.lib.format.open_memmap(folder/(phase+'_y.npy'),mode='w+',dtype=np.float32,shape=(len(ids),4096,192))
        weights=np.lib.format.open_memmap(folder/(phase+'_weights.npy'),mode='w+',dtype=np.float32,shape=(len(ids),3))
        h=hashlib.sha256()
        for slot,index in enumerate(ids):
            known=data.cases[index]; target=torch.tensor(data.references[index].transpose(1,2,0).copy())
            with torch.no_grad():
                encoded=adapter.encode(known['geometry'],known['re'])
                xp=encoded.permute(0,2,3,1).reshape(4096,665)
                yp=adapter.target_patches(target).permute(0,2,3,1).reshape(4096,192)
                energy=torch.tensor([float(target[...,:2].square().sum())]*2+[float(target[...,2].square().sum())]).clamp_min(1e-12)
                w=.5*adapter.target_std.flatten().square()/energy
                x[slot]=xp.numpy(); y[slot]=yp.numpy(); weights[slot]=w.numpy()
                h.update(np.asarray([index],dtype='<i8').tobytes())
                h.update(x[slot].tobytes()); h.update(y[slot].tobytes()); h.update(weights[slot].tobytes())
                if phase=='train':
                    descriptor=xp[0,128:640].numpy().copy(); key=known['geometry_hash']
                    if key in global_descriptors: assert np.array_equal(global_descriptors[key],descriptor)
                    else: global_descriptors[key]=descriptor
            if slot%40==0: print(phase,slot,len(ids),flush=True)
        x.flush(); y.flush(); weights.flush(); checks[phase]=h.hexdigest(); del x,y,weights
    descriptors=np.stack(list(global_descriptors.values())).astype('float64')
    centered=descriptors-descriptors.mean(0); singular=np.linalg.svd(centered,compute_uv=False)
    distances=np.linalg.norm(descriptors[:,None]-descriptors[None],axis=2)
    np.fill_diagonal(distances,np.inf)
    rank=int((singular>singular[0]*1e-7).sum()) if singular[0] else 0
    audit=dict(unique_TRAIN_geometries=len(descriptors),centered_descriptor_rank=rank,
        maximum_centered_rank=len(descriptors)-1,relative_rank_tolerance=1e-7,
        minimum_distinct_geometry_descriptor_distance=float(distances.min()),
        singular_values=singular.tolist(),targets_used_for_projection=False,
        full_raw_geometry_also_preserved=True)
    if not distances.min()>0: raise ValueError('Global descriptor collision between TRAIN geometries')
    save(folder/'descriptor_audit.json',audit)
    from smoke import smoke
    smoke(folder,data.normalization)
    save(folder/'ready.json',dict(status='FULL_NATIVE_TRAIN_VAL_FIXTURE_READY',train_cases=640,validation_cases=160,
        encoded_shape_per_case=[4096,665],target_shape_per_case=[4096,192],
        target_native_values_per_case=786432,supervision='every native cell including canonical solid zeros',
        sha256_case_order_and_x_y_weights=checks,normalization=data.normalization,
        descriptor_audit=audit,test_targets_read=False,preparation_seconds=time.perf_counter()-start,
        files_bytes={p.name:p.stat().st_size for p in folder.glob('*.npy')}))
    print('FULL_NATIVE_TRAIN_VAL_FIXTURE_READY',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True); main(p.parse_args().root)
