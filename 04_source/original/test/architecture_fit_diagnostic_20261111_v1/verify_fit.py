"""Read-only TRAIN/VAL fit diagnostic for the eight frozen point pipelines."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch

ROOT=Path('/mnt/project/perun2601449/dp_llm_cfd')


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def sample_validation(data,features,points):
    rng=np.random.default_rng(911119);xs=[];ys=[]
    mean=torch.tensor(data.normalization['target_mean']).float();std=torch.tensor(data.normalization['target_std']).float()
    with torch.inference_mode():
        for index in data.validation_ids:
            known=data.cases[index];available=torch.where(known['fluid'].flatten())[0]
            selected=available[torch.from_numpy(rng.integers(len(available),size=points))]
            xs.append(features(known,selected))
            target=torch.from_numpy(data.references[index].transpose(1,2,0).copy()).reshape(-1,3)[selected]
            ys.append((target-mean)/std)
    return torch.cat(xs),torch.cat(ys)


def score(core,x,y,ids,points,normalization,reynolds):
    with torch.inference_mode():pred=torch.cat([core(x[start:start+32768]) for start in range(0,len(x),32768)]).numpy()
    assert np.isfinite(pred).all();p=pred.astype('float64').reshape(len(ids),points,3);t=y.numpy().astype('float64').reshape(len(ids),points,3)
    error=p-t;std=np.asarray(normalization['target_std']);mean=np.asarray(normalization['target_mean'])
    physical_p=p*std+mean;physical_t=t*std+mean;physical_error=physical_p-physical_t
    raw_mse=(error*error).mean((0,1));baseline=(t*t).mean((0,1));cases=[]
    for k,index in enumerate(ids):
        pc=physical_p[k,:,2]-physical_p[k,:,2].mean();tc=physical_t[k,:,2]-physical_t[k,:,2].mean()
        cases.append(dict(index=index,reynolds=float(reynolds[index]),
            raw_normalized_mse=float(np.mean(error[k]**2)),
            sampled_velocity_l2=float(np.linalg.norm(physical_error[k,:,:2])/max(np.linalg.norm(physical_t[k,:,:2]),1e-12)),
            sample_centered_pressure_l2=float(np.linalg.norm(pc-tc)/max(np.linalg.norm(tc),1e-12)),
            pressure_reference_sample_rms=float(np.sqrt(np.mean(tc*tc)))))
    return dict(points=len(x),cases=cases,raw_normalized_mse_fields=raw_mse.tolist(),raw_normalized_mse=float(raw_mse.mean()),
        zero_normalized_output_mse_fields=baseline.tolist(),
        sampled_velocity_l2=float(np.mean([r['sampled_velocity_l2'] for r in cases])),
        sample_centered_pressure_l2=float(np.mean([r['sample_centered_pressure_l2'] for r in cases])))


def main():
    started=time.monotonic();fixture=ROOT/'results/cpu_gpu_benchmark_20261111_v1/fixture_f922ac12e7d38b1e';meta=read(fixture/'fixture.json');config=meta['config']
    source=ROOT/'test/cpu_gpu_benchmark_20261111_v1/revisions/r_35799e0a071b9fac/bench.py'
    spec=importlib.util.spec_from_file_location('frozen_benchmark',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.dependencies(config);module.configure(4,911118)
    from data_metrics import PilotData
    from models import FixedGeometryFeatures
    helper=ROOT/'test/architecture_pilot_20261110_v1/postprocessing/r_c65d7f2e9c9038c6/checkpoint_io.py'
    spec=importlib.util.spec_from_file_location('frozen_checkpoint_io',helper);io=importlib.util.module_from_spec(spec);spec.loader.exec_module(io)
    data=PilotData(dict(config,train_cases=640,validation_cases=160),'cpu');assert len(data.train_ids)==640 and len(data.validation_ids)==160
    reynolds={index:data.cases[index]['reynolds'] for index in data.train_ids+data.validation_ids}
    for name in ['x.npy','y.npy']:assert sha(fixture/name)==meta['files'][name]['sha256']
    tx=torch.from_numpy(np.load(fixture/'x.npy',allow_pickle=False));ty=torch.from_numpy(np.load(fixture/'y.npy',allow_pickle=False))
    vx,vy=sample_validation(data,FixedGeometryFeatures(data.normalization),2048)
    final=read(ROOT/'test/architecture_final_20261111_v1/revisions/r_11573498b33b3955/models.json')
    selected=[r for r in final['models'] if r['seed']==921101 and r['model'] not in ['tfno','pinn']];assert len(selected)==8
    records=[]
    for row in selected:
        assert sha(row['checkpoint'])==row['checkpoint_sha256'];cp=torch.load(row['checkpoint'],map_location='cpu',weights_only=True)
        model=io.restore_model(cp).eval().requires_grad_(False);assert cp['normalization']==data.normalization
        train=score(model.core,tx,ty,data.train_ids,2048,data.normalization,reynolds)
        val=score(model.core,vx,vy,data.validation_ids,2048,data.normalization,reynolds)
        result=dict(id=row['id'],model=row['model'],checkpoint_sha256=row['checkpoint_sha256'],train=train,validation=val)
        records.append(result);print(row['model'],'train',train['raw_normalized_mse'],'val',val['raw_normalized_mse'],flush=True);del model
    out=ROOT/'results/architecture_fit_diagnostic_20261111_v1';out.mkdir(exist_ok=True)
    result=dict(status='FROZEN_POINT_TRAIN_VAL_FIT_DIAGNOSTIC_COMPLETE',models=records,seed=921101,
        train_cases=640,validation_cases=160,points_per_case=2048,train_point_samples=1310720,validation_point_samples=327680,
        no_new_training=True,test_targets_opened=False,elapsed_seconds=time.monotonic()-started,
        note='Raw normalized loss matches the point-core objective. Sample-centered pressure L2 is a diagnostic on2048 points, not the native512x512 final metric. One frozen model per family, no hyperparameter tuning.')
    target=out/'fit_diagnostic.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'],flush=True)


if __name__=='__main__':main()
