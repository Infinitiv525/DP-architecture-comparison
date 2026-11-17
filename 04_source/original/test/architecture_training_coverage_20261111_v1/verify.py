"""Read-only reconstruction of all640 cases contributing the point fixture."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import numpy as np
import torch

ROOT=Path('/mnt/project/perun2601449/dp_llm_cfd')
FIXTURE=ROOT/'results/cpu_gpu_benchmark_20261111_v1/fixture_f922ac12e7d38b1e'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    start=time.monotonic();meta=read(FIXTURE/'fixture.json');config=meta['config']
    baseline=read(ROOT/'results/architecture_calibration_20261111_v1/run_12669e7239a2e28c/prepared/ready.json')
    assert sha(FIXTURE/'fixture.json')==baseline['fixture_sha256']
    source=ROOT/'test/cpu_gpu_benchmark_20261111_v1/revisions/r_35799e0a071b9fac/bench.py'
    spec=importlib.util.spec_from_file_location('frozen_benchmark',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.dependencies(config);module.configure(4,config['seed'])
    from data_metrics import PilotData
    from models import FixedGeometryFeatures
    data=PilotData(config,'cpu');split=read(ROOT/'results/ldc_pinn_prepare_20261104_v25/split.json')['indices']
    assert data.train_ids==split['train']==meta['data_plan']['train_ids'] and len(data.train_ids)==640
    assert config['training_points_per_case']==2048
    x,y,sample_hash=data.training_features(FixedGeometryFeatures(data.normalization),2048,config['seed']+100)
    assert x.shape==(1310720,57) and y.shape==(1310720,3) and sample_hash==meta['point_sample_sha256']
    arrays=[]
    for name,value in [('x.npy',x.numpy()),('y.npy',y.numpy())]:
        assert sha(FIXTURE/name)==meta['files'][name]['sha256']
        stored=np.load(FIXTURE/name,mmap_mode='r',allow_pickle=False)
        maximum=float(np.max(np.abs(value-stored)));assert np.allclose(value,stored,rtol=1e-6,atol=1e-7)
        arrays.append(dict(file=name,shape=list(value.shape),bitwise_equal=bool(np.array_equal(value,stored)),max_abs_difference=maximum))
    source=ROOT/'test/architecture_main_20261111_v1/revisions/r_6015802a53cffb3c'
    run=ROOT/'results/architecture_main_20261111_v1/run_6015802a53cffb3c';selection=read(source/'selection.json');rows=[]
    for task in selection['tasks']:
        recipe=next(r['recipe'] for r in selection['selected'] if r['model']==task['model'])
        folder=run/'models'/task['id'];phase=folder/'training'/recipe['id'];summary=read(phase/'summary.json');plan=read(phase/'data_plan.json')
        assert plan['train_ids']==data.train_ids and len(plan['train_ids'])==640
        if task['model']=='tfno':
            history=summary['training']['history'];assert len(history)==200 and all(r['updates']==640 for r in history)
            assert summary['training']['optimizer_updates']==128000
        else:
            assert summary['point_sample_sha256']==sample_hash
            if task['model'] in ['chebkan','pinet','dendrite','realnpu','rbf']:
                history=summary['training']['history'];assert len(history)==200 and all(r['updates']==160 for r in history)
                assert summary['training']['optimizer_updates']==32000
                if task['model']=='chebkan':
                    extension=read(folder/'continuation/summary.json');assert len(extension['history'])==200
                    assert all(r['updates']==160 for r in extension['history'])
            else:assert summary['training']['algorithm']==task['model']
        rows.append(dict(id=task['id'],model=task['model'],train_cases=640,point_samples=None if task['model']=='tfno' else 1310720))
    assert len(rows)==45
    old=read(ROOT/'results/ldc_final_test_20261108_v49/models.json');pinn=[]
    for reference in selection['archived_pinn']:
        record=next(r for r in old['models'] if r['id']==reference['id']);summary=read(record['training_summary'])
        assert set(map(int,summary['case_counts']))==set(data.train_ids) and set(summary['case_counts'].values())=={400}
        pinn.append(dict(id=reference['id'],train_cases=640,uses_per_case=400,training_seconds=summary['training_seconds']))
    result=dict(status='ALL50_TRAIN_CASE_COVERAGE_AUDIT_PASS',new_models=rows,archived_pinn=pinn,
        reconstruction=arrays,point_sample_sha256=sample_hash,fixture_metadata_sha256=sha(FIXTURE/'fixture.json'),
        cases=[dict(index=index,point_rows=2048,start=k*2048,stop=(k+1)*2048) for k,index in enumerate(data.train_ids)],
        notes='Same640 scenarios; FNO full fields; point pipelines fixed2048 fluid samples/case; archived PINN resamples data/physics points each epoch. Coverage is not an accuracy guarantee.',
        elapsed_seconds=time.monotonic()-start,test_targets_opened=False,new_training_started=False,job_id=os.environ.get('SLURM_JOB_ID'))
    output=ROOT/'results/architecture_training_coverage_20261111_v1';output.mkdir(exist_ok=True)
    target=output/'coverage.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'],json.dumps(arrays),flush=True)


if __name__=='__main__':main()
