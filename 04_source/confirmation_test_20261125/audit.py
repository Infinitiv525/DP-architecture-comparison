"""Independent CPU re-scoring of every saved TEST field and timing trace."""
import argparse
import csv
from pathlib import Path
import time
import numpy as np
from common import SOURCE,read,save,digest,dependencies,legacy_paths,configure
from evaluate import Data,check_field


def main(config,root):
    manifest=dependencies(config);configure('cpu',911115)
    from data_metrics import field_metrics
    groups=[read(root/'test/groups'/('group_'+str(i)+'.json')) for i in range(12)]
    assert all(r['status']=='FINAL_GROUP_TEST_COMPLETE' and r['models_sha256']==config['models_sha256'] for r in groups)
    ids=[m['model']['id'] for g in groups for m in g['models']]
    assert len(ids)==len(set(ids))==60 and set(ids)=={r['id'] for r in manifest['models']}
    final=dict(status='CONFIRMATION60_TEST_COMPLETE',models=[m for g in groups for m in g['models']],total_models=60,total_case_predictions=12000,models_sha256=config['models_sha256'],no_weight_updates=True,no_test_selection=True)
    save(root/'test/summary.json',final)
    data=Data(config,root,'test',device='cpu',access_tag='saved_field_audit');out=root/'audit';out.mkdir(parents=True,exist_ok=False)
    rows=[];models=[];started=time.monotonic()
    for model in manifest['models']:
        folder=root/'test/models'/model['id'];summary=read(folder/'summary.json')
        assert summary['status']=='CONFIRMATION60_MODEL_TEST_COMPLETE' and summary['model']==model
        assert summary['models_sha256']==config['models_sha256'] and summary['no_weight_updates'] and summary['no_test_selection']
        assert digest(model['checkpoint'])==model['checkpoint_sha256']
        assert digest(folder/'predictions.npy')==summary['predictions_sha256']
        assert digest(folder/'raw_timings.npz')==summary['timings_sha256']
        prediction=np.load(folder/'predictions.npy',mmap_mode='r',allow_pickle=False)
        assert prediction.shape==(200,512,512,3) and prediction.dtype==np.dtype('float32')
        with np.load(folder/'raw_timings.npz',allow_pickle=False) as raw:
            gpu=raw['gpu_ms'];host=raw['host_instrumented_ms'];assert raw['ids'].tolist()==data.ids
        assert gpu.shape==host.shape==(200,config['latency_repetitions'])
        assert np.isfinite(gpu).all() and np.isfinite(host).all() and np.all(gpu>0) and np.all(host>0)
        assert [r['index'] for r in summary['cases']]==data.ids
        for position,case in enumerate(summary['cases']):
            index=case['index'];pred=prediction[position];fluid=data.fluid[index];check_field(pred,fluid)
            reference=data.reference[index].astype('float64');error=pred.astype('float64')-reference
            values=dict(all_cell_mse=float(np.mean(error*error)),
                fluid_velocity_relative_l2=float(np.linalg.norm(error[fluid,:2])/max(np.linalg.norm(reference[fluid,:2]),1e-12)),
                fluid_pressure_relative_l2=float(np.linalg.norm(error[fluid,2])/max(np.linalg.norm(reference[fluid,2]),1e-12)))
            for key,value in values.items():assert np.isclose(value,case[key],rtol=1e-11,atol=1e-13),(model['id'],index,key)
            for label,array in [('gpu',gpu),('host_instrumented',host)]:
                for p in [50,95,99]:assert np.isclose(np.percentile(array[position],p),case['timing'][label]['p%d_ms'%p],rtol=1e-12,atol=1e-12)
                assert int(np.sum(array[position]>config['latency_target_ms']))==case['timing'][label]['above_target']
            if True:  # Independently verify physics for every saved field.
                again=field_metrics(pred,data.reference[index],fluid,case['reynolds'])
                for key in ['prediction_physics','reference_physics']:
                    for metric in ['divergence_rms','steady_ns_momentum_rms']:
                        assert np.isclose(again[key][metric],case[key][metric],rtol=1e-12,atol=1e-12)
            rows.append(dict(model_id=model['id'],architecture=model['model'],role=model['kind'],seed=model['seed'],index=index,
                geometry_hash=case['geometry_hash'],reynolds=case['reynolds'],**values,
                mse_u=case['mse_fields']['u'],mse_v=case['mse_fields']['v'],mse_p=case['mse_fields']['p'],
                divergence_rms=case['prediction_physics']['divergence_rms'],momentum_rms=case['prediction_physics']['steady_ns_momentum_rms'],
                reference_divergence_rms=case['reference_physics']['divergence_rms'],reference_momentum_rms=case['reference_physics']['steady_ns_momentum_rms'],
                **{label+'_p%d_ms'%p:case['timing'][label]['p%d_ms'%p] for label in ['gpu','host_instrumented'] for p in [50,95,99]}))
        means={key:float(np.mean([r[key] for r in rows[-200:]])) for key in ['all_cell_mse','fluid_velocity_relative_l2','fluid_pressure_relative_l2']}
        for key in means:assert np.isclose(means[key],summary['means'][key],rtol=1e-12,atol=1e-13)
        record=dict(id=model['id'],model=model['model'],role=model['kind'],seed=model['seed'],status='CONFIRMATION60_SAVED_OUTPUT_AUDIT_PASS',
            **means,worst_gpu_p99_ms=float(np.percentile(gpu,99,axis=1).max()),worst_host_p99_ms=float(np.percentile(host,99,axis=1).max()),
            gpu_5ms_pass=summary['gpu_5ms_pass'],host_instrumented_5ms_pass=summary['host_instrumented_5ms_pass'])
        assert record['gpu_5ms_pass']==(record['worst_gpu_p99_ms']<=config['latency_target_ms'])
        assert record['host_instrumented_5ms_pass']==(record['worst_host_p99_ms']<=config['latency_target_ms'])
        models.append(record);save(out/'progress.json',models);print('AUDITED',model['id'],flush=True);del prediction
    assert len(rows)==12000 and len(models)==60
    for name,table in [('case_metrics.csv',rows),('model_metrics.csv',models)]:
        with (out/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    methods=[]
    for name in dict.fromkeys(r['model'] for r in models):
        values=[r for r in models if r['model']==name];assert len(values)==20
        result=dict(model=name,role=values[0]['role'],independent_models=20,gpu_5ms_pass_models=sum(r['gpu_5ms_pass'] for r in values),host_5ms_pass_models=sum(r['host_instrumented_5ms_pass'] for r in values))
        for key in ['all_cell_mse','fluid_velocity_relative_l2','fluid_pressure_relative_l2','worst_gpu_p99_ms','worst_host_p99_ms']:
            result[key+'_mean']=float(np.mean([r[key] for r in values]));result[key+'_sample_sd']=float(np.std([r[key] for r in values],ddof=1))
        methods.append(result)
    save(out/'summary.json',dict(status='CONFIRMATION60_TEST_AND_SAVED_OUTPUT_AUDIT_COMPLETE',models=models,methods=methods,
        saved_native_fields_checked=12000,raw_timing_pairs_checked=12000000,models_sha256=config['models_sha256'],
        no_weight_updates=True,no_test_selection=True,elapsed_seconds=time.monotonic()-started))
    text=['POTVRDZUJUCI TEST ARCHITEKTUR DOKONCENY','',
        '60 novych modelov x200 TEST pripadov =12000 overenych poli.',
        'Po20 nezavislych treningov TFNO,denseFNO aRealNPU. Starych5pilotnych modelov nie je v tomto novom potvrdeni.',
        'Vsetky ulozene polia, casovania a fyzikalne diagnostiky boli skontrolovane.',
        'Merania su FP32,H200,bez TF32,vstupy uz naGPU; bez prenosu a vykreslovania.',
        'Toto je opisny report. Parove statisticke porovnanie20semien nasleduje samostatne.','']
    for r in methods:
        text.append('%s MSE=%.12g uL2=%.9g pL2=%.9g GPU5ms=%d/20 host5ms=%d/20' %(
            r['model'],r['all_cell_mse_mean'],r['fluid_velocity_relative_l2_mean'],
            r['fluid_pressure_relative_l2_mean'],r['gpu_5ms_pass_models'],r['host_5ms_pass_models']))
    (out/'VYSLEDOK.txt').write_text('\n'.join(text)+'\n')



if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(read(SOURCE/'config.json'),a.root)
