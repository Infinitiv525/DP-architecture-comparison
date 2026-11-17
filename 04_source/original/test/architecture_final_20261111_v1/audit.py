"""Independent CPU re-scoring of every saved TEST field and timing trace."""
import argparse
import csv
from pathlib import Path
import time
import numpy as np
from common import SOURCE,read,save,digest,dependencies,legacy_paths
from evaluate import Data,check_field


def main(config,root):
    e,c,manifest=dependencies(config);legacy_paths(config);e.configure('cpu',4,911115)
    from data_metrics import field_metrics
    final=read(root/'test/summary.json');assert final['status']=='FINAL50_TEST_COMPLETE' and final['total_case_predictions']==10000
    data=Data(config,root,'test',device='cpu');out=root/'audit';out.mkdir(parents=True,exist_ok=False)
    rows=[];models=[];started=time.monotonic()
    for model in manifest['models']:
        folder=root/'test/models'/model['id'];summary=read(folder/'summary.json')
        assert summary['status']=='FINAL50_MODEL_TEST_COMPLETE' and summary['model']==model
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
            if position==0:
                again=field_metrics(pred,data.reference[index],fluid,case['reynolds'])
                for key in ['prediction_physics','reference_physics']:
                    for metric in ['divergence_rms','steady_ns_momentum_rms']:
                        assert np.isclose(again[key][metric],case[key][metric],rtol=1e-12,atol=1e-12)
            rows.append(dict(model_id=model['id'],architecture=model['model'],seed=model['seed'],index=index,
                geometry_hash=case['geometry_hash'],reynolds=case['reynolds'],**values,
                mse_u=case['mse_fields']['u'],mse_v=case['mse_fields']['v'],mse_p=case['mse_fields']['p'],
                divergence_rms=case['prediction_physics']['divergence_rms'],momentum_rms=case['prediction_physics']['steady_ns_momentum_rms'],
                reference_divergence_rms=case['reference_physics']['divergence_rms'],reference_momentum_rms=case['reference_physics']['steady_ns_momentum_rms'],
                **{label+'_p%d_ms'%p:case['timing'][label]['p%d_ms'%p] for label in ['gpu','host_instrumented'] for p in [50,95,99]}))
        means={key:float(np.mean([r[key] for r in rows[-200:]])) for key in ['all_cell_mse','fluid_velocity_relative_l2','fluid_pressure_relative_l2']}
        for key in means:assert np.isclose(means[key],summary['means'][key],rtol=1e-12,atol=1e-13)
        record=dict(id=model['id'],model=model['model'],seed=model['seed'],status='FINAL50_SAVED_OUTPUT_AUDIT_PASS',
            **means,worst_gpu_p99_ms=float(np.percentile(gpu,99,axis=1).max()),worst_host_p99_ms=float(np.percentile(host,99,axis=1).max()),
            gpu_5ms_pass=summary['gpu_5ms_pass'],host_instrumented_5ms_pass=summary['host_instrumented_5ms_pass'])
        assert record['gpu_5ms_pass']==(record['worst_gpu_p99_ms']<=config['latency_target_ms'])
        assert record['host_instrumented_5ms_pass']==(record['worst_host_p99_ms']<=config['latency_target_ms'])
        models.append(record);save(out/'progress.json',models);print('AUDITED',model['id'],flush=True);del prediction
    assert len(rows)==10000 and len(models)==50
    for name,table in [('case_metrics.csv',rows),('model_metrics.csv',models)]:
        with (out/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    methods=[]
    for name in dict.fromkeys(r['model'] for r in models):
        values=[r for r in models if r['model']==name];assert len(values)==5
        result=dict(model=name,independent_models=5,gpu_5ms_pass_models=sum(r['gpu_5ms_pass'] for r in values),host_5ms_pass_models=sum(r['host_instrumented_5ms_pass'] for r in values))
        for key in ['all_cell_mse','fluid_velocity_relative_l2','fluid_pressure_relative_l2','worst_gpu_p99_ms','worst_host_p99_ms']:
            result[key+'_mean']=float(np.mean([r[key] for r in values]));result[key+'_sample_sd']=float(np.std([r[key] for r in values],ddof=1))
        methods.append(result)
    save(out/'summary.json',dict(status='FINAL50_TEST_AND_SAVED_OUTPUT_AUDIT_COMPLETE',models=models,methods=methods,
        saved_native_fields_checked=10000,raw_timing_pairs_checked=10000000,models_sha256=config['models_sha256'],
        no_weight_updates=True,no_test_selection=True,elapsed_seconds=time.monotonic()-started))
    text=['ZAVERECNE TESTOVANIE ARCHITEKTUR DOKONCENE','',
        '50 modelov x200 TEST simulacii =10000 overenych predikovanych poli.',
        'Zachovane vsetkych5 opakovani kazdej metody; ziadny vyber podlaTEST.',
        'Surove polia a casovania su naPerune. CSV obsahuje chyby a latencie.',
        'GPU a merany cas volania zo systemu su vykazane samostatne.',
        'Merany cas hostitela zahrna dve CUDA udalosti a synchronizaciu;',
        'prenos vstupov naGPU a vizualizacia nie su v tomto merani.','',
        'Metoda          TEST MSE       rychlost L2[%] tlak L2[%] GPU<=5ms host<=5ms']
    for r in methods:
        text.append('%-14s %.8g %13.3f %10.3f %8d/5 %8d/5'%(r['model'],r['all_cell_mse_mean'],100*r['fluid_velocity_relative_l2_mean'],100*r['fluid_pressure_relative_l2_mean'],r['gpu_5ms_pass_models'],r['host_5ms_pass_models']))
    text+=['','Ide o konkretne zdokumentovane postupy s roznymi vstupnymi',
        'reprezentaciami a hustotou supervizie; PINN je pomocna referencia.',
        'Split obsahuje zname geometrie. TEST uz bol otvoreny v starej studii.',
        'Toto je opisne vyhodnotenie; samostatne nevyhlasuje statisticku vyznamnost.']
    (out/'VYSLEDOK.txt').write_text('\n'.join(text)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(read(SOURCE/'config.json'),a.root)
