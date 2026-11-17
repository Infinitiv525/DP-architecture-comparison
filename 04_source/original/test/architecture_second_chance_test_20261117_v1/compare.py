"""Pair the final supplemental TEST with the same four original baselines."""
import csv
from pathlib import Path
import numpy as np
from eval_common import SOURCE,read,save,dependencies,digest


def main(root):
    config=read(SOURCE/'config.json');manifest=dependencies(config)
    assert read(root/'audit/summary.json')['status']=='SECOND_CHANCE4_TEST_AND_SAVED_OUTPUT_AUDIT_COMPLETE'
    models=[];cases=[]
    for row in manifest['models']:
        before=read(Path(config['original_baseline_result_root'])/'test/models'/row['original_baseline_id']/'summary.json')
        after=read(root/'test/models'/row['id']/'summary.json')
        assert len(before['cases'])==len(after['cases'])==200
        for a,b in zip(before['cases'],after['cases']):
            assert (a['index'],a['geometry_hash'],a['reynolds'])==(b['index'],b['geometry_hash'],b['reynolds'])
            assert a['reference_physics']==b['reference_physics']
            r=dict(model_id=row['id'],architecture=row['model'],seed=row['seed'],index=a['index'],
                   geometry_hash=a['geometry_hash'],reynolds=a['reynolds'])
            for key in ['all_cell_mse','fluid_velocity_relative_l2','fluid_pressure_relative_l2']:
                r['baseline_'+key]=a[key];r['extended_'+key]=b[key]
            cases.append(r)
        record=dict(id=row['id'],model=row['model'],seed=row['seed'],
            baseline_checkpoint_sha256=row['original_checkpoint_sha256'],extended_checkpoint_sha256=row['checkpoint_sha256'],
            baseline_gpu_5ms_pass=before['gpu_5ms_pass'],extended_gpu_5ms_pass=after['gpu_5ms_pass'],
            baseline_host_5ms_pass=before['host_instrumented_5ms_pass'],extended_host_5ms_pass=after['host_instrumented_5ms_pass'])
        for key in ['all_cell_mse','fluid_velocity_relative_l2','fluid_pressure_relative_l2']:
            a=before['means'][key];b=after['means'][key]
            record['baseline_'+key]=a;record['extended_'+key]=b;record[key+'_reduction_percent']=100*(1-b/a)
        models.append(record)
    assert len(models)==4 and len(cases)==800
    out=root/'comparison';out.mkdir(exist_ok=False)
    for name,rows in [('paired_case_metrics.csv',cases),('model_comparison.csv',models)]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(out/'complete.json',dict(status='FOUR_FIXED_SECOND_CHANCE_TESTS_COMPLETE',models=models,paired_TEST_cases=800,
        models_sha256=config['models_sha256'],no_training=True,no_TEST_selection=True,excluded_from_primary_experiment=True))
    text=['FINALNY SAMOSTATNY TEST DRUHEJ SANCE','',
        '4 modely, rovnake200 TEST pripady na kazdy model, 800 parov predikcii.',
        'Originalne vahy4h sa porovnavaju s vopred zmrazenymi predlzenymi vahami.',
        'Ziaden novy trening, vyber alebo zmena konfiguracie podla TEST.',
        'Vysledky su oddelene od povodnej55 modelovej tabulky.','',
        'Model                   rychlost L2 pred/po[%]    tlak L2 pred/po[%]   hostp99<=5ms']
    for r in models:
        text.append('%-23s %7.3f / %-7.3f        %7.3f / %-7.3f       %s'%(r['id'],
            100*r['baseline_fluid_velocity_relative_l2'],100*r['extended_fluid_velocity_relative_l2'],
            100*r['baseline_fluid_pressure_relative_l2'],100*r['extended_fluid_pressure_relative_l2'],r['extended_host_5ms_pass']))
    text+=['','Len dva explorativne seedy na rodinu; opisne porovnanie, nie dokaz',
           'statistickej vyznamnosti alebo optimalneho potrebneho casu. TEST uz',
           'bol znamy z povodnej studie. Surove polia a casovania ostavaju naPerune.']
    (out/'VYSLEDKY.txt').write_text('\n'.join(text)+'\n')
    print('FOUR_FIXED_SECOND_CHANCE_TESTS_COMPLETE',flush=True)


if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
