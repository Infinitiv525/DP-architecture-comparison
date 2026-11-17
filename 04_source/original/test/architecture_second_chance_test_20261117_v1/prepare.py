"""Freeze four VAL-selected continuations before a single supplemental TEST."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent


def read(p):return json.loads(p.read_text())


def put(name,value):
    text=value if isinstance(value,str) else json.dumps(value,indent=2)+'\n'
    p=HERE/name
    if p.exists():assert p.read_text()==text,'Preserve existing source '+name
    else:p.write_text(text,encoding='utf-8',newline='\n')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    second=BASE/'architecture_second_chance_20261112_v1';dense=BASE/'architecture_dense_fno_20261112_v1'
    receipt=read(second/'submission.json');proof=read(second/'completion_check.json')
    assert proof['status']=='ALL_FOUR_SECOND_CHANCE_LINEAGES_VERIFIED'
    summary=read(second/'report/continuation/summary/complete.json')
    original=read(BASE/'architecture_budget_test_20261112_v1/models.json')
    config=read(dense/'report/frozen_evaluation/config.json')
    config.update(name=HERE.name,main_source=receipt['source'],main_source_sha256=receipt['sha256'],
        model_count=4,models_per_group=1,test_group_count=4,
        preflight_GPU_time_limit_seconds=900,test_group_GPU_time_limit_seconds=7200,
        total_GPU_allocation_hours_limit=8.25,
        notes='Read-only four-model supplemental TEST. Frozen after completed TRAIN/VAL audits. Separate from original55.',
        original_baseline_result_root=read(BASE/'architecture_budget_test_20261112_v1/submission.json')['run_root'],
        excluded_from_primary_experiment=True)
    indices=read(BASE/'architecture_dense_fno_20261112_v1/report/frozen_evaluation/models.json')['models'][0]['expected_validation']
    ids=[r['index'] for r in indices]
    rows=[]
    for i,item in enumerate(summary['models']):
        done_path=second/'report/continuation/continuation'/item['id']/'complete.json'
        audit_path=second/'report/continuation/audit'/item['id']/'complete.json'
        done=read(done_path);audit=read(audit_path);selected=audit['metrics'][audit['selected_best_so_far']]
        assert not done['selected_is_original_baseline'],'Require explicitly reviewed continuation selection'
        assert selected['checkpoint_sha256']==done['selected_checkpoint_sha256']
        old=next(r for r in original['models'] if r['model']==item['architecture'] and r['seed']==item['seed'])
        expected=[dict(index=index,velocity_l2=selected['physical_validation']['case_metrics'][j]['velocity_relative_L2'],
            pressure_l2=selected['physical_validation']['case_metrics'][j]['pressure_relative_L2']) for j,index in enumerate(ids)]
        row=dict(id=item['id'],model=item['architecture'],seed=item['seed'],group=i,
            kind='fixed_exploratory_second_chance_TEST',checkpoint=done['selected_checkpoint'],
            checkpoint_sha256=done['selected_checkpoint_sha256'],expected_validation=expected,
            parameter_ledger=old['parameter_ledger'],original_baseline_id=old['id'],
            original_checkpoint=old['checkpoint'],original_checkpoint_sha256=old['checkpoint_sha256'],
            source_continuation_summary=receipt['run_root']+'/continuation/'+item['id']+'/complete.json',
            source_continuation_summary_sha256=sha(done_path),
            selection='Best saved continuation checkpoint on VAL, frozen before this TEST',
            excluded_from_primary_experiment=True)
        rows.append(row)
        config['metadata_sha256'][row['source_continuation_summary']]=sha(done_path)
        config['metadata_sha256'][receipt['run_root']+'/audit/'+item['id']+'/complete.json']=sha(audit_path)
    assert len(rows)==4
    manifest=dict(status='SECOND_CHANCE4_FIXED_TEST_MANIFEST',models=rows,selection_uses_test=False,
        prior_original_TEST_results_known=True,no_further_training_or_selection_from_this_TEST=True,
        excluded_from_primary_experiment=True)
    put('models.json',manifest);config['models_sha256']=sha(HERE/'models.json');put('config.json',config)
    common=(dense/'eval_common.py').read_text()
    common=common.replace("SOURCE=Path(os.environ['DENSE_FNO_RUN_ROOT'])/'frozen_evaluation'",'SOURCE=CODE_SOURCE')
    common=common.replace("DENSE_FNO5_MANIFEST_FROZEN",'SECOND_CHANCE4_FIXED_TEST_MANIFEST')
    common=common.replace("len(manifest['models'])==5","len(manifest['models'])==4")
    common=common.replace('    from runtime import install_fno_models\n    install_fno_models()\n','')
    put('eval_common.py',common)
    evaluate=(dense/'evaluate.py').read_text().replace('DENSE_FNO5_','SECOND_CHANCE4_')
    evaluate=evaluate.replace("assert cp['task']['mode']=='main' and cp['task']['time_budget_seconds']==14400", "assert cp['task']['mode']=='exploratory_continuation' and cp['task']['additional_GPU_seconds_limit']==14400\n            assert cp['task']['id']==row['id'] and cp['excluded_from_primary_experiment']")
    evaluate=evaluate.replace('choices=range(5)','choices=range(4)')
    put('evaluate.py',evaluate)
    audit=(dense/'audit.py').read_text().replace('DENSE_FNO5_','SECOND_CHANCE4_')
    audit=audit.replace('range(5)','range(4)').replace('len(ids)==len(set(ids))==5','len(ids)==len(set(ids))==4')
    audit=audit.replace('total_models=5,total_case_predictions=1000','total_models=4,total_case_predictions=800')
    audit=audit.replace('len(rows)==1000 and len(models)==5','len(rows)==800 and len(models)==4')
    audit=audit.replace('saved_native_fields_checked=1000,raw_timing_pairs_checked=1000000','saved_native_fields_checked=800,raw_timing_pairs_checked=800000')
    audit=audit.replace("assert len(values)==5","assert len(values)==2").replace('%8d/5','%8d/2')
    audit=audit.replace('5 novych dense FNO modelov x200 TEST simulacii =1000 overenych poli.','4 zmrazene modely druhej sance x200 TEST pripadov =800 overenych poli.')
    audit=audit.replace('Zachovane vsetkych5 opakovani kazdej metody; ziadny vyber podlaTEST.','Dva vopred urcene seedy na rodinu; ziadny vyber podla tohto TEST.')
    audit=audit.replace('Dense FNO dodatok: rovnake plne vstupy,supervizia a zdrojovyrozpocet ako povodnych9rodin.','Samostatny explorativny TEST druhej sance, nemeni povodnu55 modelovu tabulku.')
    audit=audit.replace('Autori FlowBench vo svojom kode pouzivaju TFNO; toto je dodatocny dense FNO variant.','Vahy boli zmrazene podlaVAL este pred otvorenim tohto TEST.')
    put('audit.py',audit)
    print('Four frozen checkpoints, 800 TEST fields, 800000 timing pairs; unchanged metrics.')


if __name__=='__main__':main()
