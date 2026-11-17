"""Prepare an additive dense-FNO supplement without changing frozen cohorts."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'architecture_budget_main_20261111_v1'
FIELD = HERE.parent / 'architecture_fair_pilot_20261111_v2'
EVAL = HERE.parent / 'architecture_budget_test_20261112_v1'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(name, value):
    text = value if isinstance(value, str) else json.dumps(value, indent=2) + '\n'
    p = HERE / name
    if p.exists():
        assert p.read_text(encoding='utf-8') == text, 'Preserve existing source: ' + name
    else:
        p.write_text(text, encoding='utf-8', newline='\n')


def main():
    worker = (BASE / 'worker.py').read_text(encoding='utf-8')
    original_loop = worker[worker.index('def run(mode,index,root):'):]
    start, end = worker.index('def dependencies():'), worker.index('\n\ndef run(mode,index,root):')
    worker = worker[:start] + 'def dependencies():\n    from runtime import training_dependencies\n    return training_dependencies()\n' + worker[end:]
    worker = worker.replace("NAMES=['tfno','chebkan','pinet','dendrite','realnpu','rbf','elm','scn','cascor']", "NAMES=['fno']")
    worker = worker.replace("manifest=json.loads((HERE/'selection.json').read_text())", "manifest=json.loads((Path(root)/'selection.json').read_text())")
    expected_loop = original_loop.replace("manifest=json.loads((HERE/'selection.json').read_text())", "manifest=json.loads((Path(root)/'selection.json').read_text())")
    assert worker[worker.index('def run(mode,index,root):'):] == expected_loop
    write('worker.py', worker)
    for name in ['budget.py', 'record_allocation.py', 'PINN_resource_reference.json']:
        write(name, (BASE / name).read_text(encoding='utf-8'))
    policy = read(BASE / 'policy.json')
    policy.update(status='ADDITIVE_DENSE_FNO_EQUAL_RESOURCE_PROTOCOL', main_repetitions=5,
                  main_GPU_hard_limit_hours=20, uniform_tuning_GPU_hard_limit_hours=2,
                  tuning_candidates_per_family=2, final_training_auto_launch_allowed=True,
                  initial_model_core_capacity='Same modes, depth, width and I/O as TFNO; dense Fourier tensors, actual parameter count reported',
                  supplement_added_after_original_test_results=True,
                  independently_blind_test=False, final_main_evaluation_period_seconds=300,
                  independent_full_test_GPU_reserve_hours=10.25,
                  per_test_group_seconds_limit=7200,
                  full_extension_GPU_hard_limit_hours=32.30,
                  final_test_models=5, final_test_cases_per_model=200)
    for key in ['projected_upper_GPU_hours_before_smoke', 'remaining_margin_GPU_hours_before_smoke',
                'total_upper_GPU_hours_including_two_control_batches_and_initial_benchmark', 'budget_origin_notes',
                'prior_calibration_GPU_hard_limit_hours', 'old8hour_plan_superseded_without_launch']:
        policy.pop(key, None)
    policy['extension_scope'] = 'Dense unfactorized FNO supplement; FlowBench authors themselves instantiate TFNO. Not a missing-author-baseline or exact-paper replication.'
    policy['reference_main_source'] = read(BASE / 'submission.json')['source']
    policy['reference_main_sha256'] = read(BASE / 'submission.json')['sha256']
    write('policy.json', policy)
    write('recipe.json', dict(n_modes=[32,32], n_layers=5, hidden_channels=48,
          in_channels=665, out_channels=192, projection_channel_ratio=4,
          lifting_channel_ratio=2, factorization=None, rank=1.0,
          implementation='factorized', positional_embedding='grid', fno_block_precision='full',
          note='Only spectral weight factorization differs from the completed TFNO backbone. implementation=factorized is the library contraction path; factorization=None creates dense tensors.'))
    write('implementation_provenance.json', dict(
        author_repository='https://github.com/baskargroup/flowbench-tools',
        author_commit='6877099b8883232138c73b2b6056ff01f7d3697b',
        author_model_path='Training/codes/models/FNO.py', author_model_git_blob='2ff7f9c4e43d075483d57005172b24154a2ed4ba',
        author_model_url='https://github.com/baskargroup/flowbench-tools/blob/6877099b8883232138c73b2b6056ff01f7d3697b/Training/codes/models/FNO.py',
        author_actual_implementation='TensorizedFNO wrapper instantiates neuralop.models.TFNO; not a separate dense FNO.',
        correction='Earlier chat claim that FlowBench authors use a separate classical FNO was incorrect.',
        implementation='neuralop.models.FNO, neuraloperator==2.0.0, factorization=None, rank=1.0',
        documentation='https://neuraloperator.github.io/dev/modules/generated/neuralop.models.FNO.html',
        backbone_comparator='Completed TFNO: modes32x32, five layers, width48, Tucker rank0.1',
        shared_input_output_adapter=True, unchanged_native_loss=True,
        original_worker_sha256=hashlib.sha256((BASE/'worker.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        original_learning_loop_sha256=hashlib.sha256(original_loop.encode()).hexdigest(),
        learning_loop_change='Only selection manifest location changes to immutable runtime-generated run_root/selection.json; optimization loop is identical.',
        changes=['one family name fno','dense FNO constructor and loader','TRAIN/VAL-only two-candidate selection at run time'],
        all_previous_checkpoints_and_results_unchanged=True))
    config = read(EVAL / 'config.json')
    config.update(name=HERE.name, model_count=5, models_per_group=1,
                  test_group_count=5, test_group_GPU_time_limit_seconds=7200,
                  preflight_GPU_time_limit_seconds=900,
                  supplement_added_after_original_test_results=True,
                  baseline_original50_result=read(EVAL/'submission.json')['run_root'])
    config.pop('models_sha256', None)
    write('evaluation_template.json', config)
    common = (EVAL/'common.py').read_text(encoding='utf-8')
    common = common.replace('import json\n', 'import json\nimport os\n')
    common = common.replace('SOURCE=Path(__file__).resolve().parent', "CODE_SOURCE=Path(__file__).resolve().parent\nSOURCE=Path(os.environ['DENSE_FNO_RUN_ROOT'])/'frozen_evaluation'")
    common = common.replace("RESOURCE_MATCHED_FINAL50_MANIFEST_FROZEN", "DENSE_FNO5_MANIFEST_FROZEN")
    common = common.replace("len(manifest['models'])==50", "len(manifest['models'])==5")
    common = common.replace('legacy_paths(config);return manifest', 'legacy_paths(config)\n    from runtime import install_fno_models\n    install_fno_models()\n    return manifest')
    write('eval_common.py', common)
    evaluate = (EVAL/'evaluate.py').read_text(encoding='utf-8')
    evaluate = evaluate.replace('from common import SOURCE', 'from eval_common import SOURCE')
    evaluate = evaluate.replace('FINAL50_', 'DENSE_FNO5_')
    evaluate = evaluate.replace('assert len(records)==10', 'assert len(records)==1')
    evaluate = evaluate.replace('total_models=10,total_case_predictions=2000', 'total_models=1,total_case_predictions=200')
    evaluate = evaluate.replace('for50 frozen models', 'for five frozen dense FNO models')
    write('evaluate.py', evaluate)
    audit = (EVAL/'audit.py').read_text(encoding='utf-8')
    audit = audit.replace('from common import SOURCE', 'from eval_common import SOURCE')
    audit = audit.replace('FINAL50_', 'DENSE_FNO5_')
    audit = audit.replace('len(ids)==len(set(ids))==50', 'len(ids)==len(set(ids))==5')
    audit = audit.replace('total_models=50,total_case_predictions=10000', 'total_models=5,total_case_predictions=1000')
    audit = audit.replace('len(rows)==10000 and len(models)==50', 'len(rows)==1000 and len(models)==5')
    audit = audit.replace('saved_native_fields_checked=10000,raw_timing_pairs_checked=10000000', 'saved_native_fields_checked=1000,raw_timing_pairs_checked=1000000')
    audit = audit.replace('50 modelov x200 TEST simulacii =10000 overenych predikovanych poli.', '5 novych dense FNO modelov x200 TEST simulacii =1000 overenych poli.')
    audit = audit.replace('Primarne9rodin ma rovnake plne vstupy,superviziu a zdrojovyrozpocet.', 'Dense FNO dodatok: rovnake plne vstupy,supervizia a zdrojovyrozpocet ako povodnych9rodin.')
    audit = audit.replace('HistorickaPINN je oddelena pomocna referencia s vlastnym postupom.', 'Autori FlowBench vo svojom kode pouzivaju TFNO; toto je dodatocny dense FNO variant.')
    write('audit.py', audit)
    print('Prepared immutable source copies; training loop unchanged apart from manifest location.')


if __name__ == '__main__':
    main()
