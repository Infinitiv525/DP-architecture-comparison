"""Freeze the already inspected parents and original resource/algorithm sources."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def put(name,data):
    value=data if isinstance(data,bytes) else ((json.dumps(data,indent=2)+'\n').encode())
    path=HERE/name
    if path.exists():assert path.read_bytes()==value,'Preserve existing different source '+name
    else:path.write_bytes(value)


def main():
    base=HERE.parent/'architecture_budget_main_20261111_v1'
    old=json.loads((base/'policy.json').read_text());original=json.loads((base/'submission.json').read_text())
    inspect=json.loads((HERE/'inspect/submission.json').read_text())
    ready=(HERE/'report/inspection/ready.json').read_bytes()
    parents=json.loads(ready);assert parents['status']=='FOUR_LAST_CHECKPOINTS_AND_ADAM_STATES_VERIFIED'
    put('parents.json',ready)
    for name in ['budget.py','record_allocation.py']:
        put(name,(base/name).read_bytes().replace(b'\r\n',b'\n'))
    fno=json.loads((HERE.parent/'architecture_dense_fno_20261112_v1/submission.json').read_text())
    policy=dict(status='EXPLORATORY_SECOND_CHANCE_NOT_PART_OF_PRIMARY_EXPERIMENT',
        dependency_source=old['dependency_source'],dependency_sha256=old['dependency_sha256'],
        fixture_root=old['fixture_root'],fixture_ready_sha256=old['fixture_ready_sha256'],
        original_main_source=original['source'],original_main_sha256=original['sha256'],
        inspection_source=inspect['source'],inspection_source_sha256=inspect['sha256'],
        inspection_ready=inspect['run_root']+'/ready.json',inspection_ready_sha256=hashlib.sha256(ready).hexdigest(),
        parents_sha256=hashlib.sha256(ready).hexdigest(),
        additional_GPU_seconds_limit=14400,cumulative_nominal_GPU_seconds=28800,
        parent_count=4,families=['chebkan','rbf'],seeds=[931101,931102],
        training_GPU='NVIDIA H200',CPUs_per_task=4,host_RAM_GiB=40,
        precision='FP32, matmul and cuDNN TF32 disabled',
        maximum_additional_GPU_hours=16,additional_16h_extension_authorized_or_scheduled=False,
        train_cases=640,validation_cases=160,test_targets_allowed=False,
        validation_period_seconds=300,optimizer_reset=False,architecture_changed=False,
        learning_rate='Hold each parent Adam param-group LR exactly at its last saved value, close to1e-5; no restart.',
        data_order='Exact recovered randperm(seed+epoch) prefix, validated against640 visit counters.',
        RNG_note='Original checkpoint lacks RNG snapshots; frozen ChebKAN/RBF have no stochastic training layers. Independent per-epoch CPU generator recovers all future permutations.',
        optimizer_step_note='Keep stored Adam state and dtype unchanged; integer case/update counters are authoritative. Float step counters may lose unit precision at large counts; do not reset or alter them silently.',
        checkpoint_selection='Report phase best and phase last separately. Overall best may retain original4h baseline; never hide deterioration.',
        final_CPU_audit='Full640TRAIN+160VAL plus native physicalVAL for original best, continuation best and continuation last; noTEST.',
        excluded_from_primary_experiment=True,
        exploratory_selection_basis='Completed original TRAIN/VAL diagnostics; original study TEST already known, not a new blind experiment.',
        wait_for_FNO_audit_job=fno['jobs']['audit']['job_id'],
        FNO_result_root=fno['run_root'],FNO_audit_expected_status='DENSE_FNO5_TEST_AND_SAVED_OUTPUT_AUDIT_COMPLETE')
    put('policy.json',policy)
    print('Four audited parent states frozen; unchanged code, Adam and exact saved LR.')


if __name__=='__main__':main()
