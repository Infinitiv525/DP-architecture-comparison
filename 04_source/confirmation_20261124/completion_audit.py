"""Validate all60 final TRAIN/VAL records and freeze best checkpoints, CPU only."""
import hashlib,json,math,re,sys,time
from pathlib import Path
SOURCE=Path(__file__).resolve().parent
RUN=Path("/mnt/project/perun2601449/dp_llm_cfd/results/architecture_confirmation_20261124_v2")
def read(p):return json.loads(Path(p).read_text())
def digest(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def all_finite(value):
    if isinstance(value,dict):return all(all_finite(v) for v in value.values())
    if isinstance(value,list):return all(all_finite(v) for v in value)
    if isinstance(value,(float,int)):return math.isfinite(value)
    return True
plan=read(SOURCE/"plan.json");tasks=read(SOURCE/"selection.json")["tasks"][:60]
records=[];failures=[];allocation_seconds=0
for row in tasks:
    try:
        folder=RUN/"main"/row["id"];value=read(folder/"complete.json")
        close=read(RUN/"allocation_closeout"/("main_"+str(row["index"])+".json"))
        assert close["worker_exit_code"]==0
        assert value["status"]=="BUDGETED_TRAINING_COMPLETE" and value["test_targets_read"] is False
        assert value["task"]["model"]==row["model"] and value["task"]["seed"]==row["seed"]
        assert value["task"]["configuration_index"]==row["configuration_index"]
        assert value["task"]["time_budget_seconds"]==14400 and value["within_time_limit"]
        assert len(value["complete_native_TRAIN_case_visits"])==640 and min(value["complete_native_TRAIN_case_visits"])>0
        assert max(value["complete_native_TRAIN_case_visits"])-min(value["complete_native_TRAIN_case_visits"])<=1
        assert len(value["physical_validation_metrics"]["case_metrics"])==160
        assert all_finite(value["physical_validation_metrics"])
        assert value["selected_checkpoint_sha256"]==digest(folder/"best.pt")
        runtime=re.search(r"\bRunTime=(\d\d):(\d\d):(\d\d)",close["scheduler_stdout"])
        assert runtime
        seconds=int(runtime[1])*3600+int(runtime[2])*60+int(runtime[3])
        assert seconds<=14400
        allocation_seconds+=seconds
        records.append(dict(id=row["id"],model=row["model"],seed=row["seed"],seed_block=row["seed_block"],
            role="new_independent20_seed_confirmation",checkpoint=str(folder/"best.pt"),
            checkpoint_sha256=value["selected_checkpoint_sha256"],training_complete=str(folder/"complete.json"),
            training_complete_sha256=digest(folder/"complete.json"),configuration_index=row["configuration_index"],
            selected_parameter_ledger=value["selected_parameter_ledger"],
            original_training_source=str(SOURCE/("dense" if row["model"]=="fno" else "primary")),
            training_case_visits_min=min(value["complete_native_TRAIN_case_visits"]),
            training_case_visits_max=max(value["complete_native_TRAIN_case_visits"]),
            allocation_closeout_GPU_hours=seconds/3600,best_VAL_loss=value["best_complete_validation_loss"]))
    except Exception as error:
        failures.append(dict(id=row["id"],index=row["index"],error=repr(error)))
status="CONFIRMATION60_TRAINING_AND_VAL_AUDIT_PASS" if not failures and len(records)==60 else "CONFIRMATION60_INCOMPLETE_OR_FAILED"
result=dict(status=status,complete_models=len(records),expected_models=60,failures=failures,
    allocation_closeout_GPU_hours=allocation_seconds/3600,final_portal_billing=False,
    test_targets_read=False,all_seeds_retained=True,pilot_models_used=False,
    source_integrity_sha256=digest(SOURCE/"source_integrity.json"))
(RUN/"training_audit.json").write_text(json.dumps(result,indent=2)+"\n")
if status=="CONFIRMATION60_TRAINING_AND_VAL_AUDIT_PASS":
    frozen=dict(status="CONFIRMATION60_CHECKPOINTS_FROZEN_BEFORE_NEW_TEST",models=records,
        source_integrity_sha256=result["source_integrity_sha256"],primary_seed_blocks=20,
        cases_per_split=plan["split"],test_evaluation_not_started=True)
    assert not (RUN/"frozen_models.json").exists()
    (RUN/"frozen_models.json").write_text(json.dumps(frozen,indent=2)+"\n")
print(json.dumps(result),flush=True)
sys.exit(0 if status=="CONFIRMATION60_TRAINING_AND_VAL_AUDIT_PASS" else 2)
