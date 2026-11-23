"""Independent audit of proposed N and GPU budgets."""
import csv,hashlib,json,math,sys,time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
from scipy import integrate,stats
out=Path(sys.argv[1]);source=Path(sys.argv[2]);start=time.monotonic()
designs=json.loads((out/"sample_size_design_summary.json").read_text())
rows=list(csv.DictReader((out/"sample_size_all_contrasts.csv").open()))
costrows=list(csv.DictReader((out/"observed_training_and_TEST_costs.csv").open()))
costs=json.loads((out/"observed_costs_by_method.json").read_text())
def mixture(n,d,sd,alpha):
    critical=stats.t.isf(alpha/2,n-1);lam=abs(d)*math.sqrt(n)/sd
    def f(u):
        bound=critical*math.sqrt(stats.chi2.ppf(u,n-1)/(n-1))
        return stats.norm.sf(bound-lam)+stats.norm.cdf(-bound-lam)
    return integrate.quad(f,0,1,epsabs=2e-10,epsrel=2e-9,limit=250)[0]
checked=[];worst_delta=0.
for item in designs:
    subset=[r for r in rows if r["design"]==item["design"] and float(r["relative_effect"])==item["relative_effect"]
            and float(r["target_power"])==item["target_power"] and r["sd_scenario"]==item["sd_scenario"]]
    assert len(subset)==item["comparisons"]
    ns=[int(r["required_training_pairs_n"]) for r in subset if r["required_training_pairs_n"]]
    assert len(ns)==len(subset) and max(ns)==item["maximum_required_n"]
    for limiting in item["limiting_contrasts"]:
        row=next(r for r in subset if all(r[k]==limiting[k] for k in ["method_a","method_b","metric"]))
        n=int(row["required_training_pairs_n"]);effect=float(row["absolute_target_difference"]);sd=float(row["assumed_planning_sd"])
        alpha=.05/item["comparisons"]
        at=mixture(n,effect,sd,alpha);before=mixture(n-1,effect,sd,alpha) if n>2 else None
        assert at>=item["target_power"]-4e-7,(item["design"],n,at,item["target_power"])
        assert before is None or before<item["target_power"]+4e-7
        worst_delta=max(worst_delta,abs(at-float(row["estimated_power_at_n"])))
        checked.append(dict(design=item["design"],relative_effect=item["relative_effect"],target_power=item["target_power"],
            sd_scenario=item["sd_scenario"],method_a=row["method_a"],method_b=row["method_b"],metric=row["metric"],
            n=n,independent_power_at_n=at,independent_power_at_n_minus_one=before))
    for reuse,mode in [(True,"extend_existing_to_n"),(False,"new_independent_cohort_n")]:
        repetitions=max(item["maximum_required_n"]-5,0) if reuse else item["maximum_required_n"]
        value=item[mode]
        assert value["new_models"]==repetitions*len(item["methods"])
        expected_train=0.;expected_test=0.;reserve=0.
        for m in item["methods"]:
            rr=[r for r in costrows if r["method"]==m];assert len(rr)==5
            expected_train+=math.fsum(float(r["observed_allocation_closeout_GPU_hours"]) for r in rr)/5
            expected_test+=math.fsum(float(r["test_model_elapsed_GPU_hours"]) for r in rr)/5
            reserve+=costs[m]["proposed_TEST_reserve_per_model_GPU_h"]
        assert math.isclose(value["expected_training_GPU_h"],repetitions*expected_train,rel_tol=2e-13,abs_tol=1e-10)
        assert math.isclose(value["total_expected_GPU_h"],repetitions*(expected_train+expected_test),rel_tol=2e-13,abs_tol=1e-10)
        assert value["training_hard_limit_GPU_h"]==4*value["new_models"]
        assert value["training_cap_plus_TEST_reserve_GPU_h"]==4*value["new_models"]+repetitions*reserve
for p in json.loads((out/"cost_source_manifest.json").read_text()):
    assert hashlib.sha256(Path(p["path"]).read_bytes()).hexdigest()==p["sha256"]
for name,item in json.loads((out/"OUTPUT_MANIFEST.json").read_text()).items():
    p=out/name
    assert p.stat().st_size==item["bytes"] and hashlib.sha256(p.read_bytes()).hexdigest()==item["sha256"]
result=dict(status="INDEPENDENT_POWER_AND_COST_AUDIT_PASS",design_scenarios=96,
    minimum_n_checks_by_independent_adaptive_integration=len(checked),
    maximum_power_discrepancy=worst_delta,all_budget_formulas_and_50_original_run_costs_verified=True,
    source_and_result_hashes_verified=True,GPU_used=False,
    completed_at=datetime.now(ZoneInfo("Europe/Bratislava")).isoformat(),elapsed_seconds=time.monotonic()-start)
(out/"independent_verification.json").write_text(json.dumps(result,indent=2)+"\n")
(out/"independent_minimum_N_checks.json").write_text(json.dumps(checked,indent=2)+"\n")
print(json.dumps(result),flush=True)
