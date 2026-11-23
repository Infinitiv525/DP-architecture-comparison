"""Prospective sample-size planning from archived pilot variability. CPU only."""
import argparse,csv,hashlib,itertools,json,math,re,sys,time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import scipy
from scipy import integrate,stats

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def csvfile(path,rows):
    with path.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
_quad_nodes,_quad_weights=np.polynomial.legendre.leggauss(128)
_quad_u=(_quad_nodes+1)/2
_quad_weights=_quad_weights/2
_bound_cache={}
_power_fallbacks=0
def power(n,effect,sd,alpha):
    global _power_fallbacks
    if sd<=0:return 1.0 if effect!=0 else alpha
    critical=stats.t.isf(alpha/2,n-1)
    lam=abs(effect)*math.sqrt(n)/sd
    if n<2000:
        value=float(stats.nct.sf(critical,n-1,lam)+stats.nct.cdf(-critical,n-1,lam))
        if np.isfinite(value):return min(1.,max(0.,value))
    # SciPy1.18.1 can return NaN at high df. Compute the defining
    # normal/chi-square mixture via fixed Gaussian quadrature instead.
    _power_fallbacks+=1
    key=(n,alpha)
    if key not in _bound_cache:
        _bound_cache[key]=critical*np.sqrt(stats.chi2.ppf(_quad_u,n-1)/(n-1))
    bounds=_bound_cache[key]
    value=float(np.sum(_quad_weights*(stats.norm.sf(bounds-lam)+stats.norm.cdf(-bounds-lam))))
    assert np.isfinite(value)
    return min(1.,max(0.,value))
def required(effect,sd,alpha,target,maximum):
    if power(maximum,effect,sd,alpha)<target:return None
    lo,hi=2,maximum
    while lo<hi:
        mid=(lo+hi)//2
        if power(mid,effect,sd,alpha)>=target:hi=mid
        else:lo=mid+1
    assert power(lo,effect,sd,alpha)>=target-1e-12
    assert lo==2 or power(lo-1,effect,sd,alpha)<target+1e-12
    return lo
def runtime_seconds(text):
    match=re.search(r"\bRunTime=(\S+)",text)
    assert match
    value=match.group(1);days=0
    if "-" in value:day,value=value.split("-");days=int(day)
    h,m,s=map(int,value.split(":"))
    return days*86400+h*3600+m*60+s
def collect_costs(root,methods):
    rows=[];proof=[]
    for kind,rel in [("primary","results/architecture_budget_main_20261111_v1/run_3e11271df45aff09"),
                     ("dense","results/architecture_dense_fno_20261112_v1/run_c93033883d5f56d4")]:
        run=root/rel
        closeouts={}
        for path in (run/"allocation_closeout").glob("*.json"):
            value=json.loads(path.read_text())
            if value.get("mode")=="main":closeouts[str(value["job_id"])]=(path,value)
        selected=["fno"] if kind=="dense" else methods[:9]
        for method in selected:
            for seed in range(931101,931106):
                model_id=f"{method}_s{seed}"
                complete=run/"main"/model_id/"complete.json"
                start=run/"main"/model_id/"started.json"
                c=json.loads(complete.read_text());s=json.loads(start.read_text())
                assert c["status"]=="BUDGETED_TRAINING_COMPLETE"
                assert c["task"]["mode"]=="main" and c["task"]["model"]==method and c["task"]["seed"]==seed
                assert c["task"]["time_budget_seconds"]==14400
                closepath,close=closeouts[str(s["allocation"]["job_id"])]
                assert close["worker_exit_code"]==0 and close["allocated_GPUs"]==1 and close["allocated_CPUs"]==4
                seconds=runtime_seconds(close["scheduler_stdout"])
                assert seconds>=c["allocation_elapsed_seconds"]-3 and seconds<=14400
                testroot=run if kind=="dense" else root/"results/architecture_budget_test_20261112_v1/run_30896f9aadca67bb"
                testfile=testroot/"test/models"/model_id/"summary.json"
                test=json.loads(testfile.read_text())
                assert test["test_cases"]==200 and test["environment_before"]["gpu"]=="NVIDIA H200"
                rows.append(dict(method=method,seed=seed,training_worker_hours=c["allocation_elapsed_seconds"]/3600,
                    observed_allocation_closeout_GPU_hours=seconds/3600,test_model_elapsed_GPU_hours=test["elapsed_seconds"]/3600,
                    closeout_is_final_portal_billing=False))
                for path in [complete,start,closepath,testfile]:
                    proof.append(dict(path=str(path),sha256=digest(path)))
    assert len(rows)==50
    aggregates={}
    for method in methods:
        group=[r for r in rows if r["method"]==method]
        assert len(group)==5
        allocations=[r["observed_allocation_closeout_GPU_hours"] for r in group]
        tests=[r["test_model_elapsed_GPU_hours"] for r in group]
        aggregates[method]=dict(mean_training_GPU_h=float(np.mean(allocations)),
            minimum_training_GPU_h=min(allocations),maximum_training_GPU_h=max(allocations),
            mean_TEST_GPU_h=float(np.mean(tests)),maximum_TEST_GPU_h=max(tests),
            training_hard_limit_GPU_h=4.,
            proposed_TEST_reserve_per_model_GPU_h=math.ceil((1.25*max(tests)+.05)*4)/4)
    return rows,aggregates,proof
def costs(methods,n,aggregate,reuse_pilot):
    repeats=max(0,n-5) if reuse_pilot else n
    return dict(new_models=repeats*len(methods),
        expected_training_GPU_h=repeats*sum(aggregate[m]["mean_training_GPU_h"] for m in methods),
        training_hard_limit_GPU_h=4*repeats*len(methods),
        expected_TEST_GPU_h=repeats*sum(aggregate[m]["mean_TEST_GPU_h"] for m in methods),
        TEST_planning_reserve_GPU_h=repeats*sum(aggregate[m]["proposed_TEST_reserve_per_model_GPU_h"] for m in methods),
        total_expected_GPU_h=repeats*sum(aggregate[m]["mean_training_GPU_h"]+aggregate[m]["mean_TEST_GPU_h"] for m in methods),
        training_cap_plus_TEST_reserve_GPU_h=repeats*sum(4+aggregate[m]["proposed_TEST_reserve_per_model_GPU_h"] for m in methods))
def anova_components(matrix,w):
    S,G=matrix.shape
    mu=matrix.mean();row=matrix.mean(axis=1);col=matrix.mean(axis=0)
    residual=matrix-row[:,None]-col[None,:]+mu
    ms_e=float(np.sum(residual**2)/((S-1)*(G-1)))
    ms_s=float(G*np.sum((row-mu)**2)/(S-1))
    ms_g=float(S*np.sum((col-mu)**2)/(G-1))
    raw_s=(ms_s-ms_e)/G;raw_g=(ms_g-ms_e)/S
    seed=max(raw_s,0);geo=max(raw_g,0);interaction=ms_e
    w2=float(np.sum(w*w))
    return dict(seed_variance_unclipped=raw_s,geometry_variance_unclipped=raw_g,
        seed_variance=seed,geometry_variance=geo,interaction_variance=interaction,
        case_weight_square_sum=w2,
        seed_reducible_variance_numerator=seed+interaction*w2,
        geometry_variance_floor=geo*w2,
        model_SE_at_five=math.sqrt(geo*w2+(seed+interaction*w2)/5),
        infinite_seed_SE_floor=math.sqrt(geo*w2))
def geometry_power(n,effect,components,alpha):
    variance=components["geometry_variance_floor"]+components["seed_reducible_variance_numerator"]/n
    if variance==0:return 1.
    lam=abs(effect)/math.sqrt(variance);z=stats.norm.isf(alpha/2)
    return float(stats.norm.sf(z-lam)+stats.norm.cdf(-z-lam))
def run(config_path,out):
    started=time.monotonic()
    c=json.loads(config_path.read_text());root=Path(c["root"]);previous=root/c["previous_statistics"]
    original_manifest=json.loads((previous/"OUTPUT_MANIFEST.json").read_text())
    path=root/c["input_npz"]
    assert digest(path)==original_manifest["files"]["analysis_inputs.npz"]["sha256"]
    bundle=np.load(path,allow_pickle=False)
    methods=bundle["method_names"].tolist()[:10];metrics=bundle["metric_names"].tolist()
    assert methods[-1]=="fno" and len(methods)==10
    cube=bundle["errors"][:10];means=cube.mean(axis=(1,2));modelmeans=cube.mean(axis=2)
    clusters=bundle["case_geometric_cluster"];groups=sorted(set(clusters.tolist()))
    weights=np.array([(clusters==g).sum()/200 for g in groups])
    grouped=np.stack([cube[:,:,clusters==g,:].mean(axis=2) for g in groups],axis=2)
    assert grouped.shape==(10,5,85,3)
    assert not out.exists()
    out.mkdir(parents=True)
    costrows,costsummary,proof=collect_costs(root,methods)
    save(out/"cost_source_manifest.json",proof)
    save(out/"observed_costs_by_method.json",costsummary)
    csvfile(out/"observed_training_and_TEST_costs.csv",costrows)
    factor=math.sqrt(4/stats.chi2.ppf(1-c["planning_sd_upper_confidence"],4))
    designs=[];individual=[];floor_rows=[];baseline_scale=[]
    pair_components={}
    old_boot=np.load(previous/"bootstrap_replicates.npz",allow_pickle=False)["paired_crossed_means"]
    for a,b in itertools.combinations(range(10),2):
        for k,metric in enumerate(metrics):
            components=anova_components(grouped[a,:,:,k]-grouped[b,:,:,k],weights)
            components["original_crossed_bootstrap_SE"]=float(np.std(old_boot[:,a,k]-old_boot[:,b,k],ddof=1))
            pair_components[(a,b,k)]=components
    for design in c["comparison_designs"]:
        indices=[methods.index(m) for m in design["methods"]]
        pairs=list(itertools.combinations(indices,2))
        if not design["all_pairs"]:
            pairs=[(a,b) for a,b in pairs if methods[a]=="tfno" or methods[b]=="tfno"]
        metric_indices=design.get("metric_indices",[0,1,2])
        assert len(pairs)*len(metric_indices)==design["family_size"]
        alpha=c["family_alpha"]/design["family_size"]
        exact_floor=math.ceil(math.log2(2/alpha))
        for rel,target,sd_mode in itertools.product(c["relative_effects"],c["target_powers"],["point_pilot_sd","upper95_pilot_sd"]):
            results=[]
            for a,b in pairs:
                for k,metric in enumerate(metrics):
                    if k not in metric_indices:continue
                    pilot_sd=float(np.std(modelmeans[a,:,k]-modelmeans[b,:,k],ddof=1))
                    independent_sd=float(math.sqrt(np.var(modelmeans[a,:,k],ddof=1)+np.var(modelmeans[b,:,k],ddof=1)))
                    sd=pilot_sd*(factor if sd_mode=="upper95_pilot_sd" else 1)
                    delta=rel*float((means[a,k]+means[b,k])/2)
                    n=required(delta,sd,alpha,target,c["maximum_n"])
                    results.append(n)
                    individual.append(dict(design=design["key"],method_a=methods[a],method_b=methods[b],metric=metric,
                        relative_effect=rel,effect_scale="fraction_of_pair_midpoint",absolute_target_difference=delta,
                        target_power=target,sd_scenario=sd_mode,pilot_paired_sd=pilot_sd,
                        point_unpaired_sd_sensitivity=independent_sd,assumed_planning_sd=sd,
                        required_training_pairs_n=n,
                        estimated_power_at_n=None if n is None else power(n,delta,sd,alpha),
                        exact_sign_flip_resolution_minimum_n=exact_floor,
                        conditional_on_same_200_TEST_cases=True))
            worst=None if None in results else max(results)
            group=[r for r in individual if r["design"]==design["key"] and r["relative_effect"]==rel
                   and r["target_power"]==target and r["sd_scenario"]==sd_mode]
            if worst is None:
                worst_rows=[r for r in group if r["required_training_pairs_n"] is None]
            else:
                worst_rows=[r for r in group if r["required_training_pairs_n"]==worst]
            item=dict(design=design["key"],methods=design["methods"],comparisons=design["family_size"],
                relative_effect=rel,target_power=target,sd_scenario=sd_mode,maximum_required_n=worst,
                exact_sign_flip_resolution_minimum_n=exact_floor,
                limiting_contrasts=[{k:r[k] for k in ["method_a","method_b","metric","pilot_paired_sd"]} for r in worst_rows])
            if worst is not None:
                item["extend_existing_to_n"]=costs(design["methods"],worst,costsummary,True)
                item["new_independent_cohort_n"]=costs(design["methods"],worst,costsummary,False)
            designs.append(item)
        # Geometry uncertainty has its own power ceiling; not fixed by N seeds.
        for a,b in pairs:
            for k,metric in enumerate(metrics):
                if k not in metric_indices:continue
                comp=pair_components[(a,b,k)]
                for rel,target in itertools.product(c["relative_effects"],c["target_powers"]):
                    effect=rel*float((means[a,k]+means[b,k])/2)
                    upper=geometry_power(1e18,effect,comp,alpha)
                    n=None
                    if upper>=target:
                        lo,hi=2,10000000
                        if geometry_power(hi,effect,comp,alpha)>=target:
                            while lo<hi:
                                mid=(lo+hi)//2
                                if geometry_power(mid,effect,comp,alpha)>=target:hi=mid
                                else:lo=mid+1
                            n=lo
                    floor_rows.append(dict(design=design["key"],method_a=methods[a],method_b=methods[b],metric=metric,
                        relative_effect=rel,target_power=target,maximum_model_power_with_infinite_seeds=upper,
                        target_unreachable_by_more_seeds_alone=bool(upper<target),
                        approximate_required_n_with_geometry_model=n,**comp))
        # Alternate, explicitly baseline-anchored effect definition.
        for a,b in pairs:
            if "tfno" not in [methods[a],methods[b]]:continue
            for k,metric in enumerate(metrics):
                if k not in metric_indices:continue
                psd=float(np.std(modelmeans[a,:,k]-modelmeans[b,:,k],ddof=1))
                for rel,target in itertools.product(c["relative_effects"],c["target_powers"]):
                    delta=rel*float(means[methods.index("tfno"),k])
                    baseline_scale.append(dict(design=design["key"],method_a=methods[a],method_b=methods[b],metric=metric,
                        relative_effect_of_TFNO=rel,target_power=target,
                        required_n_point_sd=required(delta,psd,alpha,target,c["maximum_n"]),
                        required_n_upper95_sd=required(delta,psd*factor,alpha,target,c["maximum_n"])))

    csvfile(out/"sample_size_all_contrasts.csv",individual)
    csvfile(out/"geometry_uncertainty_limits.csv",floor_rows)
    csvfile(out/"TFNO_anchored_effect_sensitivity.csv",baseline_scale)
    save(out/"sample_size_design_summary.json",designs)
    flat=[]
    for r in designs:
        q={k:v for k,v in r.items() if k not in ["methods","limiting_contrasts","extend_existing_to_n","new_independent_cohort_n"]}
        q["limiting_contrasts"]=json.dumps(r["limiting_contrasts"])
        if r["maximum_required_n"] is not None:
            for mode in ["extend_existing_to_n","new_independent_cohort_n"]:
                q.update({mode+"_"+k:v for k,v in r[mode].items()})
        flat.append(q)
    csvfile(out/"sample_size_design_summary.csv",flat)
    # Uniform, easy-to-price choices; these are options, not auto-launches.
    alternatives=[]
    for design in c["comparison_designs"]:
        for n in [10,13,15,20,25,30,40,50]:
            for reuse in [True,False]:
                alternatives.append(dict(design=design["key"],training_pairs_n=n,pilot_reused=reuse,
                    **costs(design["methods"],n,costsummary,reuse)))
    csvfile(out/"cost_options.csv",alternatives)
    save(out/"variance_components.json",[dict(method_a=methods[a],method_b=methods[b],metric=metrics[k],**v)
        for (a,b,k),v in pair_components.items()])
    # Numerical integration is independent of scipy.stats.nct's CDF path.
    numerical=[]
    for n,d,m in [(15,.8,108),(30,1.,135),(15,2.,9),(5000,.07113,108),(100000,.01,135)]:
        alpha=.05/m;critical=stats.t.isf(alpha/2,n-1);lam=d*math.sqrt(n)
        def integrand(u):
            bound=critical*math.sqrt(u/(n-1))
            return (stats.norm.sf(bound-lam)+stats.norm.cdf(-bound-lam))*stats.chi2.pdf(u,n-1)
        if n<2000:
            value,error=integrate.quad(integrand,0,np.inf,epsabs=1e-10,epsrel=1e-9)
        else:
            def uniform_integrand(u):
                bound=critical*math.sqrt(stats.chi2.ppf(u,n-1)/(n-1))
                return stats.norm.sf(bound-lam)+stats.norm.cdf(-bound-lam)
            value,error=integrate.quad(uniform_integrand,0,1,epsabs=1e-11,epsrel=1e-10)
        expected=power(n,d,1.,alpha)
        assert abs(value-expected)<3e-7,(n,value,expected)
        assert abs(power(n,0,1,alpha)-alpha)<2e-8
        numerical.append(dict(n=n,standardized_effect=d,family_size=m,noncentral_t_power=expected,
            independently_integrated_power=value,quadrature_error=error))
    rng=np.random.default_rng(c["random_seed"])
    mc=[];M=c["monte_carlo_replicates"]
    for n,d,m,shape in [(15,.8,108,"normal"),(30,1.,135,"normal"),(15,2.,9,"normal"),
                       (15,0,135,"normal"),(15,0,135,"student_t3"),
                       (15,0,135,"centered_exponential"),(30,1.,135,"student_t3")]:
        alpha=.05/m;critical=stats.t.isf(alpha/2,n-1);rejections=0
        for first in range(0,M,2000):
            size=min(2000,M-first)
            if shape=="normal":samples=rng.normal(size=(size,n))
            elif shape=="student_t3":samples=rng.standard_t(3,size=(size,n))/math.sqrt(3)
            else:samples=rng.exponential(size=(size,n))-1
            samples+=d
            statistic=samples.mean(axis=1)/(samples.std(axis=1,ddof=1)/math.sqrt(n))
            rejections+=int(np.count_nonzero(np.abs(statistic)>critical))
        rate=rejections/M;se=math.sqrt(max(rate*(1-rate),1/M)/M)
        predicted=power(n,d,1,alpha)
        if shape=="normal":assert abs(rate-predicted)<6*max(se,math.sqrt(predicted*(1-predicted)/M))+3/M
        mc.append(dict(n=n,standardized_effect=d,family_size=m,noise_shape=shape,replicates=M,
            empirical_rejection_rate=rate,mc_standard_error=se,normal_model_prediction=predicted,
            is_null_type_I_check=d==0))
    save(out/"power_computation_verification.json",dict(status="PASS",numerical_integration=numerical,
        monte_carlo=mc,note="Heavy-tail and skew checks diagnose assumptions; they are not CFD evidence."))
    save(out/"input_manifest.json",dict(input_npz=str(path),input_npz_sha256=digest(path),
        previous_statistics_archive=str(previous),source_sha256=digest(Path(__file__)),
        config_sha256=digest(config_path),no_original_model_or_score_modified=True))
    environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
        finished_at=datetime.now(ZoneInfo("Europe/Bratislava")).isoformat(),
        elapsed_seconds=time.monotonic()-started,GPU_used=False,new_training_started=False)
    save(out/"environment.json",environment)
    summary=dict(status="PROSPECTIVE_POWER_AND_GPU_COST_ANALYSIS_COMPLETE",
        sd_upper_multiplier=factor,individual_power_targets=c["target_powers"],
        relative_effects=c["relative_effects"],designs=len(designs),
        power_contrast_rows=len(individual),geometry_diagnostic_rows=len(floor_rows),
        stable_mixture_quadrature_calls=_power_fallbacks,
        environment=environment,
        caveats=["Pilot variability estimated from only five runs.",
                 "Planned power is conditional on assumed effect and distribution.",
                 "Same fixed benchmark; additional seeds do not add geometries.",
                 "Independent confirmation uses new cohort N; N-5 is only combined exploratory expansion.",
                 "No training jobs were launched."])
    save(out/"summary.json",summary)
    files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(out.iterdir()) if p.is_file()}
    save(out/"OUTPUT_MANIFEST.json",files)
    print(json.dumps(summary),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();run(a.config,a.output)
