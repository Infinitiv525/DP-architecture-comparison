"""Post-hoc crossed-factor analysis of frozen architecture TEST metrics. CPU only."""
import argparse, csv, hashlib, itertools, json, os, platform, sys, time
from pathlib import Path
import numpy as np

METRIC_LABELS = ["MSE u/v/p", "Relativna L2 rychlosti", "Relativna L2 tlaku"]

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def write_csv(path, rows):
    rows = list(rows)
    assert rows
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def holm(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values))
    adjusted = np.minimum(1.0, np.maximum.accumulate((len(values)-np.arange(len(values))) * values[order]))
    result[order] = adjusted
    return result

def excludes_zero(interval):
    return bool(interval[0] > 0 or interval[1] < 0)

def intervals(draws, observed, family_size):
    q = np.quantile(draws, [0.025, 0.975, 0.05/(2*family_size), 1-0.05/(2*family_size)], method="linear")
    return dict(ci95_basic=[float(2*observed-q[1]), float(2*observed-q[0])],
                ci_family_basic=[float(2*observed-q[3]), float(2*observed-q[2])],
                ci95_percentile=[float(q[0]),float(q[1])],
                ci_family_percentile=[float(q[2]),float(q[3])])

def exact_sign_flip(differences):
    d = np.asarray(differences, dtype=np.float64)
    signs = np.array(list(itertools.product([-1,1], repeat=len(d))), dtype=np.float64)
    observed = abs(float(d.mean()))
    null = np.abs(signs @ d / len(d))
    tolerance = max(observed, float(np.max(np.abs(d))), np.finfo(float).tiny) * 1e-12
    return float(np.count_nonzero(null >= observed-tolerance) / len(null))

def resolve_input(root, item):
    remote, public = root/item["remote"], root/item["public"]
    if public.is_file():
        return public
    if remote.is_file():
        return remote
    raise FileNotFoundError((str(public), str(remote)))

def run(config_path, input_root, output, published_summary):
    started, cpu_started = time.monotonic(), time.process_time()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary_bytes = published_summary.read_bytes()
    assert blob_sha(summary_bytes) == config["summary_blob"]
    published = json.loads(summary_bytes)
    methods, metrics = config["methods"], config["metrics"]
    assert len(methods)==11 and len(metrics)==3
    source_digest = sha256(__file__)
    inputs, tables = [], {}
    for item in config["inputs"]:
        path = resolve_input(input_root, item)
        data = path.read_bytes()
        assert blob_sha(data)==item["blob"], item["key"]
        rows = list(csv.DictReader(data.decode("utf-8").splitlines()))
        assert len(rows)==item["rows"]
        inputs.append(dict(key=item["key"], actual_path=str(path), public_path=item["public"],
                           git_blob_sha=item["blob"], sha256=hashlib.sha256(data).hexdigest(),
                           bytes=len(data), rows=len(rows)))
        tables[item["key"]]=rows
    rows=tables["primary"]+tables["fno"]
    identities={}
    for row in rows:
        index=int(row["index"])
        signature=(row["geometry_hash"],float(row["reynolds"]))
        assert index not in identities or identities[index]==signature
        identities[index]=signature
    ids=sorted(identities)
    assert len(ids)==200
    geometries=sorted(set(identities[index][0] for index in ids))
    assert len(geometries)==85
    group_by_case=np.array([geometries.index(identities[index][0]) for index in ids])
    counts=np.bincount(group_by_case,minlength=85).astype(np.float64)
    id_positions={index:j for j,index in enumerate(ids)}
    cube=np.full((11,5,200,3),np.nan,dtype=np.float64)
    metadata={}
    seed_lists=[config["pinn_seeds"] if method=="pinn" else config["architecture_seeds"] for method in methods]
    for row in rows:
        a=methods.index(row["architecture"])
        seed=int(row["seed"])
        s=seed_lists[a].index(seed)
        c=id_positions[int(row["index"])]
        assert np.isnan(cube[a,s,c]).all(), (row["model_id"],c)
        cube[a,s,c]=[float(row[key]) for key in metrics]
        model_id=row["model_id"]
        if model_id not in metadata:
            metadata[model_id]=dict(method=methods[a],seed=seed,model_id=model_id,role=row["role"],
                                    gpu_p99=[],host_p99=[])
        record=metadata[model_id]
        assert record["method"]==methods[a] and record["seed"]==seed
        record["gpu_p99"].append(float(row["gpu_p99_ms"]))
        record["host_p99"].append(float(row["host_instrumented_p99_ms"]))
    assert np.isfinite(cube).all() and (cube>=0).all()
    assert len(metadata)==55
    means=cube.mean(axis=(1,2))
    modelmeans=cube.mean(axis=2)
    group_sums=np.zeros((11,5,85,3),dtype=np.float64)
    for g in range(85):
        group_sums[:,:,g,:]=cube[:,:,group_by_case==g,:].sum(axis=2)
    published_models={row["id"]:row for row in published["models"]}
    max_aggregate_difference=0.0
    for row in rows:
        mid=row["model_id"]
        if mid not in published_models:
            raise AssertionError(mid)
    for mid, record in metadata.items():
        a=methods.index(record["method"]); s=seed_lists[a].index(record["seed"])
        old=published_models[mid]
        for k,key in enumerate(metrics):
            difference=abs(float(modelmeans[a,s,k])-old[key])
            max_aggregate_difference=max(max_aggregate_difference,difference)
            assert np.isclose(modelmeans[a,s,k],old[key],rtol=1e-11,atol=1e-15),(mid,key)
        assert np.isclose(max(record["gpu_p99"]),old["worst_gpu_p99_ms"],rtol=1e-12,atol=1e-12)
        assert np.isclose(max(record["host_p99"]),old["worst_host_p99_ms"],rtol=1e-12,atol=1e-12)
    published_methods={row["model"]:row for row in published["methods"]}
    for a,method in enumerate(methods):
        for k,key in enumerate(metrics):
            assert np.isclose(means[a,k],published_methods[method][key+"_mean"],rtol=1e-11,atol=1e-15)
    # The original paired second-chance report used a separately recomputed
    # float64 MSE. The CPU audit CSV is the canonical input here for both states.
    second_cube=np.full((2,2,200,3),np.nan,dtype=np.float64)
    second_methods=["chebkan","rbf"]
    for row in tables["second"]:
        a=second_methods.index(row["architecture"])
        s=config["architecture_seeds"][:2].index(int(row["seed"]))
        c=id_positions[int(row["index"])]
        assert identities[int(row["index"])]==(row["geometry_hash"],float(row["reynolds"]))
        assert np.isnan(second_cube[a,s,c]).all()
        second_cube[a,s,c]=[float(row[key]) for key in metrics]
    assert np.isfinite(second_cube).all()
    second_original=cube[[methods.index(v) for v in second_methods],:2,:,:]
    second_difference=(second_cube-second_original).mean(axis=1) # two fixed lineages
    second_group=np.zeros((2,85,3),dtype=np.float64)
    for g in range(85):
        second_group[:,g,:]=second_difference[:,group_by_case==g,:].sum(axis=1)
    second_point=second_difference.mean(axis=1)

    assert not output.exists(), "Use a new output folder; do not overwrite a completed analysis."
    output.mkdir(parents=True)
    write_json(output/"input_manifest.json",dict(input_commit=config["input_commit"],inputs=inputs,
        summary_git_blob=config["summary_blob"],summary_sha256=hashlib.sha256(summary_bytes).hexdigest(),
        source_sha256=source_digest,config_sha256=sha256(config_path)))
    write_json(output/"data_audit.json",dict(status="FROZEN_CSV_AND_ORIGINAL_AGGREGATES_PASS",
        main_models=45,supplementary_models=10,second_chance_models=4,
        main_csv_rows=10000,dense_fno_csv_rows=1000,second_chance_csv_rows=800,
        test_cases=200,geometric_clusters=85,cases_per_cluster=counts.astype(int).tolist(),
        maximum_aggregate_absolute_difference=max_aggregate_difference,
        finite_complete_rectangular_tensor=True,model_case_duplicates=0,
        missing_models_or_cases=0,original_scores_preserved=True))
    B=int(config["bootstrap_replicates"])
    rng=np.random.Generator(np.random.PCG64(config["bootstrap_seed"]))
    shared_seed=rng.multinomial(5,[.2]*5,size=B).astype(np.int8)
    pinn_seed=rng.multinomial(5,[.2]*5,size=B).astype(np.int8)
    geo=rng.multinomial(85,np.full(85,1/85),size=B).astype(np.int8)
    separate_seed=rng.multinomial(5,[.2]*5,size=(B,10)).astype(np.int8)
    boot=np.empty((B,11,3),dtype=np.float64)
    independent=np.empty_like(boot)
    second_boot=np.empty((B,2,3),dtype=np.float64)
    for start in range(0,B,config["chunk_size"]):
        stop=min(B,start+config["chunk_size"])
        gw=geo[start:stop].astype(float)
        den=gw@counts
        for a,method in enumerate(methods):
            sw=(pinn_seed if method=="pinn" else shared_seed)[start:stop].astype(float)
            # Sum training seed first; only small (chunk,85,3) intermediates.
            sampled=np.einsum("bs,sgk->bgk",sw,group_sums[a],optimize=True)
            boot[start:stop,a,:]=np.einsum("bg,bgk->bk",gw,sampled,optimize=True)/(5*den[:,None])
            if method=="pinn":
                independent[start:stop,a,:]=boot[start:stop,a,:]
            else:
                iw=separate_seed[start:stop,a,:].astype(float)
                sampled=np.einsum("bs,sgk->bgk",iw,group_sums[a],optimize=True)
                independent[start:stop,a,:]=np.einsum("bg,bgk->bk",gw,sampled,optimize=True)/(5*den[:,None])
        second_boot[start:stop]=np.einsum("bg,agk->bak",gw,second_group,optimize=True)/den[:,None,None]
        if start==0 or stop==B or stop%32768==0:
            write_json(output/"progress.json",dict(completed=stop,total=B,seconds=time.monotonic()-started))
            print("BOOTSTRAP",stop,"/",B,flush=True)

    # Independent arithmetic checks: explicitly expand sampled blocks/cases.
    checked_replicas=23
    for b in range(checked_replicas):
        cases=np.concatenate([np.flatnonzero(group_by_case==g) for g in np.repeat(np.arange(85),geo[b].astype(int))])
        for a in range(11):
            weights=pinn_seed[b] if methods[a]=="pinn" else shared_seed[b]
            selected=np.repeat(np.arange(5),weights.astype(int))
            direct=cube[a][selected][:,cases,:].mean(axis=(0,1))
            assert np.allclose(direct,boot[b,a],rtol=3e-13,atol=3e-15)
            weights=pinn_seed[b] if methods[a]=="pinn" else separate_seed[b,a]
            selected=np.repeat(np.arange(5),weights.astype(int))
            direct=cube[a][selected][:,cases,:].mean(axis=(0,1))
            assert np.allclose(direct,independent[b,a],rtol=3e-13,atol=3e-15)
        assert np.allclose(second_difference[:,cases,:].mean(axis=1),second_boot[b],rtol=3e-13,atol=3e-15)
    assert np.allclose(holm([.01,.04,.03]),[.03,.06,.06])
    assert np.allclose(holm([1,0,.5]),[1,0,1])
    assert exact_sign_flip(np.ones(5))==.0625
    assert exact_sign_flip(np.zeros(5))==1.0
    zero=intervals(np.zeros(1000),0,108)
    assert zero["ci_family_basic"]==[0.,0.]
    assert (1+np.count_nonzero(np.abs(np.zeros(1000))>=0))/(1001)==1
    constant=np.ones((5,85,3))
    for b in range(7):
        val=np.einsum("s,g,sgk->k",shared_seed[b],geo[b],constant)/(5*85)
        assert np.allclose(val,1)
    np.savez_compressed(output/"bootstrap_replicates.npz",
        method_names=np.asarray(methods),metric_names=np.asarray(metrics),
        paired_crossed_means=boot,independent_seed_crossed_means=independent,
        fixed_lineage_second_chance_deltas=second_boot)
    np.savez_compressed(output/"bootstrap_weights.npz",
        shared_architecture_seed_counts=shared_seed,pinn_seed_counts=pinn_seed,
        geometry_counts=geo,independent_architecture_seed_counts=separate_seed,
        geometry_hashes=np.asarray(geometries),case_counts_per_geometry=counts,
        architecture_seeds=np.asarray(config["architecture_seeds"]),
        pinn_seeds=np.asarray(config["pinn_seeds"]))
    np.savez_compressed(output/"analysis_inputs.npz",case_ids=np.asarray(ids),method_names=np.asarray(methods),
        metric_names=np.asarray(metrics),case_geometric_cluster=group_by_case,errors=cube,
        second_chance_errors=second_cube)
    pairs=[(a,b) for a in range(11) for b in range(a+1,11)]
    assert len(pairs)==55
    contrasts=[]
    exact_rows=[]
    for a,b in pairs:
        main=a<9 and b<9
        family="original_nine_methods" if main else "posthoc_FNO_and_auxiliary_PINN"
        m=108 if main else 57
        for k,key in enumerate(metrics):
            observed=float(means[a,k]-means[b,k])
            samples=boot[:,a,k]-boot[:,b,k]
            intervals_main=intervals(samples,observed,m)
            null_deviation=np.abs(samples-observed)
            extreme=int(np.count_nonzero(null_deviation>=abs(observed)))
            p=(extreme+1)/(B+1)
            independent_interval=intervals(independent[:,a,k]-independent[:,b,k],observed,m)
            half1=intervals(samples[:B//2],observed,m)
            half2=intervals(samples[B//2:],observed,m)
            equal_geometry=float((group_sums[a]/counts[None,:,None]).mean(axis=(0,1))[k]-
                                 (group_sums[b]/counts[None,:,None]).mean(axis=(0,1))[k])
            loo=[]
            if methods[a]!="pinn" and methods[b]!="pinn":
                differences=modelmeans[a,:,k]-modelmeans[b,:,k]
                loo=[float(np.delete(differences,j).mean()) for j in range(5)]
                exact_rows.append(dict(method_a=methods[a],method_b=methods[b],metric=key,
                    family=family,seed_blocks=5,permutations=32,
                    p_exact_symmetric_sign_flip=exact_sign_flip(differences),
                    minimum_possible_two_sided_p=.0625,training_seed_deltas=differences.tolist()))
            row=dict(family=family,method_a=methods[a],method_b=methods[b],metric=key,
                mean_a=float(means[a,k]),mean_b=float(means[b,k]),difference_a_minus_b=observed,
                relative_difference_percent=100*float(means[a,k]/means[b,k]-1),
                **intervals_main,family_size=m,individual_CI_nominal_coverage=1-.05/m,
                bootstrap_p_centered_two_sided=p,bootstrap_extreme_count=extreme,
                bootstrap_p_resolution=1/(B+1),p_mc_standard_error=float(np.sqrt(p*(1-p)/(B+1))),
                basic_bonferroni_excludes_zero=excludes_zero(intervals_main["ci_family_basic"]),
                percentile_bonferroni_excludes_zero=excludes_zero(intervals_main["ci_family_percentile"]),
                independent_seeds_ci_family_basic=independent_interval["ci_family_basic"],
                independent_seeds_bonferroni_excludes_zero=excludes_zero(independent_interval["ci_family_basic"]),
                equal_geometry_difference=equal_geometry,
                leave_one_seed_out_min=min(loo) if loo else None,
                leave_one_seed_out_max=max(loo) if loo else None,
                first_half_ci_family_basic=half1["ci_family_basic"],
                second_half_ci_family_basic=half2["ci_family_basic"],
                half_decisions_agree=excludes_zero(half1["ci_family_basic"])==excludes_zero(half2["ci_family_basic"]),
                half_directions_agree=bool(np.sign(half1["ci_family_basic"][0])==np.sign(half2["ci_family_basic"][0]) and
                                          np.sign(half1["ci_family_basic"][1])==np.sign(half2["ci_family_basic"][1])))
            contrasts.append(row)
    for family,size in [("original_nine_methods",108),("posthoc_FNO_and_auxiliary_PINN",57)]:
        group=[r for r in contrasts if r["family"]==family]
        assert len(group)==size
        adjusted=holm([r["bootstrap_p_centered_two_sided"] for r in group])
        for row,p in zip(group,adjusted):
            row["bootstrap_p_holm"]=float(p)
            row["holm_reject_alpha_05"]=bool(p<.05)
            row["conservative_supported_difference"]=row["basic_bonferroni_excludes_zero"]
    for family in sorted(set(r["family"] for r in exact_rows)):
        group=[r for r in exact_rows if r["family"]==family]
        for row,p in zip(group,holm([r["p_exact_symmetric_sign_flip"] for r in group])):
            row["p_holm_within_this_sensitivity_family"]=float(p)
    assert len(contrasts)==165

    second_results=[]
    for a,method in enumerate(second_methods):
        for k,key in enumerate(metrics):
            obs=float(second_point[a,k])
            q=intervals(second_boot[:,a,k],obs,6)
            second_results.append(dict(method=method,metric=key,fixed_training_lineages=2,
                geometric_clusters=85,original_mean=float(second_original[a,:,:,k].mean()),
                extended_mean=float(second_cube[a,:,:,k].mean()),difference_extended_minus_original=obs,
                relative_difference_percent=100*float(second_cube[a,:,:,k].mean()/second_original[a,:,:,k].mean()-1),
                **q,conditional_on_two_specific_training_lineages=True,
                basic_bonferroni_excludes_zero=excludes_zero(q["ci_family_basic"])))
    descriptions=[]
    latency=[]
    for a,method in enumerate(methods):
        records=[r for r in metadata.values() if r["method"]==method]
        for k,key in enumerate(metrics):
            descriptions.append(dict(method=method,cohort="primary" if a<9 else "supplementary",
                metric=key,training_runs=5,test_cases_per_model=200,geometric_clusters=85,
                mean=float(means[a,k]),sample_sd_of_five_model_means=float(modelmeans[a,:,k].std(ddof=1)),
                minimum_model_mean=float(modelmeans[a,:,k].min()),maximum_model_mean=float(modelmeans[a,:,k].max()),
                equal_geometry_mean=float((group_sums[a]/counts[None,:,None]).mean(axis=(0,1))[k])))
        latency.append(dict(method=method,cohort="primary" if a<9 else "supplementary",models=5,
            gpu_pass_models=sum(max(r["gpu_p99"])<=5 for r in records),
            host_pass_models=sum(max(r["host_p99"])<=5 for r in records),
            worst_gpu_case_p99_ms=max(max(r["gpu_p99"]) for r in records),
            worst_host_case_p99_ms=max(max(r["host_p99"]) for r in records),
            five_worst_gpu_p99_ms=[max(r["gpu_p99"]) for r in sorted(records,key=lambda r:r["seed"])],
            five_worst_host_p99_ms=[max(r["host_p99"]) for r in sorted(records,key=lambda r:r["seed"])]))
    assert all(not r["basic_bonferroni_excludes_zero"] or r["ci_family_basic"][0]*r["ci_family_basic"][1]>0 for r in contrasts)
    # The rows are already uniquely keyed before sorting/report generation.
    assert len(set((r["method_a"],r["method_b"],r["metric"]) for r in contrasts))==165
    checks=dict(status="PASS",explicit_loop_replicas_checked=checked_replicas,
        models_checked_per_replica=11,both_seed_resampling_modes_checked=True,
        second_chance_direct_expansion_checked=True,original_aggregates_reproduced=True,
        holm_known_vectors_pass=True,zero_difference_p_one=True,
        constant_tensor_invariance=True,exact_sign_flip_all_positive_p=.0625,
        output_contrasts=165,model_metric_summaries=33,
        first_second_half_decision_disagreements=[{k:r[k] for k in ["method_a","method_b","metric","family"]}
                                                 for r in contrasts if not r["half_decisions_agree"]],
        paired_vs_independent_seed_decision_disagreements=[{k:r[k] for k in ["method_a","method_b","metric","family"]}
            for r in contrasts if r["basic_bonferroni_excludes_zero"]!=r["independent_seeds_bonferroni_excludes_zero"]],
        basic_vs_percentile_decision_disagreements=[{k:r[k] for k in ["method_a","method_b","metric","family"]}
            for r in contrasts if r["basic_bonferroni_excludes_zero"]!=r["percentile_bonferroni_excludes_zero"]])
    write_json(output/"contrasts.json",contrasts)
    write_json(output/"exact_seed_sensitivity.json",exact_rows)
    write_json(output/"second_chance_conditional.json",second_results)
    write_json(output/"method_summary.json",descriptions)
    write_json(output/"latency_descriptive.json",latency)
    write_json(output/"implementation_checks.json",checks)
    # CSV fields containing intervals/vectors are valid JSON strings.
    for filename,values in [("pairwise_comparisons.csv",contrasts),("method_summary.csv",descriptions),
                            ("second_chance_conditional.csv",second_results),("latency_descriptive.csv",latency),
                            ("exact_seed_sensitivity.csv",exact_rows)]:
        write_csv(output/filename,[{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v
                                  for k,v in row.items()} for row in values])
    write_csv(output/"case_cluster_map.csv",[dict(index=index,geometry_hash=identities[index][0],
        reynolds=identities[index][1],cluster=int(group_by_case[j])) for j,index in enumerate(ids)])
    for item in inputs:
        assert sha256(item["actual_path"])==item["sha256"], "Input mutated during analysis."
    assert sha256(__file__)==source_digest
    environment=dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        requested_cpus=os.environ.get("SLURM_CPUS_PER_TASK"),gpu_requested=False,
        elapsed_seconds=time.monotonic()-started,cpu_process_seconds=time.process_time()-cpu_started,
        blas_threads={key:os.environ.get(key) for key in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]})
    write_json(output/"environment.json",environment)
    result=dict(status="ARCHITECTURE_POSTHOC_STATISTICS_COMPLETE",
        date=config["date"],posthoc=True,original_scores_and_weights_unchanged=True,
        gpu_jobs_or_new_inference=False,bootstrap_replicates=B,bootstrap_seed=config["bootstrap_seed"],
        main_family_contrasts=108,supplementary_family_contrasts=57,
        primary_basic_bonferroni_nonzero=sum(r["basic_bonferroni_excludes_zero"] for r in contrasts if r["family"]=="original_nine_methods"),
        supplementary_basic_bonferroni_nonzero=sum(r["basic_bonferroni_excludes_zero"] for r in contrasts if r["family"]!="original_nine_methods"),
        primary_holm_rejections=sum(r["holm_reject_alpha_05"] for r in contrasts if r["family"]=="original_nine_methods"),
        training_replicates_per_method=5,test_cases_per_model=200,geometry_clusters=85,
        approximate_inference_not_exact_coverage_guarantee=True,
        minimum_exact_two_sided_seed_flip_p=.0625,checks=checks,environment=environment)
    write_json(output/"summary.json",result)
    files={p.name:dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output/"OUTPUT_MANIFEST.json",dict(status="COMPLETE",files=files))
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--input-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--published-summary",type=Path,required=True)
    args=parser.parse_args()
    run(args.config,args.input_root,args.output,args.published_summary)
