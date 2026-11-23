"""Independent verification of published-statistics candidates; no model inference."""
import csv, hashlib, itertools, json, math, os, sys, time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np

root=Path(sys.argv[1]);out=Path(sys.argv[2]);source=Path(sys.argv[3])
started=time.monotonic()
config=json.loads((source/"config.json").read_text())
manifest=json.loads((out/"input_manifest.json").read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
tables={}
for item in manifest["inputs"]:
    p=Path(item["actual_path"])
    data=p.read_bytes()
    assert hashlib.sha256(data).hexdigest()==item["sha256"]
    assert hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()==item["git_blob_sha"]
    tables[item["key"]]=list(csv.DictReader(data.decode().splitlines()))
bundle=np.load(out/"analysis_inputs.npz",allow_pickle=False)
stats=np.load(out/"bootstrap_replicates.npz",allow_pickle=False)
weights=np.load(out/"bootstrap_weights.npz",allow_pickle=False)
methods=bundle["method_names"].tolist();metrics=bundle["metric_names"].tolist()
ids=bundle["case_ids"].tolist()
cube=np.zeros((11,5,200,3));observed=set()
for row in tables["primary"]+tables["fno"]:
    a=methods.index(row["architecture"])
    seedlist=config["pinn_seeds"] if a==10 else config["architecture_seeds"]
    s=seedlist.index(int(row["seed"]));c=ids.index(int(row["index"]))
    key=(a,s,c);assert key not in observed;observed.add(key)
    cube[a,s,c]=[float(row[m]) for m in metrics]
assert len(observed)==11000
assert np.array_equal(cube,bundle["errors"])
flatmeans=np.empty((11,3))
for a in range(11):
    for k in range(3):
        flatmeans[a,k]=math.fsum(float(v) for v in cube[a,:,:,k].flat)/1000
assert np.allclose(flatmeans,cube.mean(axis=(1,2)),rtol=1e-13,atol=1e-16)
groups=bundle["case_geometric_cluster"]
sample_indices=np.unique(np.concatenate((np.arange(7),np.linspace(0,199999,74,dtype=int))))
boot=stats["paired_crossed_means"]
independent=stats["independent_seed_crossed_means"]
for b in sample_indices:
    geomweights=weights["geometry_counts"][b]
    caseweights=geomweights[groups].astype(np.float64)
    for a in range(11):
        for mode in ["paired","independent"]:
            if a==10:
                seedweights=weights["pinn_seed_counts"][b]
            elif mode=="paired":
                seedweights=weights["shared_architecture_seed_counts"][b]
            else:
                seedweights=weights["independent_architecture_seed_counts"][b,a]
            fullweights=seedweights[:,None]*caseweights[None,:]
            direct=np.sum(cube[a]*fullweights[:,:,None],axis=(0,1))/np.sum(fullweights)
            expected=boot[b,a] if mode=="paired" else independent[b,a]
            assert np.allclose(direct,expected,rtol=3e-13,atol=3e-15),(int(b),a,mode)
rows=json.loads((out/"contrasts.json").read_text())
for row in rows:
    a=methods.index(row["method_a"]);b=methods.index(row["method_b"]);k=metrics.index(row["metric"])
    point=row["difference_a_minus_b"]
    assert math.isclose(flatmeans[a,k]-flatmeans[b,k],point,rel_tol=1e-10,abs_tol=1e-15)
    samples=boot[:,a,k]-boot[:,b,k]
    q=np.percentile(samples,[100*.05/(2*row["family_size"]),100*(1-.05/(2*row["family_size"]))])
    basic=[2*point-q[1],2*point-q[0]]
    assert np.allclose(basic,row["ci_family_basic"],rtol=2e-11,atol=1e-14)
    extreme=int(np.sum(np.abs(samples-point)>=abs(point)))
    assert extreme==row["bootstrap_extreme_count"]
    assert row["bootstrap_p_centered_two_sided"]==(extreme+1)/(len(samples)+1)
    assert row["basic_bonferroni_excludes_zero"]==bool(basic[0]>0 or basic[1]<0)
for family in sorted(set(r["family"] for r in rows)):
    group=sorted([r for r in rows if r["family"]==family],key=lambda r:r["bootstrap_p_centered_two_sided"])
    running=0.
    for rank,row in enumerate(group):
        running=max(running,min(1.,(len(group)-rank)*row["bootstrap_p_centered_two_sided"]))
        assert math.isclose(running,row["bootstrap_p_holm"],rel_tol=1e-13,abs_tol=1e-15)
exact=json.loads((out/"exact_seed_sensitivity.json").read_text())
for row in exact:
    differences=row["training_seed_deltas"];obs=abs(sum(differences)/5)
    threshold=obs-max(obs,max(abs(d) for d in differences),np.finfo(float).tiny)*1e-12
    count=0
    for mask in range(32):
        perm=abs(sum(d*(1 if mask&(1<<j) else -1) for j,d in enumerate(differences))/5)
        count+=perm>=threshold
    assert count/32==row["p_exact_symmetric_sign_flip"]
# Concentration explains why all-cell MSE and normalized L2 need not rank
# methods identically or give the same uncertainty.
concentration=[]
for a,method in enumerate(methods):
    for k,metric in enumerate(metrics):
        percase=cube[a,:,:,k].mean(axis=0)
        totals=np.array([percase[groups==g].sum() for g in range(85)])
        order=np.argsort(totals)[::-1]
        concentration.append(dict(method=method,metric=metric,
            largest_geometry_share_of_error_sum=float(totals[order[0]]/totals.sum()),
            largest_five_geometries_share=float(totals[order[:5]].sum()/totals.sum()),
            largest_geometry_hash=str(weights["geometry_hashes"][order[0]])))
(out/"error_concentration.json").write_text(json.dumps(concentration,indent=2)+"\n")
second_cube=np.full((2,2,200,3),np.nan)
for row in tables["second"]:
    a=["chebkan","rbf"].index(row["architecture"])
    s=config["architecture_seeds"][:2].index(int(row["seed"]))
    c=ids.index(int(row["index"]))
    second_cube[a,s,c]=[float(row[m]) for m in metrics]
assert np.array_equal(second_cube,bundle["second_chance_errors"])
second_baseline=cube[[methods.index("chebkan"),methods.index("rbf")],:2]
second_case_delta=(second_cube-second_baseline).mean(axis=1)
for b in sample_indices:
    w=weights["geometry_counts"][b][groups].astype(float)
    direct=(second_case_delta*w[None,:,None]).sum(axis=1)/w.sum()
    assert np.allclose(direct,stats["fixed_lineage_second_chance_deltas"][b],rtol=3e-13,atol=3e-15)
for row in json.loads((out/"second_chance_conditional.json").read_text()):
    a=["chebkan","rbf"].index(row["method"]);k=metrics.index(row["metric"])
    draws=stats["fixed_lineage_second_chance_deltas"][:,a,k]
    point=float(second_case_delta[a,:,k].mean())
    assert math.isclose(point,row["difference_extended_minus_original"],rel_tol=1e-12,abs_tol=1e-15)
    q=np.percentile(draws,[100*.05/12,100*(1-.05/12)])
    assert np.allclose([2*point-q[1],2*point-q[0]],row["ci_family_basic"],rtol=2e-11,atol=1e-14)
checks=json.loads((out/"implementation_checks.json").read_text())
for name,item in json.loads((out/"OUTPUT_MANIFEST.json").read_text())["files"].items():
    assert (out/name).stat().st_size==item["bytes"]
    assert sha(out/name)==item["sha256"]
result=dict(status="INDEPENDENT_STATISTICAL_VERIFICATION_PASS",
    finished_at=datetime.now(ZoneInfo("Europe/Bratislava")).isoformat(),
    csv_rows_reparsed=11800,independent_means="math.fsum",sampled_replicates_checked=int(len(sample_indices)),
    methods_checked=11,seed_sampling_modes_checked=2,
    resampling_implementation="full case x seed outer-product weights; no analysis helper imports",
    all_165_interval_and_pvalue_checks=True,all_Holm_adjustments_recomputed=True,
    all_exact_sign_flip_enumerations_recomputed=True,all_original_output_hashes_verified=True,
    GPU_used=False,new_training_or_model_inference=False,elapsed_seconds=time.monotonic()-started,
    analysis_source_sha256=sha(source/"analyze.py"),
    verifier_source_sha256=sha(__file__),
    important_sensitivity_flags=checks)
(out/"independent_verification.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result),flush=True)
