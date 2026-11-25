"""Pre-specified paired20-seed confirmation analysis; CPU only."""
import argparse,csv,hashlib,itertools,json,math,os,re,sys,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import scipy
from scipy import stats
ROOT=Path('/mnt/project/perun2601449/dp_llm_cfd')
TEST=ROOT/'results/architecture_confirmation_test_20261125_v2'
TRAIN=ROOT/'results/architecture_confirmation_20261124_v2'
SRC=ROOT/'test/architecture_confirmation_test_20261125_v2'
METHODS=['tfno','fno','realnpu']
METRICS=['all_cell_mse','fluid_velocity_relative_l2','fluid_pressure_relative_l2']
PROOFS={}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 p=Path(p);PROOFS[str(p)]=digest(p);return json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvfile(p,rows):
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def holm(p):
 p=np.asarray(p,float);order=np.argsort(p,kind='stable');out=np.empty_like(p)
 out[order]=np.minimum(1.,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
 return out
def runtime(text,index):
 lines=text.splitlines()
 selected=[x for x in lines if re.search(r'\bArrayTaskId='+str(index)+r'(?:\s|$)',x)]
 if selected:
  assert len(selected)==1
  text=selected[0]
 else:
  assert len(lines)==1
 match=re.search(r'\bRunTime=(\S+)',text);assert match
 x=match.group(1);day=0
 if '-' in x:day,x=x.split('-');day=int(day)
 h,m,s=map(int,x.split(':'));return day*86400+h*3600+m*60+s
def costs(out):
 groups={};proof=[]
 for name,root in [('training',TRAIN),('test_v2',TEST),('technical_test_v1',ROOT/'results/architecture_confirmation_test_20261125_v1')]:
  rows=[]
  for p in sorted((root/'allocation_closeout').glob('*.json')):
   v=read(p);s=v['scheduler_stdout'];g=v['allocated_GPUs'];c=v['allocated_CPUs']
   seconds=runtime(s,v.get('index',0))
   rows.append(dict(file=str(p),role=v.get('role',v.get('mode')),exit_code=v['worker_exit_code'],
    allocated_GPU_h=g*seconds/3600,allocated_CPU_core_h=c*seconds/3600,
    final_portal_billing=False))
  groups[name]=dict(records=rows,observed_closeout_GPU_h=sum(x['allocated_GPU_h'] for x in rows),
   observed_closeout_CPU_core_h=sum(x['allocated_CPU_core_h'] for x in rows))
 save(out/'resource_closeouts.json',groups)
 return groups
def load_inputs(out):
 audit=read(TEST/'audit/summary.json');assert audit['status']=='CONFIRMATION60_TEST_AND_SAVED_OUTPUT_AUDIT_COMPLETE'
 assert audit['saved_native_fields_checked']==12000 and audit['raw_timing_pairs_checked']==12000000
 assert len(audit['models'])==60 and len(audit['methods'])==3
 manifest=read(SRC/'models.json');freeze=read(TEST/'freeze.json');config=read(SRC/'config.json')
 assert digest(SRC/'models.json')==config['models_sha256']==audit['models_sha256']
 frozen=read(TRAIN/'frozen_models.json');assert len(frozen['models'])==60
 protocol=ROOT/'test/architecture_confirmation_20261124_v2/protokol.txt'
 assert digest(protocol)=='2401e5a6761a48d5623f7b49ec23accd0bcb6d24e679d052e4e1cd78c01eb8cc'
 PROOFS[str(protocol)]=digest(protocol)
 trainplan=read(ROOT/'test/architecture_confirmation_20261124_v2/plan.json')
 seeds=trainplan['main_seeds'];assert seeds==[24112001+100000000*i for i in range(20)]
 assert trainplan['primary_family_size']==9 and trainplan['existing_pilot_models_in_primary_analysis'] is False
 split=read(Path(config['split_path']))['indices'];caseids=split['test'];assert len(caseids)==200
 path=TEST/'audit/case_metrics.csv';PROOFS[str(path)]=digest(path)
 with path.open() as f:rows=list(csv.DictReader(f))
 assert len(rows)==12000
 cube=np.full((3,20,200,3),np.nan);casepos={v:i for i,v in enumerate(caseids)}
 known={};identities={m['id']:m for m in manifest['models']}
 for v in rows:
  m=identities[v['model_id']];a=METHODS.index(v['architecture']);s=seeds.index(int(v['seed']));c=casepos[int(v['index'])]
  assert v['model_id']==m['id'] and m['seed']==seeds[s] and m['model']==METHODS[a]
  assert m['seed_block']==s and m['kind']=='new_independent20_seed_confirmation'
  assert np.isnan(cube[a,s,c]).all()
  cube[a,s,c]=[float(v[k]) for k in METRICS]
  identity=(v['geometry_hash'],float(v['reynolds']))
  assert int(v['index']) not in known or known[int(v['index'])]==identity
  known[int(v['index'])]=identity
 assert np.isfinite(cube).all() and (cube>=0).all()
 geometry=sorted({v[0] for v in known.values()});assert len(geometry)==85
 clusters=np.array([geometry.index(known[c][0]) for c in caseids],dtype='int64')
 means=cube.mean(2);published={v['id']:v for v in audit['models']};timing=[]
 maxdiff=0.
 for row in manifest['models']:
  a=METHODS.index(row['model']);s=seeds.index(row['seed']);old=published[row['id']]
  for j,key in enumerate(METRICS):
   diff=abs(float(means[a,s,j])-old[key]);maxdiff=max(maxdiff,diff)
   assert np.isclose(means[a,s,j],old[key],rtol=1e-11,atol=1e-15)
  p=TEST/'test/models'/row['id']/'summary.json';v=read(p)
  assert v['status']=='CONFIRMATION60_MODEL_TEST_COMPLETE' and v['model']==row
  assert v['models_sha256']==audit['models_sha256'] and v['no_weight_updates'] and v['no_test_selection']
  assert [x['index'] for x in v['cases']]==caseids
  assert len(v['cases'])==200
  for j,key in enumerate(METRICS):
   assert np.isclose(means[a,s,j],v['means'][key],rtol=1e-10,atol=1e-13)
  timing.append(dict(id=row['id'],method=row['model'],seed=row['seed'],seed_block=s,
   worst_case_gpu_p99_ms=v['worst_case_gpu_p99_ms'],worst_case_host_p99_ms=v['worst_case_host_p99_ms'],
   GPU_pass=v['gpu_5ms_pass'],host_pass=v['host_instrumented_5ms_pass'],
   GPU_single_queries_above5ms=sum(x['timing']['gpu']['above_target'] for x in v['cases']),
   host_single_queries_above5ms=sum(x['timing']['host_instrumented']['above_target'] for x in v['cases']),
   maximum_GPU_query_ms=max(x['timing']['gpu']['maximum_ms'] for x in v['cases']),
   maximum_host_query_ms=max(x['timing']['host_instrumented']['maximum_ms'] for x in v['cases']),
   TEST_model_elapsed_GPU_h=v['elapsed_seconds']/3600))
  assert row['checkpoint_sha256']==next(x['checkpoint_sha256'] for x in frozen['models'] if x['id']==row['id'])
 np.savez_compressed(out/'analysis_inputs.npz',errors=cube,methods=np.asarray(METHODS),metrics=np.asarray(METRICS),
  seeds=np.asarray(seeds,dtype='int64'),case_ids=np.asarray(caseids),geometry_cluster=clusters,
  geometry_hashes=np.asarray(geometry))
 save(out/'latency_by_model.json',timing)
 save(out/'input_validation.json',dict(status='ALL12000_ROWS_AND60_MODEL_MEANS_REPRODUCED',
  rows=len(rows),models=60,seeds_per_method=20,case_count=200,geometry_clusters=85,
  maximum_difference_from_saved_CPU_audit=maxdiff,old_pilot_models_included=False))
 return cube,seeds,caseids,clusters,timing
def main(out):
 start=time.monotonic();assert not out.exists();out.mkdir(parents=True)
 cube,seeds,caseids,clusters,timing=load_inputs(out);cost=costs(out)
 modelmeans=cube.mean(2);point=modelmeans.mean(1);pairs=list(itertools.combinations(range(3),2))
 contrasts=[(a,b,k) for a,b in pairs for k in range(3)];family=len(contrasts);assert family==9
 D=np.column_stack([modelmeans[a,:,k]-modelmeans[b,:,k] for a,b,k in contrasts])
 delta=D.mean(0);sd=D.std(0,ddof=1);se=sd/math.sqrt(20)
 crit=float(stats.t.isf(.05/(2*9),19));tt=delta/se
 p=np.array([stats.ttest_rel(modelmeans[a,:,k],modelmeans[b,:,k]).pvalue for a,b,k in contrasts])
 adjusted=holm(p);primary=[]
 for j,(a,b,k) in enumerate(contrasts):
  lo,hi=(delta[j]-crit*se[j],delta[j]+crit*se[j])
  q=stats.probplot(D[:,j],dist='norm',fit=True)
  sh=stats.shapiro(D[:,j]);wins=int((D[:,j]<0).sum());losses=int((D[:,j]>0).sum());ties=20-wins-losses
  primary.append(dict(method_a=METHODS[a],method_b=METHODS[b],metric=METRICS[k],n_pairs=20,
   mean_a=float(point[a,k]),mean_b=float(point[b,k]),mean_difference_a_minus_b=float(delta[j]),
   relative_difference_vs_b_percent=float(100*delta[j]/point[b,k]),
   symmetric_relative_difference_percent=float(200*delta[j]/(point[a,k]+point[b,k])),
   paired_SD=float(sd[j]),paired_SE=float(se[j]),paired_standardized_effect_dz=float(delta[j]/sd[j]),
   t=float(tt[j]),df=19,p_two_sided=float(p[j]),p_Holm_9=float(adjusted[j]),
   reject_Holm05=bool(adjusted[j]<.05),Bonferroni_family95_t_CI=[float(lo),float(hi)],
   CI_excludes_zero=bool(lo>0 or hi<0),A_lower_error_seed_pairs=wins,B_lower_error_seed_pairs=losses,tied_seed_pairs=ties,
   paired_seed_differences=D[:,j].tolist(),
   distribution_diagnostic=dict(shapiro_W=float(sh.statistic),shapiro_p=float(sh.pvalue),
    skewness=float(stats.skew(D[:,j],bias=False)),excess_kurtosis=float(stats.kurtosis(D[:,j],bias=False)),
    QQ_theoretical=q[0][0].tolist(),QQ_observed=q[0][1].tolist(),QQ_line=list(map(float,q[1])),
    used_to_change_primary_test=False)))
 save(out/'paired_t_primary.json',primary)
 # All2^20 sign patterns; for fixed sumsq, absolute t is monotone in absolute signed sum.
 extreme=np.zeros(9,dtype='int64');patterns=1<<20;target=abs(D.sum(0))
 tol=64*np.finfo(float).eps*np.abs(D).sum(0)
 for first in range(0,patterns,32768):
  ii=np.arange(first,min(first+32768,patterns),dtype=np.uint32)
  signs=(((ii[:,None]>>np.arange(20,dtype=np.uint32))&1).astype('float64')*2-1)
  sums=abs(signs@D);extreme+=(sums>=target-tol).sum(0)
 assert (extreme>=2).all() and (extreme%2==0).all()
 pf=extreme/patterns;pfh=holm(pf);signs=[]
 ps=[]
 for j in range(9):
  n=int(np.count_nonzero(D[:,j]));positive=int((D[:,j]>0).sum())
  ps.append(float(stats.binomtest(positive,n,.5,alternative='two-sided').pvalue) if n else 1.)
 psh=holm(ps)
 for j,(a,b,k) in enumerate(contrasts):
  signs.append(dict(method_a=METHODS[a],method_b=METHODS[b],metric=METRICS[k],
   exhaustive_sign_patterns=patterns,extreme_sign_patterns=int(extreme[j]),roundoff_tie_tolerance=float(tol[j]),
   signflip_p=float(pf[j]),signflip_p_Holm_9=float(pfh[j]),signflip_reject_Holm05=bool(pfh[j]<.05),
   signflip_assumption='Independent differences symmetric about0 under null; not a randomized architecture assignment.',
   binomial_sign_p=float(ps[j]),binomial_sign_p_Holm_9=float(psh[j]),binomial_sign_reject_Holm05=bool(psh[j]<.05),
   sign_test_estimand='Probability positive versus negative paired difference; median direction under continuity, not mean effect.'))
 save(out/'exact_sign_sensitivities.json',signs)
 # Crossed paired bootstrap: resample20seed IDs and85geometry clusters independently.
 sizes=np.bincount(clusters,minlength=85).astype(float)
 group_sums=np.stack([cube[:,:,clusters==g,:].sum(2) for g in range(85)],axis=2)
 assert group_sums.shape==(3,20,85,3)
 rng=np.random.default_rng(25112501);B=200000;draws=np.empty((B,3,3))
 for first in range(0,B,1000):
  count=min(1000,B-first);sc=rng.multinomial(20,np.full(20,1/20),size=count)/20
  gc=rng.multinomial(85,np.full(85,1/85),size=count);den=gc@sizes
  for a in range(3):
   tmp=np.einsum('bs,sgk->bgk',sc,group_sums[a],optimize=True)
   draws[first:first+count,a,:]=np.einsum('bg,bgk->bk',gc,tmp,optimize=True)/den[:,None]
 assert np.isfinite(draws).all()
 bootdiff=np.column_stack([draws[:,a,k]-draws[:,b,k] for a,b,k in contrasts])
 np.savez_compressed(out/'crossed_bootstrap_replicates.npz',method_means=draws,paired_differences=bootdiff,
  seed=np.asarray([25112501]),iterations=np.asarray([B]))
 bootstrap=[];bp=[]
 for j,(a,b,k) in enumerate(contrasts):
  lo,hi=np.quantile(bootdiff[:,j],[.05/(2*9),1-.05/(2*9)],method='linear')
  pl,ph=np.quantile(bootdiff[:,j],[.025,.975],method='linear')
  central=np.abs(bootdiff[:,j]-delta[j]);prob=(1+np.count_nonzero(central>=abs(delta[j])))/(B+1)
  bp.append(float(prob))
  basic=[float(2*delta[j]-hi),float(2*delta[j]-lo)]
  bootstrap.append(dict(method_a=METHODS[a],method_b=METHODS[b],metric=METRICS[k],
   point_difference=float(delta[j]),paired_crossed_bootstrap_SE=float(np.std(bootdiff[:,j],ddof=1)),
   approximate_Bonferroni_family95_basic_CI=basic,
   approximate_Bonferroni_family95_percentile_CI=[float(lo),float(hi)],
   approximate_individual95_basic_CI=[float(2*delta[j]-ph),float(2*delta[j]-pl)],
   approximate_centered_bootstrap_p=float(prob),basic_family_CI_excludes_zero=bool(basic[0]>0 or basic[1]<0),
   resampling='Paired20seed IDs and85geometry clusters. All cases of sampled clusters included; ratio divided by sampled case count.',
   interpretation='Approximate geometry sensitivity only. Does not replace fixed-benchmark paired t inference or establish unseen-geometry generalization.'))
 bh=holm(bp)
 for j,x in enumerate(bootstrap):x['approximate_bootstrap_p_Holm_9']=float(bh[j])
 save(out/'crossed_geometry_sensitivity.json',bootstrap)
 methods=[]
 for a,m in enumerate(METHODS):
  timingrows=[v for v in timing if v['method']==m]
  d=dict(method=m,independent_training_seeds=20,TEST_cases_per_seed=200,nominal_training_budget_GPU_h_per_model=4,
   GPU_5ms_pass_models=sum(v['GPU_pass'] for v in timingrows),
   host_5ms_pass_models=sum(v['host_pass'] for v in timingrows),
   worst_case_GPU_p99_ms=max(v['worst_case_gpu_p99_ms'] for v in timingrows),
   worst_case_host_p99_ms=max(v['worst_case_host_p99_ms'] for v in timingrows),
   individual_GPU_queries_above5ms=sum(v['GPU_single_queries_above5ms'] for v in timingrows),
   individual_host_queries_above5ms=sum(v['host_single_queries_above5ms'] for v in timingrows),
   maximum_individual_GPU_query_ms=max(v['maximum_GPU_query_ms'] for v in timingrows),
   maximum_individual_host_query_ms=max(v['maximum_host_query_ms'] for v in timingrows))
  for j,key in enumerate(METRICS):
   d[key]=float(point[a,j]);d[key+'_seed_SD']=float(modelmeans[a,:,j].std(ddof=1))
  methods.append(d)
 save(out/'method_summary.json',methods)
 csvfile(out/'primary_t_tests.csv',[{k:v for k,v in x.items() if k not in ['paired_seed_differences','distribution_diagnostic']} for x in primary])
 csvfile(out/'per_seed_model_means.csv',[dict(method=m,seed=seed,seed_block=s,**{key:float(modelmeans[a,s,k]) for k,key in enumerate(METRICS)})
  for a,m in enumerate(METHODS) for s,seed in enumerate(seeds)])
 # This verifies only the p-value arithmetic of the assumed normal t model.
 rng=np.random.default_rng(25112502);mc=[]
 for n,shift in [(20,0.),(20,.8)]:
  x=rng.normal(size=(100000,n))+shift;t=x.mean(1)/(x.std(1,ddof=1)/math.sqrt(n));critical=stats.t.isf(.05/(2*9),n-1)
  empirical=float((abs(t)>critical).mean());expected=float(stats.nct.sf(critical,n-1,shift*math.sqrt(n))+stats.nct.cdf(-critical,n-1,shift*math.sqrt(n)))
  se_mc=math.sqrt(max(expected*(1-expected),1e-8)/len(x))
  assert abs(empirical-expected)<6*se_mc+2/len(x)
  mc.append(dict(N=n,normal_location_shift=shift,rejection=empirical,normal_model_probability=expected))
 save(out/'calculation_checks.json',dict(normal_model_Monte_Carlo=mc,sign_patterns=patterns,
  signflip_p_minimum=2/patterns,primary_family_size=9,confidence_critical_t=crit,
  raw_case_means_verified=True,all_three_metrics_reported=True))
 save(out/'input_manifest.json',dict(files=PROOFS,source_sha256=digest(Path(__file__)),
  original_model_weights_or_results_changed=False,old_five_pilot_models_in_primary=False))
 summary=dict(status='CONFIRMATION60_STATISTICAL_COMPARISON_COMPLETE',methods=METHODS,metrics=METRICS,
  training_models=60,independent_seed_blocks=20,TEST_cases_per_model=200,geometry_clusters=85,
  primary_contrasts=9,primary_Holm_rejections=sum(x['reject_Holm05'] for x in primary),
  primary_Bonferroni_CI_exclusions=sum(x['CI_excludes_zero'] for x in primary),
  signflip_Holm_rejections=sum(x['signflip_reject_Holm05'] for x in signs),
  sign_test_Holm_rejections=sum(x['binomial_sign_reject_Holm05'] for x in signs),
  crossed_basic_CI_exclusions=sum(x['basic_family_CI_excludes_zero'] for x in bootstrap),
  TEST_and_full_field_audit_pass=True,all60_per_case_p99_GPU_pass=all(x['GPU_5ms_pass_models']==20 for x in methods),
  bootstrap_iterations=B,bootstrap_seed=25112501,no_new_training=True,no_GPU_used=True,
  finished_at=datetime.now(ZoneInfo('Europe/Bratislava')).isoformat(),
  elapsed_seconds=time.monotonic()-start,environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__),
  inference='Prespecified paired Student t over20seedwise200case means, two sided, Holm over9contrasts, Bonferroni family95%t CIs.',
  limitations=['Fixed known200case benchmark with85geometries also present inTRAIN.',
   'Three architectures selected after earlier TEST results were known. This is a dated independent-seed extension, not a blind new-geometry experiment.',
   'Twenty differences require distributional assumptions for exact t inference. Shape diagnostics do not change the preselected primary test.',
   'Sign-flip assumes null symmetry; sign test concerns median direction; crossed bootstrap is approximate sensitivity.',
   'Nonsignificant results do not establish equivalence.',
   'All60models pass per-casep99<=5ms onH200 with resident inputs; this does not mean every individual call was<=5ms.',
   'Results concern fixed implementations/configurations and equal allocated-resource budgets, not universal architectural dominance.'])
 save(out/'summary.json',summary)
 save(out/'OUTPUT_MANIFEST.json',{p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in out.iterdir() if p.is_file()})
 print(json.dumps(summary),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
