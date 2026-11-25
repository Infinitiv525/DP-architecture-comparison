"""Independent scalar t/beta, exact meet-in-middle signs and saved-bootstrap audit."""
import csv,hashlib,json,math,statistics,time
from pathlib import Path
import numpy as np
from scipy import special,optimize
R=Path('/mnt/project/perun2601449/dp_llm_cfd')
O=R/'results/architecture_confirmation_statistics_20261125_v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,rtol=1e-10,atol=1e-16):
 assert math.isclose(float(a),float(b),rel_tol=rtol,abs_tol=atol),(a,b)
def holm(p):
 order=sorted(range(len(p)),key=lambda k:(p[k],k));result=[None]*len(p);previous=0.
 for rank,i in enumerate(order):
  previous=max(previous,min(1.,(len(p)-rank)*p[i]));result[i]=previous
 return result
start=time.monotonic();inp=read(O/'input_manifest.json')
for path,h in inp['files'].items():assert sha(Path(path))==h,path
assert sha(R/'test/architecture_confirmation_statistics_20261125_v1/analyze.py')==inp['source_sha256']
b=np.load(O/'analysis_inputs.npz',allow_pickle=False);cube=b['errors'];methods=b['methods'].tolist();metrics=b['metrics'].tolist()
seeds=b['seeds'].tolist();ids=b['case_ids'].tolist();geometry=b['geometry_cluster'];assert cube.shape==(3,20,200,3)
table={}
with (R/'results/architecture_confirmation_test_20261125_v2/audit/case_metrics.csv').open() as f:
 for row in csv.DictReader(f):table[(row['architecture'],int(row['seed']),int(row['index']))]=row
assert len(table)==12000
means={}
for a,m in enumerate(methods):
 for s,seed in enumerate(seeds):
  for k,key in enumerate(metrics):
   x=[float(table[(m,seed,index)][key]) for index in ids]
   assert np.array_equal(np.asarray(x),cube[a,s,:,k])
   means[(m,seed,key)]=statistics.fmean(x)
rows=read(O/'paired_t_primary.json');ps=[];flip=read(O/'exact_sign_sensitivities.json')
critical=optimize.brentq(lambda t:special.betainc(19/2,.5,19/(19+t*t))-.05/9,0,20,xtol=1e-13)
for j,x in enumerate(rows):
 d=[means[(x['method_a'],seed,x['metric'])]-means[(x['method_b'],seed,x['metric'])] for seed in seeds]
 assert len(d)==20
 mu=statistics.fmean(d);sd=statistics.stdev(d);se=sd/math.sqrt(20);t=mu/se
 p=float(special.betainc(19/2,.5,19/(19+t*t)));ps.append(p)
 for a,c in [(mu,x['mean_difference_a_minus_b']),(sd,x['paired_SD']),(se,x['paired_SE']),(t,x['t']),(p,x['p_two_sided'])]:close(a,c,atol=1e-300 if c==x['p_two_sided'] else 1e-16)
 close(mu-critical*se,x['Bonferroni_family95_t_CI'][0])
 close(mu+critical*se,x['Bonferroni_family95_t_CI'][1])
 # Different algorithm:1024left+1024right sign sums, no million-row matrix.
 halves=[]
 for values in [d[:10],d[10:]]:
  z=np.array([0.])
  for v in values:z=np.concatenate((z+v,z-v))
  halves.append(z)
 left,right=halves;right=np.sort(right);target=abs(sum(d));tol=flip[j]['roundoff_tie_tolerance'];assert target>tol
 upper=int(np.sum(1024-np.searchsorted(right,target-tol-left,side='left')))
 lower=int(np.sum(np.searchsorted(right,-target+tol-left,side='right')))
 assert upper+lower==flip[j]['extreme_sign_patterns'],(j,upper+lower,flip[j])
 close((upper+lower)/2**20,flip[j]['signflip_p'])
 pos=sum(v>0 for v in d);neg=sum(v<0 for v in d);n=pos+neg
 prob=min(1.,2*sum(math.comb(n,i) for i in range(min(pos,neg)+1))/2**n)
 close(prob,flip[j]['binomial_sign_p'])
for actual,expected in zip(holm(ps),rows):close(actual,expected['p_Holm_9'],atol=1e-300)
for key,outkey in [('signflip_p','signflip_p_Holm_9'),('binomial_sign_p','binomial_sign_p_Holm_9')]:
 for actual,expected in zip(holm([x[key] for x in flip]),flip):close(actual,expected[outkey])
boot=np.load(O/'crossed_bootstrap_replicates.npz',allow_pickle=False);draws=boot['method_means'];diff=boot['paired_differences']
assert draws.shape==(200000,3,3) and diff.shape==(200000,9)
rng=np.random.default_rng(25112501);sc=rng.multinomial(20,np.full(20,1/20),size=1000);gc=rng.multinomial(85,np.full(85,1/85),size=1000)
sizes=np.bincount(geometry,minlength=85);gs=np.stack([cube[:,:,geometry==g,:].sum(2) for g in range(85)],axis=2)
for i in range(3):
 den=sum(int(gc[i,g])*int(sizes[g]) for g in range(85))
 for a in range(3):
  for k in range(3):
   value=sum(int(sc[i,s])*int(gc[i,g])*float(gs[a,s,g,k]) for s in range(20) for g in range(85))/(20*den)
   close(value,draws[i,a,k],atol=1e-14)
sens=read(O/'crossed_geometry_sensitivity.json');bp=[]
for j,x in enumerate(rows):
 a=methods.index(x['method_a']);bb=methods.index(x['method_b']);k=metrics.index(x['metric'])
 assert np.array_equal(diff[:,j],draws[:,a,k]-draws[:,bb,k])
 ordered=np.sort(diff[:,j])
 def quantile(q):
  p=(len(ordered)-1)*q;lo=int(math.floor(p));hi=int(math.ceil(p))
  return float(ordered[lo]+(p-lo)*(ordered[hi]-ordered[lo]))
 lo=quantile(.05/18);hi=quantile(1-.05/18);mu=x['mean_difference_a_minus_b']
 for value,expected in zip([2*mu-hi,2*mu-lo],sens[j]['approximate_Bonferroni_family95_basic_CI']):close(value,expected)
 for value,expected in zip([lo,hi],sens[j]['approximate_Bonferroni_family95_percentile_CI']):close(value,expected)
 p=(1+np.count_nonzero(np.abs(diff[:,j]-mu)>=abs(mu)))/(len(diff)+1);bp.append(float(p))
 close(p,sens[j]['approximate_centered_bootstrap_p'])
for p,x in zip(holm(bp),sens):close(p,x['approximate_bootstrap_p_Holm_9'])
summary=read(O/'summary.json')
assert summary['primary_Holm_rejections']==sum(x['p_Holm_9']<.05 for x in rows)
assert summary['primary_Bonferroni_CI_exclusions']==sum(x['CI_excludes_zero'] for x in rows)
assert summary['primary_contrasts']==9 and summary['independent_seed_blocks']==20
report=dict(status='INDEPENDENT_INPUT_PRIMARY_EXACT_SIGN_AND_BOOTSTRAP_AUDIT_PASS',
 input_files_verified=len(inp['files']),raw_case_rows_reproduced=12000,per_model_metric_means_verified=180,
 primary_t_contrasts_verified=9,t_pvalue_route='regularized incomplete beta; scalar statistics module',
 signflip_route='Meet-in-the-middle1024x1024 sign sums with sorted tail counting, agrees with full2^20enumeration',
 sign_test_route='Exact integer binomial sums',bootstrap_full_intervals_verified=9,
 independently_replayed_bootstrap_replicates=3,old_pilot_models_included=False,no_GPU=True,
 elapsed_seconds=time.monotonic()-start)
(O/'independent_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
