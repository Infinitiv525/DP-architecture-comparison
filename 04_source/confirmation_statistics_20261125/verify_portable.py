"""Usage:python verify_portable.py PATH_TO_EXTRACTED_STATISTICS_RESULTS"""
import json,math,sys
from pathlib import Path
import numpy as np
from scipy import special
p=Path(sys.argv[1]);b=np.load(p/'analysis_inputs.npz',allow_pickle=False)
m=b['methods'].tolist();k=b['metrics'].tolist();means=b['errors'].mean(2);rows=json.loads((p/'paired_t_primary.json').read_text())
ps=[]
for row in rows:
 d=means[m.index(row['method_a']),:,k.index(row['metric'])]-means[m.index(row['method_b']),:,k.index(row['metric'])]
 t=d.mean()/(d.std(ddof=1)/math.sqrt(20));pv=float(special.betainc(19/2,.5,19/(19+t*t)))
 assert math.isclose(pv,row['p_two_sided'],rel_tol=1e-9,abs_tol=1e-100)
 ps.append(pv)
order=sorted(range(9),key=lambda i:ps[i]);adj=[None]*9;previous=0.
for rank,i in enumerate(order):
 previous=max(previous,min(1.,(9-rank)*ps[i]));adj[i]=previous
for x,y in zip(rows,adj):assert math.isclose(y,x['p_Holm_9'],rel_tol=1e-9,abs_tol=1e-100)
assert sum(v<.05 for v in adj)==8
print('PORTABLE_PRIMARY_STATISTICS_REPRODUCED:9contrasts,8Holm_rejections,60models,20paired_seeds')
