import pathlib,sys,json,time,hashlib,traceback,os,math
R=pathlib.Path('/mnt/project/perun2601449/dp_llm_cfd');OUT=R/'results/research_review_20261119_v1/synthetic_diagnostics';OUT.mkdir(parents=True,exist_ok=True);D=R/'test/architecture_fair_pilot_20261111_v2/revisions/r_0bc20031d1e9f75f';sys.path.insert(0,str(D))
import torch,numpy as np
import common,operator_models as models,weighted_fit as fit

def mse(core,batches):
 with torch.no_grad():return float(torch.stack([(core(x)-y).square().mean() for x,y,w in batches]).mean())
def batches(x,y):return [(x[i:i+128],y[i:i+128],torch.ones(3)) for i in range(0,len(x),128)]
def run():
 common.setup('cpu',191101);m=models.mathematical_cores();results=[]
 for seed in [191101,191102,191103]:
  common.setup('cpu',seed);x=torch.randn(512,665);xv=torch.randn(256,665);coeff=torch.linspace(.5,1.5,192)[None]
  # Positive ELM representation check: the exact target is inside the frozen random basis.
  elm=m.ELM(665,32);elm.beta=torch.zeros(33,192);teacher=torch.randn(33,192)*.1
  with torch.no_grad():y=elm.basis(x)@teacher;yv=elm.basis(xv)@teacher
  rows=batches(x,y);metadata=fit.fit_output(elm,lambda:iter(rows),1e-8);results.append(dict(test='ELM_in_basis_positive_control',seed=seed,train_mse=mse(elm,rows),validation_mse=mse(elm,batches(xv,yv)),fit=metadata));assert results[-1]['validation_mse']<1e-7
  # Independent high-frequency target: width sensitivity is descriptive, not an acceptance gate.
  y=(torch.sin(5*x[:,:1])+torch.cos(4*x[:,1:2]))*coeff;yv=(torch.sin(5*xv[:,:1])+torch.cos(4*xv[:,1:2]))*coeff
  for width in [16,64,256]:
   torch.manual_seed(seed);core=m.ELM(665,width);core.beta=torch.zeros(width+1,192);rows=batches(x,y);metadata=fit.fit_output(core,lambda:iter(rows),1e-6);results.append(dict(test='ELM_independent_smooth_target',seed=seed,width=width,train_mse=mse(core,rows),validation_mse=mse(core,batches(xv,yv))))
  # White-box positive witness: one candidate is deliberately in the frozen candidate bank.
  g=torch.Generator(device='cpu').manual_seed(seed+1);w=(torch.rand(16,665,generator=g)*2-1)*.5/math.sqrt(665);b=(torch.rand(16,generator=g)*2-1)*.5
  y=torch.tanh(torch.nn.functional.linear(x,w[:1],b[:1]))*coeff;rows=batches(x,y);core=m.SCN(665,0);core.beta=torch.zeros(1,192);fit.fit_output(core,lambda:iter(rows),1e-8);before=mse(core,rows);node,meta=fit.scn_candidate(core,lambda:iter(rows),pool=16,seed=seed+1);assert node is not None;fit.append_scn(core,node);fit.fit_output(core,lambda:iter(rows),1e-8);after=mse(core,rows);assert after<before;results.append(dict(test='SCN_known_admissible_candidate_positive_control',seed=seed,before_mse=before,after_mse=after,metadata=meta,scope='Constructed algebraic witness, not an unbiased PDE or architecture benchmark'))
  # Case-constant nonlinear dependence: original per-case centering removes this entire residual mode.
  rows=[];features=[];targets=[]
  for value in torch.linspace(-1,1,8):
   xx=torch.zeros(128,665);xx[:,0]=value;yy=torch.tanh(3*value)*coeff.expand(128,-1);rows.append((xx,yy,torch.ones(3)));features.append(torch.tanh(3*value).expand(128));targets.append(yy)
  core=m.CascadeCorrelation(665,0);core.beta=torch.zeros(666,192);fit.fit_output(core,lambda:iter(rows),1e-6);before=mse(core,rows)
  with torch.no_grad():residual=torch.cat([y-core(x) for x,y,w in rows]);h=torch.cat(features);global_cov=float(((h-h.mean())[:,None]*(residual-residual.mean(0))).sum(0).abs().sum())
  node,meta=fit.cascade_candidate(core,lambda:iter(rows),pool=8,epochs=1);core.add(node);fit.fit_output(core,lambda:iter(rows),1e-6);results.append(dict(test='CasCor_case_constant_nonlinear_mode',seed=seed,before_mse=before,after_one_random_or_learned_node_mse=mse(core,rows),candidate_metadata=meta,global_teacher_covariance=global_cov,interpretation='Per-case centering removes errors constant inside a case; a nonzero global witness does not prove this is the sole cause of FlowBench results'))
  # Within-case variation remains visible to exactly the same original candidate code.
  rows=[]
  for shift in [-.8,-.2,.2,.8]:
   xx=torch.zeros(128,665);xx[:,0]=torch.linspace(-1,1,128);xx[:,1]=shift;yy=torch.tanh(3*xx[:,:1])*coeff;rows.append((xx,yy,torch.ones(3)))
  core=m.CascadeCorrelation(665,0);core.beta=torch.zeros(666,192);fit.fit_output(core,lambda:iter(rows),1e-6);before=mse(core,rows);node,meta=fit.cascade_candidate(core,lambda:iter(rows),pool=8,epochs=1);core.add(node);fit.fit_output(core,lambda:iter(rows),1e-6);results.append(dict(test='CasCor_within_case_nonlinear_mode',seed=seed,before_mse=before,after_mse=mse(core,rows),candidate_metadata=meta))
  common.save(OUT/'progress.json',dict(completed_seed=seed,rows=len(results)))
 summary=dict(status='SYNTHETIC_DIAGNOSTICS_COMPLETE',scope='Post-experiment implementation/representation diagnostics only; no CFD/TEST data, no changes to main models',seeds=[191101,191102,191103],input_dimensions=665,output_dimensions=192,results=results,source_sha256={name:hashlib.sha256((D/name).read_bytes()).hexdigest() for name in ['weighted_fit.py','operator_models.py']},job=os.environ['SLURM_JOB_ID']);common.save(OUT/'complete.json',summary);print(json.dumps({'status':summary['status'],'tests':len(results)}))
if __name__=='__main__':
 try:run()
 except Exception as e:common.save(OUT/'failure.json',dict(error=type(e).__name__,traceback=traceback.format_exc()));raise
