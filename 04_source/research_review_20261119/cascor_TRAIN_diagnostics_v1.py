import pathlib,sys,hashlib,json,os,traceback
R=pathlib.Path('/mnt/project/perun2601449/dp_llm_cfd');D=R/'test/architecture_fair_pilot_20261111_v2/revisions/r_0bc20031d1e9f75f';OUT=R/'results/research_review_20261119_v1/cascor_TRAIN_diagnostics';OUT.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(D))
import torch,common,train,operator_models as models

def main():
 common.setup('cuda',931101);p=R/'results/architecture_budget_main_20261111_v1/run_3e11271df45aff09/main/cascor_s931101/last.pt';sha=lambda:hashlib.sha256(p.read_bytes()).hexdigest();before=sha();core=models.restore_model(torch.load(p,map_location='cuda',weights_only=False)).cuda().eval().core;fixture=train.Fixture(R/'results/architecture_fair_pilot_20261111_v2/pilot_0bc20031d1e9f75f','cuda');total=torch.zeros(192,device='cuda',dtype=torch.float64);within=total.clone();cross=total.clone();norm=total.clone();cases=rows=0
 with torch.no_grad():
  for x,y,w in fixture.iterator():
   e=(y-core(x)).double();ww=w.double().repeat_interleave(64);mu=e.mean(0);total+=(e.square()*ww).sum(0);within+=((e-mu).square()*ww).sum(0);cross+=(e*ww).sum(0);norm+=len(e)*ww;cases+=1;rows+=len(e)
 global_centered=(total-cross.square()/norm).clamp_min(0);between=(global_centered-within).clamp_min(0);assert sha()==before
 fields=[]
 for i,name in enumerate(['u','v','p']):
  z=slice(i*64,(i+1)*64);fields.append(dict(field=name,raw_weighted_energy=float(total[z].sum()),global_centered_weighted_energy=float(global_centered[z].sum()),within_case_weighted_energy=float(within[z].sum()),between_case_weighted_energy=float(between[z].sum()),fraction_global_centered_energy_removed_by_per_case_centering=float(between[z].sum()/global_centered[z].sum())))
 result=dict(status='CASCOR_FROZEN_TRAIN_RESIDUAL_DECOMPOSITION_COMPLETE',seed=931101,checkpoint=str(p),checkpoint_sha256=before,checkpoint_unchanged=True,cases=cases,patch_rows=rows,hidden_units=len(core.units),fields=fields,total_fraction_global_centered_energy_removed=float(between.sum()/global_centered.sum()),scope='Only frozen TRAIN residuals. Weighted variance decomposition diagnoses modes omitted by the per-case covariance objective; no retraining, no TEST and no proof of achievable accuracy with a global objective.',job=os.environ['SLURM_JOB_ID'],torch=torch.__version__,hardware=torch.cuda.get_device_name());common.save(OUT/'complete.json',result);print(json.dumps(result))
if __name__=='__main__':
 try:main()
 except Exception:common.save(OUT/'failure.json',dict(traceback=traceback.format_exc()));raise
