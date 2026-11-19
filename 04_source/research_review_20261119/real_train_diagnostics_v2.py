import pathlib,sys,json,hashlib,os,traceback,math,time
R=pathlib.Path('/mnt/project/perun2601449/dp_llm_cfd');D=R/'test/architecture_fair_pilot_20261111_v2/revisions/r_0bc20031d1e9f75f';OUT=R/'results/research_review_20261119_v1/real_TRAIN_diagnostics_v2';OUT.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(D))
import torch,numpy as np
import common,train,operator_models as models
MAIN=R/'results/architecture_budget_main_20261111_v1/run_3e11271df45aff09/main'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
 common.setup('cuda',931101);assert 'H200' in torch.cuda.get_device_name();policy=json.loads((R/'results/architecture_budget_main_20261111_v1/run_3e11271df45aff09/main/scn_s931101/complete.json').read_text());checkpoint=MAIN/'scn_s931101/last.pt';before=sha(checkpoint);saved=torch.load(checkpoint,map_location='cuda',weights_only=False);model=models.restore_model(saved).cuda().eval();grown_units=len(model.core.weight);fixture=train.Fixture(R/'results/architecture_fair_pilot_20261111_v2/pilot_0bc20031d1e9f75f','cuda');core=model.core;g=torch.Generator(device='cuda').manual_seed(931101+grown_units+1);rows=[]
 with torch.no_grad():
  for radius in [.5,1.,2.,4.]:
   w=(torch.rand(16,665,device='cuda',generator=g)*2-1)*radius/math.sqrt(665);b=(torch.rand(16,device='cuda',generator=g)*2-1)*radius;cross=torch.zeros(16,192,dtype=torch.float64,device='cuda');norm=torch.zeros(16,3,dtype=torch.float64,device='cuda');energy=torch.zeros(192,dtype=torch.float64,device='cuda')
   for x,y,weight in fixture.iterator():
    h=torch.tanh(torch.nn.functional.linear(x,w,b)).double();residual=(y-core(x)).double();weights=weight.repeat_interleave(64).double();cross+=h.T@(residual*weights);norm+=h.square().sum(0)[:,None]*weight.double()[None];energy+=(residual.square()*weights).sum(0)
   projection=cross.square()/norm.repeat_interleave(64,1).clamp_min(1e-30);active=energy>1e-20;ratio=projection[:,active]/energy[active][None];thresholds=[]
   for relaxation in [.9,.99,.999,.9999,.99999,.9999999]:
    xi=projection-(1-relaxation)*energy[None];valid=(xi[:,active]>0).all(1);aggregate=xi[:,active].sum(1)>0;thresholds.append(dict(relaxation=relaxation,required_projection_fraction=1-relaxation,original_all_output_admissible_candidates=int(valid.sum()),hypothetical_aggregate_admissible_candidates=int(aggregate.sum()),counterfactual_note='Criterion comparison only; aggregate acceptance is not the tested SCN, no model was changed or fitted'))
   worst=ratio.min(1).values;best=int(worst.argmax());active_ids=torch.where(active)[0];bad=torch.argsort(ratio[best])[:12];rows.append(dict(radius=radius,unique_candidates=16,active_outputs=int(active.sum()),best_candidate_minimum_projection_fraction=float(worst[best]),best_candidate_blocking_outputs=[dict(output=int(active_ids[j]),field=int(active_ids[j])//64,patch_offset=int(active_ids[j])%64,projection_fraction=float(ratio[best,j]),residual_energy=float(energy[active_ids[j]])) for j in bad],thresholds=thresholds))
   common.save(OUT/'progress.json',dict(stage='SCN_full_TRAIN_admissibility',radius=radius))
 assert sha(checkpoint)==before
 elm_path=MAIN/'elm_s931101/best.pt';es=sha(elm_path);elm=models.restore_model(torch.load(elm_path,map_location='cuda',weights_only=False)).cuda().eval();indices=list(range(0,640,10));patches=torch.arange(0,4096,32,device='cuda')
 with torch.no_grad():
  x=torch.cat([fixture.tables['train'][0][i,patches] for i in indices]);h=torch.tanh(torch.nn.functional.linear(x,elm.core.weight,elm.core.bias));spectrum=torch.linalg.svdvals(h[:1024,:256].double());shaped=h.reshape(len(indices),len(patches),-1);stats=dict(sampled_TRAIN_cases=len(indices),sampled_patches=len(x),features=h.shape[1],fraction_absolute_activation_above_099=float((h.abs()>.99).float().mean()),median_feature_std=float(h.std(0).median()),mean_within_case_variance=float(shaped.var(1).mean()),total_variance=float(h.var(0).mean()),spectrum_submatrix_shape=[1024,256],submatrix_singular_values=spectrum.cpu().tolist(),submatrix_rank_relative_1e6=int((spectrum>spectrum[0]*1e-6).sum()),scope='Diagnostic sample of frozen ELM activations, not a full feature-rank proof or a new training')
 assert sha(elm_path)==es
 original=[]
 for method in ['elm','scn','cascor']:
  for seed in range(931101,931106):
   p=MAIN/(method+'_s'+str(seed))/'complete.json';v=json.loads(p.read_text());original.append(dict(method=method,seed=seed,stop_reason=v['stop_reason'],grown_units=v['final_grown_units'],candidate_history=v['candidate_history'],selected_parameters=v['selected_parameter_ledger'],last_parameters=v['last_parameter_ledger']))
 result=dict(status='READ_ONLY_TRAIN_DIAGNOSTICS_COMPLETE',scope='Frozen real TRAIN only; no TEST read, no optimizer and no changes to original models',SCN=dict(checkpoint=str(checkpoint),sha256=before,seed=931101+grown_units+1,frozen_hidden_units=grown_units,unique_candidate_vectors=64,criterion_evaluations=384,all_640_TRAIN_cases_used=True,rows=rows),ELM=dict(checkpoint=str(elm_path),sha256=es,activation_diagnostics=stats),original_runs=original,job=os.environ['SLURM_JOB_ID'],hardware=torch.cuda.get_device_name(),torch=torch.__version__,numpy=np.__version__);common.save(OUT/'complete.json',result);print(json.dumps({'status':result['status'],'SCN_valid_at_most_relaxed':sum(r['thresholds'][-1]['original_all_output_admissible_candidates'] for r in rows),'ELM_saturation_fraction':stats['fraction_absolute_activation_above_099']}))
if __name__=='__main__':
 try:run()
 except Exception as e:common.save(OUT/'failure.json',dict(error=type(e).__name__,traceback=traceback.format_exc()));raise
