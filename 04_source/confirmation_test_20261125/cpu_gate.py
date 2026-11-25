"""Read-only integrity, CPU restoration, and all-four-GPU validation gate."""
import argparse,json,math,sys,time,gc
from pathlib import Path
from common import SOURCE,read,save,digest,dependencies,configure
def main(mode,root):
 c=read(SOURCE/'config.json');manifest=dependencies(c)
 import evaluate,audit
 assert Path(evaluate.__file__).resolve()==SOURCE/'evaluate.py'
 assert Path(audit.__file__).resolve()==SOURCE/'audit.py'
 if mode=='gate':
  records=[]
  for i in range(4):
   p=root/'preflight_parts'/str(i)/'preflight/ready.json';d=read(p)
   assert d['status']=='CONFIRMATION60_VALIDATION_PREFLIGHT_PASS'
   assert d['models_sha256']==c['models_sha256'] and not d['test_targets_read']
   assert d['environment']['gpu']=='NVIDIA H200'
   assert d['environment']['matmul_tf32'] is False and d['environment']['cudnn_tf32'] is False
   assert len(d['models'])==15
   assert {m['id'] for m in d['models']}=={m['id'] for m in manifest['models'] if m['preflight_group']==i}
   records.extend(d['models'])
  assert len({m['id'] for m in records})==60
  save(root/'preflight/ready.json',dict(status='CONFIRMATION60_VALIDATION_PREFLIGHT_PASS',
   models=60,validation_fields=120,parts=4,models_sha256=c['models_sha256'],
   test_targets_read=False,all_models_validation_parity_pass=True))
  print('ALL60_GPU_VALIDATION_PREFLIGHT_PASS',flush=True);return
 assert mode=='cpu';configure('cpu',911113)
 import torch
 from operator_models import restore_model
 prepared=read(c['prepared_path']);normalization=read(Path(prepared['cache'])/'normalization.json')
 trainrun=Path(c['training_run'])
 audit=read(trainrun/'training_audit.json')
 assert audit['status']=='CONFIRMATION60_TRAINING_AND_VAL_AUDIT_PASS' and audit['complete_models']==60
 start=time.monotonic();rows=[]
 for row in manifest['models']:
  assert digest(row['checkpoint'])==row['checkpoint_sha256']
  cp=torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
  assert cp['normalization']==normalization and cp['model']==row['model']
  assert cp['task']['seed']==row['seed'] and cp['task']['mode']=='main'
  assert cp['task']['time_budget_seconds']==14400
  model=restore_model(cp);ledger=model.parameter_ledger()
  for k in ['learned_or_fitted_total','total_core_coefficients_plus_normalization']:
   assert ledger[k]==row['parameter_ledger'][k]
  for t in model.state_dict().values():
   if isinstance(t,torch.Tensor):assert torch.isfinite(t).all()
  rows.append(dict(id=row['id'],checkpoint_sha256=row['checkpoint_sha256'],CPU_restore_pass=True))
  del cp,model;gc.collect()
 save(root/'cpu_gate.json',dict(status='CONFIRMATION60_CPU_CHECKPOINT_GATE_PASS',models=60,
  restored=rows,elapsed_seconds=time.monotonic()-start,models_sha256=c['models_sha256'],
  source_integrity_sha256=digest(SOURCE/'source_integrity.json'),test_targets_read=False))
 print('ALL60_CPU_CHECKPOINT_GATE_PASS',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['cpu','gate'],required=True)
 p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.mode,a.root)
