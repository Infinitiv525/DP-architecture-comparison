"""All-VAL preflight parts, gated TEST groups, and independent CPU output audit."""
import argparse,json,os,time,traceback
from pathlib import Path
from common import SOURCE,read,save,digest,dependencies,configure
def main(mode,root,index):
 c=read(SOURCE/'config.json');manifest=dependencies(c)
 if mode=='audit':
  configure('cpu',911115)
  from audit import main as audit
  # audit.main calls dependencies once itself; remove the dispatch proxy first.
  del __import__('sys').modules['operator_models']
  audit(c,root);return
 configure('cuda',911114)
 import torch,numpy as np
 assert torch.__version__=='2.12.0+cu130' and np.__version__=='2.5.4'
 import evaluate as evaluation_module
 assert Path(evaluation_module.__file__).resolve()==SOURCE/'evaluate.py'
 from evaluate import preflight,evaluate
 if mode=='preflight':
  assert read(root/'cpu_gate.json')['status']=='CONFIRMATION60_CPU_CHECKPOINT_GATE_PASS'
  part=root/'preflight_parts'/str(index)
  models=[m for m in manifest['models'] if m['preflight_group']==index]
  assert len(models)==15
  preflight(c,part,models)
 else:
  assert mode=='test'
  ready=read(root/'preflight/ready.json')
  assert ready['status']=='CONFIRMATION60_VALIDATION_PREFLIGHT_PASS'
  assert ready['models_sha256']==c['models_sha256'] and ready['models']==60
  models=[m for m in manifest['models'] if m['group']==index]
  assert len(models)==5
  evaluate(c,root,models,index)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['preflight','test','audit'],required=True)
 p.add_argument('--root',type=Path,required=True);p.add_argument('--index',type=int,default=0);a=p.parse_args()
 try:main(a.mode,a.root,a.index)
 except Exception:
  save(a.root/(a.mode+'_failure_'+str(a.index)+'.json'),dict(error=traceback.format_exc(),mode=a.mode,index=a.index))
  raise
