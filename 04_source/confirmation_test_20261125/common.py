"""Frozen confirmation evaluation imports and explicit mixed-family restoration."""
import hashlib,importlib,importlib.util,json,sys,types
from pathlib import Path
import torch
SOURCE=Path(__file__).resolve().parent
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fingerprint(path):
 s=Path(path).stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino)
def legacy_paths(c):
 for folder in c['legacy_paths']:sys.path.insert(0,folder)
 sys.path.insert(0,c['field_source'])
 p=Path(c['metric_source']);sp=importlib.util.spec_from_file_location('data_metrics',p)
 m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
def dependencies(c):
 frozen=read(SOURCE/'source_integrity.json')
 for name,h in frozen['files'].items():assert digest(SOURCE/name)==h,name
 for key in ['legacy_source_sha256','metadata_sha256','extra_source_sha256']:
  for p,h in c[key].items():assert digest(p)==h,p
 for name,h in c['field_source_sha256'].items():assert digest(Path(c['field_source'])/name)==h
 for p,v in c['data_fingerprints'].items():assert fingerprint(p)==v,p
 assert digest(SOURCE/'models.json')==c['models_sha256']
 manifest=read(SOURCE/'models.json')
 assert manifest['status']=='CONFIRMATION60_MANIFEST_FROZEN'
 assert len(manifest['models'])==60 and not manifest['selection_uses_test']
 legacy_paths(c)
 primary=importlib.import_module('operator_models')
 # Keep the original two restoration functions intact; dispatch by checkpoint family.
 assert hasattr(primary,'PackedFieldOperator') and hasattr(primary,'restore_model')
 sys.path.insert(0,c['dense_source'])
 dense=importlib.import_module('models_fno')
 proxy=types.ModuleType('operator_models')
 def restore(cp):
  assert cp['model'] in ['tfno','fno','realnpu']
  return dense.restore_model(cp) if cp['model']=='fno' else primary.restore_model(cp)
 proxy.restore_model=restore
 sys.modules['operator_models']=proxy
 sys.path.insert(0,str(SOURCE))  # Evaluation and audit must come from this frozen package.
 return manifest
def configure(device,seed):
 torch.set_num_threads(4);torch.manual_seed(seed)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if device=='cuda':
  assert torch.cuda.is_available() and 'H200' in torch.cuda.get_device_name() and torch.cuda.device_count()==1
  torch.cuda.manual_seed_all(seed)
