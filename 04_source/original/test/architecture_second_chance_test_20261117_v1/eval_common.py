"""Frozen evaluation imports. No training, tuning, or use of old cohort results."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import torch

CODE_SOURCE=Path(__file__).resolve().parent
SOURCE=CODE_SOURCE


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def fingerprint(path):
    s=Path(path).stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino)


def legacy_paths(config):
    for folder in config['legacy_paths']:sys.path.insert(0,folder)
    sys.path.insert(0,config['field_source'])
    metrics=Path(config['metric_source']);spec=importlib.util.spec_from_file_location('data_metrics',metrics)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)


def dependencies(config):
    for folder,names in [(config['main_source'],config['main_source_sha256']),
                         (config['field_source'],config['field_source_sha256'])]:
        for name,sha in names.items():assert digest(Path(folder)/name)==sha
    assert digest(config['metric_source'])==config['metric_source_sha256']
    for path,sha in config['legacy_source_sha256'].items():assert digest(path)==sha
    for path,sha in config['metadata_sha256'].items():assert digest(path)==sha
    for path,expected in config['data_fingerprints'].items():assert fingerprint(path)==expected
    assert digest(SOURCE/'models.json')==config['models_sha256']
    manifest=read(SOURCE/'models.json');assert manifest['status']=='SECOND_CHANCE4_FIXED_TEST_MANIFEST'
    assert len(manifest['models'])==4 and not manifest['selection_uses_test']
    legacy_paths(config)
    return manifest


def configure(device,seed):
    torch.set_num_threads(4);torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if device=='cuda':
        assert torch.cuda.is_available() and 'H200' in torch.cuda.get_device_name() and torch.cuda.device_count()==1
        torch.cuda.manual_seed_all(seed)
