"""Shared frozen-source loading for the five-repeat architecture experiment."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

SOURCE=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def engine(config):
    for label in ['calibration','review']:
        folder=Path(config[label+'_source'])
        for name,sha in config[label+'_source_sha256'].items():
            if digest(folder/name)!=sha:raise ValueError('Frozen dependency mismatch: '+label+'/'+name)
    module=load_module('calibration_engine',Path(config['calibration_source'])/'train.py')
    c=read(Path(config['calibration_source'])/'config.json')
    if 'run_seed' in config:c['calibration_seed']=int(config['run_seed'])
    module.dependencies(c)
    return module,c


def manifest(config):
    assert digest(SOURCE/'selection.json')==config['selection_sha256']
    value=read(SOURCE/'selection.json')
    assert value['status']=='ARCHITECTURE_SELECTION_FROZEN' and len(value['selected'])==9
    assert value['final_seeds']==[921101,921102,921103,921104,921105]
    assert len(value['tasks'])==45 and len({r['id'] for r in value['tasks']})==45
    assert all(r['status']=='MODEL_ARTIFACT_AUDIT_PASS' for r in value['audit_evidence'])
    return value
