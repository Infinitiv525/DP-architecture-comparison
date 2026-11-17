import hashlib
import importlib.util
import json
from pathlib import Path
import sys

SOURCE=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for value in iter(lambda:f.read(1024*1024),b''):h.update(value)
    return h.hexdigest()


def engine(config):
    folder=Path(config['calibration_source'])
    for name,sha in config['calibration_source_sha256'].items():
        if digest(folder/name)!=sha:raise ValueError('Frozen calibration source mismatch: '+name)
    spec=importlib.util.spec_from_file_location('calibration_engine',folder/'train.py');module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module;spec.loader.exec_module(module)
    c=read(folder/'config.json');module.dependencies(c)
    return module,c
