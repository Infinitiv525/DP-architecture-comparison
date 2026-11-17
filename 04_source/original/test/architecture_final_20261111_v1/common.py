"""Frozen final-evaluation dependencies; no optimizer or model selection."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

SOURCE=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def fingerprint(path):
    s=Path(path).stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino)


def dependencies(config):
    for name,sha in config['main_source_sha256'].items():assert digest(Path(config['main_source'])/name)==sha
    spec=importlib.util.spec_from_file_location('frozen_main_common',Path(config['main_source'])/'common.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    mc=read(Path(config['main_source'])/'config.json');e,c=module.engine(mc);module.manifest(mc)
    for path,sha in config['legacy_source_sha256'].items():assert digest(path)==sha
    for path,sha in config['metadata_sha256'].items():assert digest(path)==sha
    for path,expected in config['data_fingerprints'].items():assert fingerprint(path)==expected
    assert digest(SOURCE/'models.json')==config['models_sha256']
    return e,c,read(SOURCE/'models.json')


def legacy_paths(config):
    for folder in config['legacy_paths']:sys.path.insert(0,folder)
