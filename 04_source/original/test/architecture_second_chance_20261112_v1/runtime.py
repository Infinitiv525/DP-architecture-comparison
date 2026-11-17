"""Frozen original dependencies and exact deterministic data-cursor recovery."""
import hashlib
import importlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text())


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def dependencies():
    policy=read(HERE/'policy.json')
    for source,hashes in [(policy['dependency_source'],policy['dependency_sha256']),
                          (policy['original_main_source'],policy['original_main_sha256']),
                          (policy['inspection_source'],policy['inspection_source_sha256'])]:
        for name,sha in hashes.items():assert digest(Path(source)/name)==sha,name
    assert digest(Path(policy['fixture_root'])/'fixture/ready.json')==policy['fixture_ready_sha256']
    assert digest(policy['inspection_ready'])==policy['inspection_ready_sha256']
    assert digest(HERE/'parents.json')==policy['parents_sha256']
    parents=read(HERE/'parents.json')
    assert parents['status']=='FOUR_LAST_CHECKPOINTS_AND_ADAM_STATES_VERIFIED'
    assert [(r['model'],r['seed']) for r in parents['tasks']]==[
        ('chebkan',931101),('chebkan',931102),('rbf',931101),('rbf',931102)]
    sys.path.insert(0,policy['dependency_source'])
    return policy,parents,importlib.import_module('common'),importlib.import_module('train'),importlib.import_module('operator_models')


def recovered_cursor(counts,seed,updates):
    import numpy as np
    import torch
    counts=np.asarray(counts,dtype='int64');assert counts.shape==(640,) and int(counts.sum())==updates
    full,position=divmod(updates,640);epoch=full+1
    order=torch.randperm(640,generator=torch.Generator(device='cpu').manual_seed(seed+epoch)).tolist()
    expected=np.full(640,full,dtype='int64');expected[order[:position]]+=1
    assert np.array_equal(counts,expected)
    return epoch,position,order


def optimizer_steps(optimizer):
    return [dict(step=float(state['step']),dtype=str(state['step'].dtype)) for state in optimizer.state.values()]


def check_parents(row):
    for path,sha in [(row['original_last'],row['original_last_sha256']),
                     (row['original_best'],row['original_best_sha256']),
                     (row['original_complete'],row['original_complete_sha256']),
                     (row['original_closeout'],row['original_closeout_sha256'])]:
        assert digest(path)==sha,path
