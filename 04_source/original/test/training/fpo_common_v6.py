"""Non-Torch utilities shared by CPU preparation and GPU training."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/project/perun2601449/dp_llm_cfd')
WORK = ROOT / 'results/fpo_pinn_stabilization_20261031_v6'
PILOT = ROOT / 'results/fpo_pinn_pilot_20261030_v1'
REV = '937c672604d4c9f39aceedf1fa2f0a603695264b'


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def verify():
    frozen = json.loads((WORK / 'freeze.json').read_text())
    for path, checksum in frozen['files'].items():
        if digest(path) != checksum:
            raise ValueError('Frozen source or protocol changed: ' + path)


def coords(k, y, x):
    return np.column_stack((4 + (x + .5) * (16 / 1024),
                            6 + (y + .5) * (4 / 256), (k - 3) * .05)).astype('float32')
