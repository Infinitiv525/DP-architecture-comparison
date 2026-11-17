"""Native-field pilot: immutable dependencies, TRAIN/VAL only and atomic reports."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path('/mnt/project/perun2601449/dp_llm_cfd')
PILOT = ROOT/'test/architecture_pilot_20261110_v1/revisions/r_e9dafab653bac9e5'
DATA_SHA = '3a004a845e1bc83f9fea41db3732b494a132358e9b24838bd6cdaf52dc75458a'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data_module():
    path = PILOT/'data_metrics.py'
    assert digest(path) == DATA_SHA
    spec = importlib.util.spec_from_file_location('native_fair_data', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


def load_data(train=1, validation=1):
    config = dict(root=str(ROOT), prepared_path='results/ldc_pinn_prepare_20261104_v25/prepared.json',
                  split_path='results/ldc_pinn_prepare_20261104_v25/split.json',
                  train_cases=train, validation_cases=validation, test_targets_allowed=False)
    data = data_module().PilotData(config, device='cpu')
    data.plan['case_selection'] = f'first {train} TRAIN / first {validation} VAL in frozen split order; no target-based selection'
    return data


def setup(device, seed):
    torch.set_num_threads(4)
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == 'cuda':
        assert torch.cuda.is_available()
        torch.cuda.manual_seed_all(seed)


def sync(device):
    if device == 'cuda': torch.cuda.synchronize()


def timer(function, device, repeats=3):
    samples=[]
    for _ in range(repeats):
        sync(device); start=time.perf_counter(); function(); sync(device)
        samples.append(time.perf_counter()-start)
    return dict(seconds=samples, median_seconds=float(np.median(samples)))


def recipe(name):
    row = next(x for x in read(HERE/'capacity_plan.json')['models'] if x['model']==name)
    return dict(width=row['width'], degree=row['degree'] or 3)


def case_on_device(case, device):
    return {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in case.items()}


def rows(model, case, target):
    with torch.no_grad():
        x=model.adapter.encode(case['geometry'],case['re']).permute(0,2,3,1).reshape(-1,665)
        y=model.adapter.target_patches(target).permute(0,2,3,1).reshape(-1,192)
        energy=torch.stack((target[...,:2].square().sum(),target[...,:2].square().sum(),target[...,2].square().sum())).clamp_min(1e-12)
        std=model.adapter.target_std.flatten()
        weight=.5*std.square()/energy
    return x,y,weight


def patch_loss(prediction, target, channel_weight):
    """Exactly the raw-native case-balanced loss, not a latent surrogate."""
    return ((prediction-target).reshape(-1,3,64).square().sum((0,2))*channel_weight).sum()


def patch_prediction(core, name, x):
    if name == 'tfno':
        value=core(x.T.reshape(1,665,64,64))
        return value.permute(0,2,3,1).reshape(-1,192)
    return core(x)
