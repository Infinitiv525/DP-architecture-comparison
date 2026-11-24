"""Validate frozen imports, then bind the dense-FNO constructor only."""
import hashlib
import importlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def install_fno_models():
    import models_fno
    sys.modules['operator_models'] = models_fno
    return models_fno


def training_dependencies():
    policy = json.loads((HERE/'policy.json').read_text())
    source = Path(policy['dependency_source'])
    for name, sha in policy['dependency_sha256'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == sha, name
    for name, sha in policy['reference_main_sha256'].items():
        assert hashlib.sha256((Path(policy['reference_main_source'])/name).read_bytes()).hexdigest() == sha, name
    ready = Path(policy['fixture_root'])/'fixture/ready.json'
    assert hashlib.sha256(ready.read_bytes()).hexdigest() == policy['fixture_ready_sha256']
    sys.path.insert(0, str(source))
    common = importlib.import_module('common')
    model = install_fno_models()

    def fno_recipe(name):
        assert name == 'fno'
        return model.recipe()

    def patch_prediction(core, name, x):
        assert name == 'fno' and x.shape == (4096,665)
        value = core(x.T.reshape(1,665,64,64))
        return value.permute(0,2,3,1).reshape(-1,192)

    common.recipe = fno_recipe
    common.patch_prediction = patch_prediction
    training = importlib.import_module('train')
    return policy, common, training, model, importlib.import_module('weighted_fit')
