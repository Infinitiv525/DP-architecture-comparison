"""Dense FNO backbone inside the unchanged native-field interface."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
POLICY = json.loads((HERE/'policy.json').read_text())
FIELD = Path(POLICY['dependency_source'])
PATH = FIELD/'operator_models.py'
assert hashlib.sha256(PATH.read_bytes()).hexdigest() == POLICY['dependency_sha256']['operator_models.py']
sys.path.insert(0, str(FIELD))
spec = importlib.util.spec_from_file_location('dense_fno_unchanged_interface', PATH)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
from field_adapter import FullFieldAdapter


def recipe():
    return json.loads((HERE/'recipe.json').read_text())


class PackedFieldOperator(legacy.PackedFieldOperator):
    def __init__(self, name, recipe, normalization, dynamic=None, capacity_construction=False):
        nn.Module.__init__(self)
        assert name == 'fno' and not dynamic and not capacity_construction
        self.name = name
        self.recipe = recipe
        self.normalization = normalization
        self.adapter = FullFieldAdapter(normalization)
        from neuralop.models import FNO
        kwargs = {k:v for k,v in recipe.items() if k != 'note'}
        kwargs['n_modes'] = tuple(kwargs['n_modes'])
        self.core = FNO(**kwargs)
        assert self.core.factorization is None

    def raw_patch_prediction(self, known):
        value = self.adapter.encode(known['geometry'], known['re'])
        assert value.shape == (1,665,64,64)
        return self.core(value)


def restore_model(checkpoint):
    assert checkpoint['model'] == 'fno' and checkpoint['recipe'] == recipe()
    model = PackedFieldOperator('fno', checkpoint['recipe'], checkpoint['normalization'], checkpoint.get('dynamic'))
    state = copy.copy(checkpoint['state'])
    envelope = state.pop('_metadata')
    assert envelope == model.core._init_kwargs
    model.load_state_dict(state, strict=True)
    return model.eval()
