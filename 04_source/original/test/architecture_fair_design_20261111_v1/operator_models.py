"""DRAFT patch-conditioned operator formulations of the agreed nine families.

Reuses our frozen mathematical cores with a common input/output layout.
No fitting functions are called here. Dense/streamed training is not implemented
by this constructor module and no experiment is authorized automatically.
"""
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import torch
from torch import nn
from field_adapter import FullFieldAdapter

FROZEN=Path('/mnt/project/perun2601449/dp_llm_cfd/test/architecture_pilot_20261110_v1/revisions/r_e9dafab653bac9e5/models.py')
FROZEN_SHA='2ee19101e62a09867807e36d1509f93f9afc7f262f5c13270aa4974fe6894f56'


def mathematical_cores():
    assert hashlib.sha256(FROZEN.read_bytes()).hexdigest()==FROZEN_SHA
    name='frozen_fair_mathematical_cores'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,FROZEN);module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]


def core_model(name,recipe,dynamic=None,capacity_construction=False):
    d,o=FullFieldAdapter.inputs,FullFieldAdapter.outputs;dynamic=dynamic or {}
    if name=='tfno':
        from neuralop.models import TFNO
        return TFNO(n_modes=(32,32),n_layers=5,hidden_channels=48,in_channels=d,out_channels=o,
            projection_channel_ratio=4,lifting_channel_ratio=2,factorization='Tucker',rank=.1,
            implementation='factorized',positional_embedding='grid',fno_block_precision='full')
    m=mathematical_cores();width=recipe['width'];degree=recipe.get('degree',3)
    if name=='chebkan':
        core=m.ChebKAN(d,width,degree);core.layers[-1]=m.ChebLayer(width,o,degree)
    elif name in ['pinet','dendrite','realnpu','rbf']:
        constructor={'pinet':m.PiNet,'dendrite':m.DendriteNet,'realnpu':m.RealNPU,'rbf':m.RBF}[name]
        core=constructor(d,width,degree) if name in ['pinet','dendrite'] else constructor(d,width)
        core.output=nn.Linear(width,o)
    elif name=='elm':
        core=m.ELM(d,width);core.beta=torch.zeros(width+1,o)
    elif name=='scn':
        units=width if capacity_construction else dynamic.get('scn_units',0)
        core=m.SCN(d,units);core.beta=torch.zeros(units+1,o)
    elif name=='cascor':
        units=width if capacity_construction else dynamic.get('cascor_units',0)
        core=m.CascadeCorrelation(d,units);core.beta=torch.zeros(d+units+1,o)
    else:raise ValueError(name)
    return core


class PackedFieldOperator(nn.Module):
    def __init__(self,name,recipe,normalization,dynamic=None,capacity_construction=False):
        super().__init__();self.name=name;self.recipe=recipe;self.normalization=normalization
        self.adapter=FullFieldAdapter(normalization)
        self.core=core_model(name,recipe,dynamic,capacity_construction)

    def raw_patch_prediction(self,known):
        value=self.adapter.encode(known['geometry'],known['re'])
        if value.shape[0]!=1:raise ValueError('Current operator contract is one complete case per call')
        if self.name=='tfno':return self.core(value)
        batch,_,ny,nx=value.shape
        result=self.core(value.permute(0,2,3,1).reshape(-1,self.adapter.inputs))
        return result.reshape(batch,ny,nx,self.adapter.outputs).permute(0,3,1,2)

    def raw_native_prediction(self,known):
        value=self.adapter.decode_raw(self.raw_patch_prediction(known))
        return value.permute(0,2,3,1)[0]

    def forward(self,known):
        value=self.adapter.decode_raw(self.raw_patch_prediction(known))
        value=self.adapter.project_known_constraints(value,known['fluid'])
        return value.permute(0,2,3,1)[0]

    def parameter_ledger(self):
        real=lambda value:value.numel()*(2 if value.is_complex() else 1)
        core=self.core;gradient=fitted=selected=fixed=0
        if self.name in ['elm','scn','cascor']:
            fitted=real(core.beta)
            if self.name=='elm':fixed=real(core.weight)+real(core.bias)
            elif self.name=='scn':selected=real(core.weight)+real(core.bias)
            else:fitted+=sum(real(p) for p in core.parameters())
        else:gradient=sum(real(p) for p in core.parameters())
        normalization=10
        return dict(trainable_real_scalars=gradient,data_fitted_frozen_real_scalars=fitted+normalization,
            data_selected_frozen_real_scalars=selected,data_independent_core_real_scalars=fixed,
            learned_or_fitted_total=gradient+fitted+selected+normalization,
            total_core_coefficients_plus_normalization=gradient+fitted+selected+fixed+normalization,
            common_adapter_buffer_bytes=sum(b.numel()*b.element_size() for b in self.adapter.buffers()),
            core_tensor_bytes=sum(t.numel()*t.element_size() for t in list(core.parameters())+list(core.buffers())),
            accounting_only=True)


def restore_model(checkpoint):
    model=PackedFieldOperator(checkpoint['model'],checkpoint['recipe'],checkpoint['normalization'],checkpoint.get('dynamic'))
    state=copy.copy(checkpoint['state'])
    if model.name=='tfno':
        envelope=state.pop('_metadata');assert envelope==model.core._init_kwargs
    model.load_state_dict(state,strict=True);return model.eval()
