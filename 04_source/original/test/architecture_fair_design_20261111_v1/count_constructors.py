"""CPU construction/accounting only. A hook forbids every neural forward call."""
import gc
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from operator_models import PackedFieldOperator

HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/project/perun2601449/dp_llm_cfd')


def main():
    torch.set_num_threads(1);assert not torch.cuda.is_available()
    forward_calls=[]
    def forbid_forward(module,args):
        forward_calls.append(type(module).__name__)
        raise RuntimeError('Forward/inference is forbidden in constructor-only preparation')
    handle=torch.nn.modules.module.register_module_forward_pre_hook(forbid_forward)
    prepared=json.loads((ROOT/'results/ldc_pinn_prepare_20261104_v25/prepared.json').read_text())
    normalization=json.loads((Path(prepared['cache'])/'normalization.json').read_text())
    plan=json.loads((HERE/'capacity_plan.json').read_text());records=[]
    try:
        for recipe in plan['models']:
            torch.manual_seed(931001)
            model=PackedFieldOperator(recipe['model'],recipe,normalization,capacity_construction=True)
            ledger=model.parameter_ledger();assert ledger['learned_or_fitted_total']==recipe['planned_total_fitted_scalars'],(recipe,ledger)
            assert ledger['total_core_coefficients_plus_normalization']==recipe['planned_total_coefficient_scalars'],(recipe,ledger)
            records.append(dict(model=recipe['model'],width=recipe['width'],ledger=ledger,model_state='UNTRAINED_PLANNED_FULL_CAPACITY'))
            del model;gc.collect()
    finally:handle.remove()
    assert not forward_calls
    print(json.dumps(dict(status='ALL9_UNTRAINED_CONSTRUCTOR_COUNTS_VERIFIED',models=records,
        forward_calls=0,training_updates=0,simulation_fields_read=0,GPU_allocated=False,
        purpose='Model construction and scalar accounting only; not an inference/training experiment',
        learned_or_fitted_reference=plan['reference_total_fitted_real_scalars'],torch=torch.__version__)))


if __name__=='__main__':main()
