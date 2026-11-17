"""Arithmetic only: parameter accounting and a naive point-ELM memory estimate."""
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    plan=json.loads((HERE/'plan.json').read_text());budget=plan['capacity_proposal']['reference_total_core_coefficients_plus_normalization']
    core_budget=budget-10;hidden=core_budget//3-1
    assert (hidden+1)*3==core_budget
    tolerance=plan['capacity_proposal']['matched_band_relative_tolerance']
    result=dict(kind='STATIC_ARITHMETIC_NOT_A_MODEL_TEST',reference_fitted_real_scalars=budget,
        matched_lower=math.ceil(budget*(1-tolerance)),matched_upper=math.floor(budget*(1+tolerance)),
        naive_point_elm=dict(input_features=57,output_features=3,hidden_units=hidden,
            fitted_beta_scalars=(hidden+1)*3,data_independent_weights_and_bias=hidden*58,
            total_core_stored_scalars=hidden*58+(hidden+1)*3,
            unchunked_fp32_hidden_activation_bytes=512*512*hidden*4,
            unchunked_fp32_hidden_activation_gib=512*512*hidden*4/(1024**3),
            example_is_old_three_output_point_signature=True,
            note='Old point formulation expanded to the new core budget, not the new192-output operator. Chunking can reduce peak memory; this is not measured runtime or a claim of impossibility for another formulation.'),
        new_training_started=False,new_model_test_started=False)
    (HERE/'rozpocet.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
