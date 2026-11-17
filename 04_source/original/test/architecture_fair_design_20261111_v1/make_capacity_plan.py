"""Static capacity choices for the common665-input/192-output interface."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
D=665
O=192


def core_count(name,width,degree=3):
    if name=='chebkan':return (degree+1)*(D*width+width*width+width*O)+2*width+O
    if name=='pinet':return degree*(D+1)*width+(width+1)*O
    if name=='dendrite':return (D+1)*width+(degree-1)*width*width+(width+1)*O
    if name=='realnpu':return width*(D+O)+D+O
    if name in ['rbf','scn']:return (D+O+1)*width+O
    if name=='elm':return (D+1)*width+(width+1)*O
    if name=='cascor':return width*(D+1)+width*(width-1)//2+(D+width+1)*O
    raise ValueError(name)


def main():
    fno=json.loads((HERE/'fno_constructor_count.json').read_text());budget=fno['real_scalars']+10
    models=[dict(model='tfno',width=48,degree=None,planned_core_coefficient_scalars=fno['real_scalars'],planned_total_coefficient_scalars=budget,planned_total_fitted_scalars=budget)]
    for name in ['chebkan','pinet','dendrite','realnpu','rbf','cascor','elm','scn']:
        width=min(range(1,10000),key=lambda w:abs(core_count(name,w)+10-budget))
        value=core_count(name,width)
        models.append(dict(model=name,width=width,degree=3 if name in ['chebkan','pinet','dendrite'] else None,
            planned_core_coefficient_scalars=value,planned_total_coefficient_scalars=value+10,
            planned_total_fitted_scalars=(width+1)*O+10 if name=='elm' else value+10,
            relative_budget_difference=(value+10-budget)/budget,
            constructive_units_are_intended_not_yet_selected=name in ['scn','cascor']))
    plan=dict(status='STATIC_CAPACITY_PLAN_NO_TRAINING',reference_total_fitted_real_scalars=budget,reference_total_coefficient_scalars=budget,
        primary_matching_metric='all_core_coefficients_including_fixed_random_plus_normalization',
        input_channels=D,output_channels=O,normalization_scalars_included=10,models=models,
        note='Common full-field packing and deterministic geometry/context transforms. No claim of equal accuracy or runtime. Construction counts must be verified before any pilot.')
    (HERE/'capacity_plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    for row in models:print(row['model'],row['width'],'total',row['planned_total_coefficient_scalars'],'data-fitted',row['planned_total_fitted_scalars'])


if __name__=='__main__':main()
