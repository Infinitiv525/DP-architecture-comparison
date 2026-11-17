"""Exercise new trainer utilities before the full-data array is released."""
from pathlib import Path
import numpy as np
import torch
from common import patch_loss,patch_prediction,recipe,save,setup
from operator_models import PackedFieldOperator,restore_model
from train import atomic_checkpoint,physical_validation_metrics,validation,dynamic
from weighted_fit import fit_output,scn_candidate,append_scn,cascade_candidate


def smoke(folder,normalization):
    setup('cpu',932001);folder=Path(folder)
    tables={phase:tuple(torch.from_numpy(np.array(np.load(folder/(phase+'_'+label+'.npy'),mmap_mode='r',allow_pickle=False)[0],copy=True))
                       for label in ['x','y','weights']) for phase in ['train','validation']}
    class OneCase:
        def iterator(self,phase='train'):
            yield tables[phase]
    fixture=OneCase(); records=[]
    for name in ['tfno','chebkan','pinet','dendrite','realnpu','rbf','elm','scn','cascor']:
        model=PackedFieldOperator(name,recipe(name),normalization)
        x,y,w=tables['train'];meta={}
        if name in ['elm','scn','cascor']:
            fit_output(model.core,lambda:fixture.iterator())
            if name=='scn':
                node,meta=scn_candidate(model.core,lambda:fixture.iterator(),seed=932001)
                if node is not None:append_scn(model.core,node)
                fit_output(model.core,lambda:fixture.iterator())
            if name=='cascor':
                node,meta=cascade_candidate(model.core,lambda:fixture.iterator())
                model.core.add(node);fit_output(model.core,lambda:fixture.iterator())
        else:
            optimizer=torch.optim.Adam(model.core.parameters(),lr=.001)
            loss=patch_loss(patch_prediction(model.core,name,x),y,w)
            optimizer.zero_grad();loss.backward();optimizer.step()
            if hasattr(model.core,'project_parameters'):model.core.project_parameters()
        checkpoint=dict(model=name,recipe=recipe(name),normalization=normalization,
            state=model.state_dict(),dynamic=dynamic(model))
        path=folder/'smoke_checkpoints'/(name+'.pt');path.parent.mkdir(exist_ok=True)
        atomic_checkpoint(path,checkpoint)
        restored=restore_model(torch.load(path,map_location='cpu',weights_only=False))
        with torch.no_grad():
            a=patch_prediction(model.core,name,x);b=patch_prediction(restored.core,name,x)
            assert torch.allclose(a,b,atol=1e-6,rtol=1e-5)
        value=validation(restored,fixture);physical=physical_validation_metrics(restored,fixture)
        assert value['cases']==1 and len(physical['case_metrics'])==1
        records.append(dict(model=name,validation=value,physical=physical,
            restore_max_difference=float((a-b).abs().max()),constructive_first_step=meta))
    save(folder/'trainer_smoke.json',dict(status='ALL9_TRAINER_UTILITY_SMOKE_COMPLETE',models=records,
        TRAIN_cases=1,VAL_cases=1,test_targets_read=False,accuracy_cutoff=None,
        note='One-case integration check, not a full training or accuracy acceptance test.'))
