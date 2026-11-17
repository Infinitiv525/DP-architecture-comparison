"""Check all 45 independent checkpoints on VAL; TEST remains sealed."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import torch
from common import SOURCE,read,save,digest,engine,manifest


def main(config,root):
    e,c=engine(config);e.configure('cuda',4,911113);selection=manifest(config)
    from checkpoint_io import restore_model
    output=root/'validation';output.mkdir(parents=True,exist_ok=False);store=e.Store(c,'cuda');rows=[]
    for task in selection['tasks']:
        path=root/'models'/task['id']/'summary.json'
        if not path.is_file():rows.append(dict(id=task['id'],status='TRAINING_FAILED_OR_INCOMPLETE'));continue
        s=read(path);assert s['status']=='ARCHITECTURE_MAIN_MODEL_COMPLETE' and s['task']==task
        assert s['selection_sha256']==config['selection_sha256'] and not s['test_targets_read']
        best=s['best'];checkpoint=Path(best['checkpoint']);assert digest(checkpoint)==best['checkpoint_sha256']
        cp=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert cp['model']==task['model'] and cp['config']['seed']==task['seed'] and cp['normalization']==store.normalization
        assert cp['source']==str(SOURCE) and not cp['test_targets_read']
        for name,sha in cp['source_sha256'].items():assert digest(Path(cp['source'])/name)==sha
        model=restore_model(cp).cuda().eval();score=e.validate(model,store,store.val_ids)
        assert np.isclose(score['mse'],best['validation_mse'],rtol=5e-5,atol=1e-11)
        core=model.core;counts=dict(parameter_elements=sum(p.numel() for p in core.parameters()),
            parameter_real_scalars=sum(p.numel()*(2 if p.is_complex() else 1) for p in core.parameters()),
            buffer_elements=sum(p.numel() for p in core.buffers()),
            buffer_real_scalars=sum(p.numel()*(2 if p.is_complex() else 1) for p in core.buffers()))
        save(output/task['id']/'validation.json',score)
        row=dict(id=task['id'],status='MAIN_VALIDATION_PASS',model=task['model'],seed=task['seed'],
            checkpoint=str(checkpoint),checkpoint_sha256=best['checkpoint_sha256'],epoch=cp['epoch'],
            validation_mse=score['mse'],velocity_l2=score['velocity_l2'],pressure_l2=score['pressure_l2'],counts=counts)
        rows.append(row);save(output/'progress.json',rows);print('VALIDATED',json.dumps(row),flush=True)
        del model,cp;gc.collect();torch.cuda.empty_cache()
    completed=sum(r['status']=='MAIN_VALIDATION_PASS' for r in rows)
    save(output/'summary.json',dict(status='ARCHITECTURE_MAIN_VALIDATION_COMPLETE' if completed==45 else 'ARCHITECTURE_MAIN_INCOMPLETE',
        models=rows,completed_models=completed,expected_models=45,full_validation_cases=160,
        selection_sha256=config['selection_sha256'],test_targets_read=False,final_test_evaluation_started=False))
    if completed!=45:raise RuntimeError('Missing final trainings; inspect the manifest and job logs')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    main(read(SOURCE/'config.json'),a.root)
