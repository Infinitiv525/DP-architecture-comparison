"""CPU restoration and full validation recomputation of frozen completed models."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent


def main(root,group):
    start=time.perf_counter();manifest=json.loads((HERE/'manifest.json').read_text())
    for folder,hashes in [(Path(manifest['source']),manifest['source_sha256']),
                          (Path(manifest['dependency_source']),manifest['dependency_sha256'])]:
        for name,expected in hashes.items():
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==expected,str(folder/name)
    manifest['models']=manifest['models'][group::2]
    root=Path(root)/('group_'+str(group))
    sys.path.insert(0,manifest['dependency_source'])
    from common import save,setup
    from operator_models import restore_model
    from train import validation,physical_validation_metrics
    setup('cpu',933001)
    fixture_folder=Path(manifest['fixture_root'])/'fixture'
    assert hashlib.sha256((fixture_folder/'ready.json').read_bytes()).hexdigest()==manifest['fixture_ready_sha256']
    tables=tuple(torch.from_numpy(np.load(fixture_folder/('validation_'+label+'.npy'),allow_pickle=False))
                 for label in ['x','y','weights'])
    assert tables[0].shape==(160,4096,665) and tables[1].shape==(160,4096,192)
    class ValidationOnly:
        def iterator(self,phase='validation'):
            assert phase=='validation'
            for slot in range(160):yield tuple(t[slot] for t in tables)
    fixture=ValidationOnly();results=[]
    for row in manifest['models']:
        checkpoint=Path(row['checkpoint']);assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==row['checkpoint_sha256']
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert payload['task']==row['task']
        saved=payload['history'][-1]['validation'];assert saved['cases']==160
        assert abs(saved['mean_raw_case_balanced_loss']-row['expected_complete_validation_loss'])<1e-12
        model=restore_model(payload).cpu().eval()
        score=validation(model,fixture);physical=physical_validation_metrics(model,fixture)
        before=np.asarray(saved['case_losses']);after=np.asarray(score['case_losses'])
        # Tolerances concern CPU/GPU arithmetic, never scientific accuracy.
        if not np.allclose(before,after,atol=2e-7,rtol=5e-4):
            raise ValueError('CPU/GPU validation mismatch '+row['id']+': '+str(np.max(np.abs(before-after))))
        expected=row['expected_physical']['case_metrics'];observed=physical['case_metrics']
        fields=['velocity_relative_L2','pressure_relative_L2']
        for field in fields:
            a=np.asarray([v[field] for v in expected]);b=np.asarray([v[field] for v in observed])
            if not np.allclose(a,b,atol=2e-5,rtol=1e-3):raise ValueError('Physical metric mismatch '+row['id']+'/'+field)
        ledger=model.parameter_ledger()
        for field in ['total_core_coefficients_plus_normalization','learned_or_fitted_total']:
            assert ledger[field]==row['selected_parameter_ledger'][field]
        item=dict(id=row['id'],status='CHECKPOINT_AND_ALL160_VALIDATION_FIELDS_VERIFIED',
            checkpoint_sha256=row['checkpoint_sha256'],CPU_validation=score,CPU_physical_validation=physical,
            maximum_raw_case_loss_CPU_GPU_difference=float(np.max(np.abs(before-after))),
            parameter_ledger=ledger,test_targets_read=False,training_updates=0)
        results.append(item);save(Path(root)/'models'/(row['id']+'.json'),item)
        print(row['id'],item['status'],flush=True)
    save(Path(root)/'complete.json',dict(status='COMPLETED_TUNING_MODEL_AUDIT_PASS',models=len(results),
        native_validation_fields=len(results)*160,source=str(HERE),results=results,
        test_targets_read=False,training_updates=0,GPU_allocated=False,accuracy_threshold=None,
        note='Numerical restoration and report consistency only; no accuracy acceptance threshold.',seconds=time.perf_counter()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--group',type=int,choices=[0,1],required=True);a=p.parse_args();main(a.root,a.group)
