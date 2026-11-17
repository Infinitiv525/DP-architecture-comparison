"""Remote, bounded CPU audit of weights, raw outputs, metrics and timing traces."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from deploy import HERE, ROOT, ssh_python


def remote_audit(job, source):
    import hashlib
    import json
    import os
    import sys
    from pathlib import Path
    import numpy as np
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, source)
    from data_metrics import read, digest, PilotData, field_metrics
    from models import PointPipeline, AuthorTFNOPipeline, SCN, CascadeCorrelation
    base = Path('/mnt/project/perun2601449/dp_llm_cfd/results/architecture_pilot_20261110_v1') / ('job_' + job)
    frozen = read(base / 'source_freeze.json')
    for name, expected in frozen['files'].items():
        assert digest(Path(source) / name) == expected, ('source', name)
        assert digest(base / 'source' / name) == expected, ('archived source', name)
    assert digest(Path(sys.prefix) / 'freeze.txt') == frozen['runtime_freeze_sha256']
    config = read(base / 'protocol.json')
    data = PilotData(config, 'cpu')
    finished = read(base / 'summary.json')
    assert finished['status'] == 'PILOT_FINISHED' and finished['successful'] == 10
    def finite_tree(value):
        if isinstance(value, torch.Tensor): return bool(torch.isfinite(value).all())
        if isinstance(value, dict): return all(finite_tree(v) for v in value.values())
        if isinstance(value, (list, tuple)): return all(finite_tree(v) for v in value)
        if isinstance(value, float): return bool(np.isfinite(value))
        return True
    checks = []; sample_hashes = set(); artifacts = {}
    for row in finished['models']:
        name = row['model']; folder = base / name
        assert row['status'] == 'COMPLETE'
        if name != 'pinn_reference':
            checkpoint = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True)
            assert checkpoint['source_sha256'] == frozen['files']
            assert checkpoint['normalization'] == data.normalization
            assert checkpoint['config'] == config
            assert finite_tree(checkpoint['state'])
            model = restore_model(checkpoint)
            if name != 'tfno':
                with torch.inference_mode():
                    assert torch.isfinite(model.core(torch.randn(17, model.features.inputs))).all()
                sample_hashes.add(row['summary']['training']['point_sample_sha256'])
            del model, checkpoint
            artifacts[name + '/model.pt'] = digest(folder / 'model.pt')
            if name not in ['elm', 'scn', 'cascor']:
                state = torch.load(folder / 'optimizer_final.pt', map_location='cpu', weights_only=True)
                assert state['updates'] == config['gradient_updates']
                assert state['optimizer']['state']
                artifacts[name + '/optimizer_final.pt'] = digest(folder / 'optimizer_final.pt')
                del state
        else:
            ref = row['summary']['training']['reference']
            assert digest(ref['checkpoint']) == ref['checkpoint_sha256']
            artifacts['pinn_reference/original_model.pt'] = ref['checkpoint_sha256']
        traces = np.load(folder / 'raw_timings.npz', allow_pickle=False)
        assert traces['ids'].tolist() == data.validation_ids
        assert traces['gpu_ms'].shape == (4, config['latency_repetitions'])
        assert traces['wall_ms'].shape == (4, config['wall_repetitions'])
        assert np.isfinite(traces['gpu_ms']).all() and (traces['gpu_ms'] > 0).all()
        assert np.isfinite(traces['wall_ms']).all() and (traces['wall_ms'] > 0).all()
        cases = read(folder / 'case_metrics.json')
        for position, case in enumerate(cases):
            index = case['index']; known = data.cases[index]; fluid = known['fluid'].numpy()
            path = folder / ('prediction_%04d.npz' % index)
            with np.load(path, allow_pickle=False) as stored:
                pred = stored['prediction']
                assert pred.shape == (512, 512, 3) and np.isfinite(pred).all()
                assert np.count_nonzero(pred[~fluid]) == 0
                assert abs(pred[:, :, 2][fluid].mean(dtype=np.float64)) < 1e-5
                again = field_metrics(pred, data.references[index].transpose(1, 2, 0), fluid, known['reynolds'])
            assert again == case['metrics'], (name, index, 'metrics differ from stored raw prediction')
            assert np.isclose(np.percentile(traces['gpu_ms'][position], 99), case['timing']['gpu']['p99_ms'], atol=1e-12, rtol=0)
            assert np.isclose(np.percentile(traces['wall_ms'][position], 99), case['timing']['synchronized_wall']['p99_ms'], atol=1e-12, rtol=0)
            artifacts[name + '/' + path.name] = digest(path)
        assert len(cases) == 4 and {v['index'] for v in cases} == set(data.validation_ids)
        assert np.isclose(np.mean([c['metrics']['all_cell_mse'] for c in cases]), row['summary']['mean_validation_mse'])
        assert np.isclose(max(c['timing']['gpu']['p99_ms'] for c in cases), row['summary']['worst_case_gpu_p99_ms'])
        artifacts[name + '/raw_timings.npz'] = digest(folder / 'raw_timings.npz')
        artifacts[name + '/summary.json'] = digest(folder / 'summary.json')
        checks.append(dict(model=name, strict_checkpoint_load=True if name != 'pinn_reference' else 'original SHA256 checked',
                           raw_fields_recomputed=4, timing_traces_recomputed=True))
    assert len(sample_hashes) == 1
    return dict(status='PILOT_ARTIFACT_AUDIT_PASS', job=job, source=source,
                source_hashes_verified=len(frozen['files']), runtime_freeze_verified=True,
                models=checks, identical_point_training_sample_sha256=next(iter(sample_hashes)),
                validation_fields_recomputed=40, test_targets_read=False, artifact_sha256=artifacts,
                accounting_is_not_validated=True)


def main(job):
    import inspect
    from checkpoint_io import restore_model
    receipt = json.loads((HERE / 'submission.json').read_text())
    assert str(job) == receipt['job_id']
    code = 'import json\n' + inspect.getsource(restore_model) + '\n' + inspect.getsource(remote_audit) + '\nprint(json.dumps(remote_audit(' + repr(str(job)) + ',' + repr(receipt['source']) + ')))'
    wrapper = '''import subprocess,sys,os
r=subprocess.run([PY,'-B','-c',CODE],stdout=subprocess.PIPE,stderr=subprocess.PIPE,
 env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',WANDB_MODE='disabled',WANDB_DISABLED='true',DO_NOT_TRACK='1'))
sys.stdout.write(r.stdout.decode());sys.stderr.write(r.stderr.decode());sys.exit(r.returncode)
'''.replace('PY', repr(ROOT + '/envs/architecture-pilot-20261110-v1/bin/python')).replace('CODE', repr(code))
    result = json.loads(ssh_python(wrapper))
    result['audit_implementation_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['checkpoint_loader_sha256'] = hashlib.sha256((HERE / 'checkpoint_io.py').read_bytes()).hexdigest()
    destination = HERE / ('report_' + str(job)) / 'artifact_audit.json'
    destination.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'artifact_sha256'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('job')
    main(parser.parse_args().job)
