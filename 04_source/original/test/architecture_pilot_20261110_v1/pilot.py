"""Bounded serial GPU feasibility pilot. Never opens TEST target arrays."""
import argparse
import csv
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from models import PointPipeline, AuthorTFNOPipeline, fit_elm, fit_scn, fit_cascade
from data_metrics import PilotData, read, save, digest, field_metrics, plot_comparison

SOURCE = Path(__file__).resolve().parent


def configure(config):
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name():
        raise RuntimeError('One allocated H200 is required')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible GPU is required')


class ArchivedPINN:
    def __init__(self, config, normalization):
        root = Path(config['root'])
        for directory in ['training', 'analysis', 'llm']:
            sys.path.insert(0, str(root / 'test' / directory))
        from ldc_shared_pinn_v26 import SharedSteadyPINN, regular_grid
        from ldc_store_v26 import query
        manifest = read(root / config['reference_manifest'])
        selected = [m for m in manifest['models'] if m['id'] == config['reference_id']]
        if len(selected) != 1 or selected[0]['kind'] != 'fixed':
            raise ValueError('A prespecified fixed reference is required')
        self.item = selected[0]
        if digest(self.item['checkpoint']) != self.item['checkpoint_sha256']:
            raise ValueError('Archived reference checksum mismatch')
        checkpoint = torch.load(self.item['checkpoint'], map_location='cuda', weights_only=True)
        if checkpoint['normalization'] != normalization:
            raise ValueError('Reference normalization differs')
        self.model = SharedSteadyPINN(checkpoint['config'], normalization).cuda().eval()
        self.model.load_state_dict(checkpoint['state'])
        self.q = regular_grid()
        self.query = query

    def __call__(self, known):
        previous_known = dict(geometry_native=known['geometry'][None], fluid=known['fluid'], reynolds=known['reynolds'])
        return self.query(self.model, previous_known, self.q)


def normalized_point_score(model, x, y):
    with torch.inference_mode():
        return float((model.core(x) - y).square().mean())


def gradient_train(name, model, data, config, output):
    generator = torch.Generator(device='cuda').manual_seed(config['seed'] + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    history = []
    if name != 'tfno':
        x, y, sample_sha = data.training_features(model.features, config['training_points_per_case'], config['seed'] + 100)
        before = normalized_point_score(model, x, y)
    else:
        x = y = None
        sample_sha = None
        before = None
    started = time.monotonic()
    for update in range(config['gradient_updates']):
        if name == 'tfno':
            index = data.train_ids[update % len(data.train_ids)]
            pred = model(data.cases[index])
            loss = ((pred - data.targets[index]) / data.std).square().mean()
        else:
            indices = torch.randint(len(x), (config['gradient_batch_points'],), generator=generator, device='cuda')
            loss = (model.core(x[indices]) - y[indices]).square().mean()
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite optimization loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if hasattr(getattr(model, 'core', None), 'project_parameters'):
            model.core.project_parameters()
        if update % 20 == 0 or update + 1 == config['gradient_updates']:
            row = dict(update=update + 1, normalized_loss=float(loss), gradient_norm_before_clip=float(grad),
                       elapsed_seconds=time.monotonic() - started)
            history.append(row)
            save(output / 'training_history.json', history)
            print('TRAIN', name, json.dumps(row), flush=True)
    torch.cuda.synchronize()
    torch.save(dict(optimizer=optimizer.state_dict(), updates=config['gradient_updates'],
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                    minibatch_rng=generator.get_state()), output / 'optimizer_final.pt')
    return dict(algorithm='Adam, gradient clipping1, data loss only', updates=config['gradient_updates'],
        initial_point_training_mse=before,
        final_point_training_mse=None if x is None else normalized_point_score(model, x, y),
        point_sample_sha256=sample_sha, point_samples=None if x is None else len(x), history=history)


def constructive_train(name, model, data, config, output):
    x, y, sample_sha = data.training_features(model.features, config['training_points_per_case'], config['seed'] + 100)
    kwargs = dict(ridge=config['ridge'])
    if name == 'elm': history = fit_elm(model.core, x, y, **kwargs)
    elif name == 'scn':
        history = fit_scn(model.core, x, y, width=config['constructive_width'], pool=config['constructive_pool'], **kwargs)
    else:
        history = fit_cascade(model.core, x, y, width=config['constructive_width'], pool=config['cascade_pool'],
                              candidate_steps=config['cascade_candidate_steps'], **kwargs)
    save(output / 'training_history.json', history)
    return dict(algorithm=name, point_samples=len(x), point_sample_sha256=sample_sha,
                history=history, final_point_training_mse=normalized_point_score(model, x, y))


def percentiles(values):
    p = np.percentile(np.asarray(values, dtype=np.float64), [50, 95, 99])
    return dict(p50_ms=float(p[0]), p95_ms=float(p[1]), p99_ms=float(p[2]), max_ms=float(np.max(values)))


def benchmark(query, known, config):
    with torch.inference_mode():
        for _ in range(config['latency_warmup']): query(known)
        torch.cuda.synchronize()
        pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                 for _ in range(config['latency_repetitions'])]
        for start, end in pairs:
            start.record();query(known);end.record()
        torch.cuda.synchronize()
        gpu = np.asarray([a.elapsed_time(b) for a, b in pairs], dtype=np.float64)
        wall = []
        for _ in range(config['wall_repetitions']):
            torch.cuda.synchronize();start = time.perf_counter_ns()
            query(known);torch.cuda.synchronize()
            wall.append((time.perf_counter_ns() - start) / 1e6)
    result = dict(gpu=percentiles(gpu), synchronized_wall=percentiles(wall),
                  gpu_p99_target_pass=bool(np.percentile(gpu, 99) <= config['latency_target_p99_ms']))
    return result, gpu, np.asarray(wall, dtype=np.float64)


def run_trial(name, config, output):
    configure(config)
    output.mkdir(parents=True, exist_ok=False)
    data = PilotData(config)
    save(output / 'data_plan.json', data.plan)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic();cpu_start = time.process_time()
    if name == 'pinn_reference':
        wrapped = ArchivedPINN(config, data.normalization)
        model = wrapped.model
        query = wrapped
        training = dict(algorithm='NO_RETRAIN_archived_fixed_k00', reference=wrapped.item,
                        not_comparable_training_budget=True)
    else:
        model = (AuthorTFNOPipeline(data.normalization, config['tfno']) if name == 'tfno'
                 else PointPipeline(name, data.normalization, config)).cuda()
        query = model
        training_start = time.monotonic()
        if name in ['elm', 'scn', 'cascor']: training = constructive_train(name, model, data, config, output)
        else: training = gradient_train(name, model, data, config, output)
        torch.cuda.synchronize()
        training['wall_seconds'] = time.monotonic() - training_start
        model.eval()
        dynamic = dict(scn_units=len(model.core.weight)) if name == 'scn' else (
                  dict(cascor_units=len(model.core.units)) if name == 'cascor' else {})
        torch.save(dict(model=name, state=model.state_dict(), config=config, normalization=data.normalization,
                        dynamic=dynamic, source_sha256=read(output.parent / 'source_freeze.json')['files']), output / 'model.pt')
        save(output / 'training_summary.json', training)
    core = model if name == 'pinn_reference' else model.core
    core_parameters = sum(p.numel() for p in core.parameters())
    core_buffers = sum(p.numel() for p in core.buffers())
    fitted_scalars = (core.beta.numel() if name == 'elm' else
                      core_buffers if name == 'scn' else
                      core_parameters + core.beta.numel() if name == 'cascor' else core_parameters)
    save(output / 'model_description.json', dict(name=name, repr=repr(model),
         total_parameters=sum(p.numel() for p in model.parameters()),
         trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
         buffer_values=sum(p.numel() for p in model.buffers()),
         core_parameter_values=core_parameters, core_buffer_values=core_buffers,
         data_fitted_or_selected_core_scalars=fitted_scalars,
         counting_note='ELM fixed random weights are buffers; SCN selected weights and CasCor frozen units also count as fitted/selected. Static coordinate buffers are not learned weights.',
         point_encoding=None if name in ['tfno', 'pinn_reference'] else config['point_representation']))
    cases = [];gpu_traces = [];wall_traces = []
    for position, index in enumerate(data.validation_ids):
        known = data.cases[index]
        with torch.inference_mode():
            prediction = query(known)
            if prediction.shape != (512, 512, 3) or not torch.isfinite(prediction).all():
                raise FloatingPointError('Invalid native field')
            prediction_np = prediction.cpu().numpy()
        fluid = known['fluid'].cpu().numpy()
        reference = data.references[index].transpose(1, 2, 0)
        metrics = field_metrics(prediction_np, reference, fluid, known['reynolds'])
        timing, gpu, wall = benchmark(query, known, config)
        cases.append(dict(index=index, reynolds=known['reynolds'], metrics=metrics, timing=timing))
        gpu_traces.append(gpu);wall_traces.append(wall)
        np.savez_compressed(output / f'prediction_{index:04d}.npz', prediction=prediction_np)
        if position == 0:
            plot_comparison(output / 'comparison.png', prediction_np, reference, fluid, name + ' / VAL case ' + str(index))
        save(output / 'case_metrics.json', cases)
    np.savez_compressed(output / 'raw_timings.npz', ids=np.asarray(data.validation_ids), gpu_ms=np.stack(gpu_traces), wall_ms=np.stack(wall_traces))
    summary = dict(status='TECHNICAL_PILOT_COMPLETE', model=name,
        validation_cases=len(cases), mean_validation_mse=float(np.mean([c['metrics']['all_cell_mse'] for c in cases])),
        mean_velocity_relative_l2=float(np.mean([c['metrics']['fluid_velocity_relative_l2'] for c in cases])),
        mean_pressure_relative_l2=float(np.mean([c['metrics']['fluid_pressure_relative_l2'] for c in cases])),
        worst_case_gpu_p99_ms=max(c['timing']['gpu']['p99_ms'] for c in cases),
        gpu_p99_pass_all_cases=all(c['timing']['gpu_p99_target_pass'] for c in cases),
        wall_seconds=time.monotonic() - start, cpu_seconds=time.process_time() - cpu_start,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        training=training, device=torch.cuda.get_device_name(), torch=torch.__version__,
        neuraloperator=metadata.version('neuraloperator'), test_targets_read=False,
        final_architecture_ranking_allowed=False, statistical_claims_allowed=False)
    save(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def coordinate(config, output):
    if output.exists(): raise FileExistsError('A fresh output directory is required')
    output.mkdir(parents=True)
    source_files = sorted(p for p in SOURCE.iterdir() if p.is_file() and p.suffix in ['.py', '.json', '.sh', '.sbatch', '.txt'])
    archive = output / 'source';archive.mkdir()
    for path in source_files: shutil.copy2(path, archive / path.name)
    save(output / 'source_freeze.json', dict(files={p.name: digest(p) for p in source_files},
         runtime_freeze_sha256=digest(Path(sys.prefix) / 'freeze.txt'),
         python=sys.executable, source=str(SOURCE), slurm_job_id=os.environ.get('SLURM_JOB_ID')))
    save(output / 'protocol.json', config)
    # No model work is started unless the independent mathematical checks pass.
    with (output / 'core_checks.log').open('w') as stream:
        checked = subprocess.run([sys.executable, '-B', str(SOURCE / 'checks.py'),
                    '--data-config', str(SOURCE / 'pilot_config.json')], stdout=stream, stderr=subprocess.STDOUT)
    if checked.returncode: raise RuntimeError('Core mathematical checks failed; see core_checks.log')
    records = []
    for name in config['models']:
        path = output / name
        print('START', name, flush=True)
        with (output / (name + '.log')).open('w') as stream:
            began = time.monotonic()
            try:
                finished = subprocess.run([sys.executable, '-B', '-u', str(SOURCE / 'pilot.py'), '--model', name,
                    '--config', str(SOURCE / 'pilot_config.json'), '--output', str(path)],
                    stdout=stream, stderr=subprocess.STDOUT, timeout=config['trial_timeout_seconds'])
                returncode = finished.returncode
                failure = None if returncode == 0 else 'child_process_failed'
            except subprocess.TimeoutExpired:
                returncode = None;failure = 'bounded_trial_timeout'
        summary_path = path / 'summary.json'
        summary = read(summary_path) if summary_path.exists() else None
        success = returncode == 0 and summary is not None and summary.get('status') == 'TECHNICAL_PILOT_COMPLETE'
        records.append(dict(model=name, status='COMPLETE' if success else 'FAILED', returncode=returncode,
            failure=failure, wall_seconds=time.monotonic() - began, summary=summary))
        save(output / 'progress.json', dict(models=records, completed=len(records), planned=len(config['models'])))
        print('FINISH', name, records[-1]['status'], flush=True)
    with (output / 'overview.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['model', 'status', 'validation_mse', 'velocity_l2', 'pressure_l2', 'worst_gpu_p99_ms', 'p99_pass'])
        writer.writeheader()
        for item in records:
            s = item['summary'] or {}
            writer.writerow(dict(model=item['model'], status=item['status'], validation_mse=s.get('mean_validation_mse'),
                velocity_l2=s.get('mean_velocity_relative_l2'), pressure_l2=s.get('mean_pressure_relative_l2'),
                worst_gpu_p99_ms=s.get('worst_case_gpu_p99_ms'), p99_pass=s.get('gpu_p99_pass_all_cases')))
    complete = all(r['status'] == 'COMPLETE' for r in records)
    save(output / 'summary.json', dict(status='PILOT_FINISHED' if complete else 'PILOT_FINISHED_WITH_FAILURES',
        models=records, successful=sum(r['status'] == 'COMPLETE' for r in records), planned=len(records),
        final_architecture_ranking_allowed=False, test_targets_read=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--config', type=Path, default=SOURCE / 'pilot_config.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        configuration = read(args.config)
        if args.model:
            run_trial(args.model, configuration, args.output)
        else:
            coordinate(configuration, args.output)
    except Exception:
        traceback.print_exc()
        raise
