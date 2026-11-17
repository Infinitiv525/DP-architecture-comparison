import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class PilotData:
    def __init__(self, config, device='cuda'):
        if config['test_targets_allowed'] is not False:
            raise ValueError('This pilot never authorizes TEST targets')
        root = Path(config['root'])
        prepared = read(root / config['prepared_path'])
        if prepared['status'] != 'LDC_DATA_READY':
            raise ValueError('Prepared-data audit is required')
        split = read(root / config['split_path'])['indices']
        self.train_ids = split['train'][:config['train_cases']]
        self.validation_ids = split['validation'][:config['validation_cases']]
        if set(self.train_ids) & set(self.validation_ids): raise ValueError('TRAIN/VAL overlap')
        if (set(self.train_ids) | set(self.validation_ids)) & set(split['test']): raise ValueError('TEST cases selected')
        cache = Path(prepared['cache'])
        self.normalization = read(cache / 'normalization.json')
        inputs = read(cache / 'inputs.json')
        raw_x = np.load(cache.parent / 'raw/X_original.npy', mmap_mode='r', allow_pickle=False)
        fluids = np.load(cache / 'fluid.npy', mmap_mode='r', allow_pickle=False)
        # Intentionally no Y_test file/API exists in this pilot.
        y_train = np.load(cache / 'Y_train.npy', mmap_mode='r', allow_pickle=False)
        y_val = np.load(cache / 'Y_validation.npy', mmap_mode='r', allow_pickle=False)
        self.cases = {}
        self.references = {}
        schema = inputs['schema']
        for phase, ids, table, order in [('train', self.train_ids, y_train, split['train']),
                                         ('validation', self.validation_ids, y_val, split['validation'])]:
            slots = {index: slot for slot, index in enumerate(order)}
            for index in ids:
                fluid = np.array(fluids[index], dtype=bool, copy=True)
                sdf = np.array(raw_x[index, schema['sdf_channel']], dtype=np.float32, copy=True) * schema['sdf_orientation']
                geometry = np.stack((fluid.astype('float32'), sdf))
                target = np.array(table[slots[index]], dtype=np.float32, copy=True)
                if target.shape != (3, 512, 512) or geometry.shape != (2, 512, 512):
                    raise ValueError('Native shape mismatch')
                if not np.isfinite(target).all() or not np.isfinite(geometry).all():
                    raise FloatingPointError('Nonfinite input/reference')
                if not fluid.any() or float(inputs['reynolds'][index]) <= 0:
                    raise ValueError('Empty fluid or invalid Reynolds number')
                if np.count_nonzero(target[:, ~fluid]) or abs(float(target[2, fluid].mean(dtype=np.float64))) > 1e-5:
                    raise ValueError('Reference solid mask/pressure gauge differs from the frozen protocol')
                self.cases[index] = dict(index=index, phase=phase,
                    geometry=torch.tensor(geometry, device=device), fluid=torch.tensor(fluid, device=device),
                    re=torch.tensor(float(inputs['reynolds'][index]), device=device),
                    reynolds=float(inputs['reynolds'][index]), geometry_hash=inputs['geometry_hashes'][index])
                self.references[index] = target
        self.std = torch.tensor(self.normalization['target_std'], device=device).float()
        self.mean = torch.tensor(self.normalization['target_mean'], device=device).float()
        self.targets = {i: torch.tensor(self.references[i].transpose(1, 2, 0).copy(), device=device) for i in self.train_ids}
        self.plan = dict(train_ids=self.train_ids, validation_ids=self.validation_ids,
            cases=[dict(index=i, phase=v['phase'], reynolds=v['reynolds'], geometry_hash=v['geometry_hash']) for i, v in self.cases.items()],
            input_revision=prepared['revision'], normalization=self.normalization,
            prepared_sha256=digest(root / config['prepared_path']), split_sha256=digest(root / config['split_path']),
            test_targets_read=False, known_geometry_split=True,
            case_selection='first16 TRAIN/first4 VAL in frozen split order; no target-based selection')

    def training_features(self, features, count, seed):
        rng = np.random.default_rng(seed)
        xs, ys = [], []
        sample_hash = hashlib.sha256()
        with torch.no_grad():
            for index in self.train_ids:
                case = self.cases[index]
                available = torch.where(case['fluid'].flatten())[0]
                selected = available[torch.tensor(rng.integers(len(available), size=count), device=available.device)]
                xs.append(features(case, selected))
                target = self.targets[index].reshape(-1, 3)[selected]
                ys.append((target - self.mean) / self.std)
                sample_hash.update(np.asarray([index], dtype='<i8').tobytes())
                sample_hash.update(selected.cpu().numpy().astype('<i8').tobytes())
        return torch.cat(xs), torch.cat(ys), sample_hash.hexdigest()


def field_metrics(prediction, reference, fluid, reynolds):
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if not np.isfinite(prediction).all(): raise FloatingPointError('Nonfinite predicted field')
    error = prediction - reference
    mse_fields = (error * error).mean(axis=(0, 1))
    velocity = np.linalg.norm(error[fluid, :2]) / max(np.linalg.norm(reference[fluid, :2]), 1e-12)
    pressure = np.linalg.norm(error[fluid, 2]) / max(np.linalg.norm(reference[fluid, 2]), 1e-12)
    interior = fluid.copy()
    for axis in [0, 1]:
        for shift in [1, 2, 3]:
            interior &= np.roll(fluid, shift, axis) & np.roll(fluid, -shift, axis)
    interior[:3] = False;interior[-3:] = False;interior[:, :3] = False;interior[:, -3:] = False

    def physics(value):
        u, v, p = np.moveaxis(value, -1, 0)
        uy, ux = np.gradient(u, 1/512, edge_order=2)
        vy, vx = np.gradient(v, 1/512, edge_order=2)
        py, px = np.gradient(p, 1/512, edge_order=2)
        lap_u = np.gradient(ux, 1/512, axis=1, edge_order=2) + np.gradient(uy, 1/512, axis=0, edge_order=2)
        lap_v = np.gradient(vx, 1/512, axis=1, edge_order=2) + np.gradient(vy, 1/512, axis=0, edge_order=2)
        divergence = ux + vy
        residual = np.stack((u * ux + v * uy + px - lap_u / reynolds,
                             u * vx + v * vy + py - lap_v / reynolds), -1)
        return dict(divergence_rms=float(np.sqrt(np.mean(divergence[interior] ** 2))),
                    steady_ns_momentum_rms=float(np.sqrt(np.mean(residual[interior] ** 2))))

    return dict(all_cell_mse=float(mse_fields.mean()), mse_fields=dict(zip(['u', 'v', 'p'], mse_fields.tolist())),
                fluid_velocity_relative_l2=float(velocity), fluid_pressure_relative_l2=float(pressure),
                prediction_physics=physics(prediction), reference_physics=physics(reference),
                diagnostic_points=int(interior.sum()))


def audit_data(config):
    """CPU preflight: verifies the exact selected arrays, without optimization."""
    data = PilotData(config, device='cpu')
    arrays = {}
    for index, known in data.cases.items():
        arrays[str(index)] = dict(
            geometry_sha256=hashlib.sha256(known['geometry'].numpy().tobytes()).hexdigest(),
            reference_sha256=hashlib.sha256(data.references[index].tobytes()).hexdigest(),
            fluid_points=int(known['fluid'].sum()))
    return dict(status='PILOT_DATA_AUDIT_PASS', plan=data.plan, selected_array_digests=arrays)


def plot_comparison(path, prediction, reference, fluid, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(10, 9), constrained_layout=True)
    for c, field in enumerate(['u', 'v', 'p']):
        low, high = np.quantile(reference[:, :, c][fluid], [.01, .99])
        for j, values in enumerate([reference[:, :, c], prediction[:, :, c], np.abs(prediction[:, :, c] - reference[:, :, c])]):
            image = np.where(fluid, values, np.nan)
            im = axes[c, j].imshow(image, origin='lower', cmap='magma' if j == 2 else 'coolwarm',
                                  vmin=0 if j == 2 else low, vmax=None if j == 2 else high, extent=(0, 2, 0, 2))
            axes[c, j].set_title(field + ' ' + ['CFD', 'model', 'absolute error'][j])
            fig.colorbar(im, ax=axes[c, j], shrink=.8)
    fig.suptitle(title)
    fig.savefig(path, dpi=130)
    plt.close(fig)
