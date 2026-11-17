"""Shared spatial PINN with a matched 2-versus-20-frame known history.

Both arms anchor at original frame21 and predict22..241. Older observations
enter the same expanded CNN; their columns are initially zero in BOTH arms.
"""
import numpy as np
import torch
from torch import nn
from fpo_spatial_context_v21 import SpatialPINN
from fpo_step_v5a import load_cases as load_original_cases, project_known_constraints

ANCHOR = 21
TARGET_COUNT = 220
DT = .05
END_TIME = TARGET_COUNT * DT
HISTORY_LENGTH = 20


def target_frame(offset):
    if not 1 <= offset <= TARGET_COUNT:
        raise ValueError('Outside fixed prediction interval')
    return ANCHOR + offset


def read_known_history(reference, fluid, device='cuda'):
    # Only original frames2..21 are read, including for development validation.
    observed = np.array(reference[2:ANCHOR+1], dtype=np.float32, copy=True)
    if observed.shape != (20, *fluid.shape, 3) or not np.isfinite(observed).all():
        raise ValueError('Invalid known history')
    history = torch.as_tensor(observed, device=device)
    mask = torch.as_tensor(fluid, device=device)
    p = history[:, :, :, 2]
    p -= p[:, mask].mean(1)[:, None, None]
    return history * mask[None, :, :, None]


def attach_history(case, history):
    if len(history) != HISTORY_LENGTH:
        raise ValueError('Expected20 known frames')
    case['history'] = history
    case['raw'] = torch.cat((history[-1], history[-1]-history[-2]), 2)
    return case


def pool_history(history):
    k, ny, nx, channels = history.shape
    if k != HISTORY_LENGTH or channels != 3 or ny % 8 or nx % 8:
        raise ValueError('Unexpected history dimensions')
    return history.reshape(k, ny//8, 8, nx//8, 8, 3).mean((2, 4)).permute(0, 3, 1, 2)


def conditioning(model, case):
    # Recomputed from known input fields for every query, including benchmarks.
    low = pool_history(case['history'])
    state = torch.cat((low[-1], low[-1]-low[-2]), 0)[None]
    older = (low[:-2]-low[-1:]).reshape(1, 54, *low.shape[-2:])
    return model.context(case['geometry'], state, case['record']['reynolds'], older)


class HistoryPINN(SpatialPINN):
    def __init__(self, config, mean, std):
        super().__init__(config, mean, std)
        self.history_enabled = bool(config['history_enabled'])
        if not self.local_context:
            raise ValueError('V21 spatial path fixed in both history arms')
        old = self.encoder[0]
        new = nn.Conv2d(62, old.out_channels, old.kernel_size, padding=old.padding)
        with torch.no_grad():
            new.weight[:, :8].copy_(old.weight)
            new.weight[:, 8:].zero_()
            new.bias.copy_(old.bias)
        self.encoder[0] = new

    def load_source(self, state):
        if state['encoder.0.weight'].shape[1] != 8 or state['net.0.weight'].shape[1] != self.net[0].in_features:
            raise ValueError('Expected V21 spatial source')
        adapted = dict(state)
        w = state['encoder.0.weight']
        adapted['encoder.0.weight'] = torch.cat((w, w.new_zeros((w.shape[0], 54, *w.shape[2:]))), 1)
        self.load_state_dict(adapted, strict=True)

    def context(self, geometry, state_low, reynolds, older):
        if older.shape[1] != 54 or older.shape[-2:] != state_low.shape[-2:]:
            raise ValueError('Explicit known history required')
        obs = (state_low[:, :3]-self.mean[None, :, None, None])/self.std[None, :, None, None]
        change = state_low[:, 3:]/self.std[None, :, None, None]
        std = self.std.repeat(18)[None, :, None, None]
        past = older/std if self.history_enabled else torch.zeros_like(older)
        value = torch.cat((geometry, obs, change, past), 1)
        for i in range(6):
            value = self.encoder[i](value)
        spatial = value[0]
        z = value
        for i in range(6, len(self.encoder)):
            z = self.encoder[i](z)
        re = z.new_tensor(float(reynolds)).log().reshape(1, 1)/8
        return dict(global_context=torch.cat((z, re), 1), spatial=spatial)


def query(model, known, qbase, time_value):
    if not 0 <= time_value <= END_TIME + 1e-6:
        raise ValueError('Query outside available future')
    q = qbase.clone()
    q[:, 2] = time_value
    context = conditioning(model, known)
    value = model.grid_prediction(q, context, known['raw'])
    return project_known_constraints(value, known['fluid'])


def load_cases(records):
    cases = load_original_cases(records)
    for c in cases:
        c['reference'] = np.load(c['record']['field'], mmap_mode='r', allow_pickle=False)
        if c['record']['proposed_split'] == 'train':
            history = c['data'][:20]  # zero-copy view of TRAIN frames2..21
        else:
            if 'data' in c:
                raise ValueError('Future validation labels on GPU')
            history = read_known_history(c['reference'], c['fluid_np'])
            c.pop('prefix', None)
        attach_history(c, history)
        c['interior_gpu'] = torch.zeros_like(c['fluid'])
        y, x = c['interior'].T
        c['interior_gpu'][y, x] = True
    return cases


def known_case(record):
    if record['proposed_split'] not in ['train', 'validation']:
        raise ValueError('Final test sealed')
    with np.load(str(record['cache'])+'/geometry.npz', allow_pickle=False) as z:
        fluid = z['fluid'].copy()
        geometry = z['encoded'].copy()
    reference = np.load(record['field'], mmap_mode='r', allow_pickle=False)
    return attach_history(dict(record={'reynolds': record['reynolds']},
        geometry=torch.tensor(geometry[None], device='cuda'),
        fluid=torch.tensor(fluid, device='cuda')),
        read_known_history(reference, fluid))
