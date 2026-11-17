"""Single-case PINN inputs with disjoint temporal label roles.

Only TRAIN-frame labels and the known prefix enter GPU memory.  The reference
mmap remains available to a separate CPU scorer.  This is reconstruction of
one case, not zero-shot forecasting of a new case from its initial state.
"""
import hashlib
from pathlib import Path
import numpy as np
import torch
from fpo_step_v5a import fingerprint


def split_frames():
    roles = dict(train=[], validation=[], audit=[])
    for frame in range(4, 242):
        remainder = (frame-4) % 6
        role = 'train' if remainder < 4 else ('validation' if remainder == 4 else 'audit')
        roles[role].append(frame)
    return roles


TRAIN_FRAMES = frozenset(split_frames()['train'])


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(4*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def canonicalize_labels(block, fluid, device='cuda'):
    """Per-frame pressure gauge and known solid mask, without changing input."""
    mask = torch.as_tensor(fluid, dtype=torch.bool, device=device)
    if mask.ndim != 2 or not bool(mask.any()):
        raise ValueError('Nonempty two-dimensional fluid mask required')
    copied = np.array(block, dtype=np.float32, copy=True)
    if copied.ndim != 4 or copied.shape[1:] != (*mask.shape, 3):
        raise ValueError('Expected time-height-width-three labels')
    if not np.isfinite(copied).all():
        raise ValueError('Nonfinite label after float32 conversion')
    value = torch.as_tensor(copied, device=device)
    pressure = value[:, :, :, 2]
    pressure -= pressure[:, mask].mean(1)[:, None, None]
    return value * mask[None, :, :, None]


def read_training_observations(reference, fluid, device='cuda', block_size=8):
    """The index list is frozen TRAIN only; never bulk-read withheld frames."""
    if not 1 <= block_size <= 8:
        raise ValueError('Read block must contain between one and eight frames')
    if tuple(reference.shape) != (242, *np.shape(fluid), 3):
        raise ValueError('Unexpected original field shape')
    frames = split_frames()['train']
    observed = {}
    for start in range(0, len(frames), block_size):
        selected = frames[start:start+block_size]
        block = canonicalize_labels(reference[selected], fluid, device)
        for index, frame in enumerate(selected):
            observed[frame] = block[index]
    if set(observed) != TRAIN_FRAMES:
        raise ValueError('Incomplete or contaminated TRAIN observations')
    return observed


def normalization_from_observed(observed, fluid):
    """Population statistics over canonical TRAIN fluid cells, weighted equally.

    The function does not mutate observations.  Float64 reductions limit
    cancellation, and the exported lists remain ordinary JSON-safe floats.
    """
    if set(observed) != TRAIN_FRAMES:
        raise ValueError('Normalizer requires exactly the declared TRAIN frames')
    first = observed[min(TRAIN_FRAMES)]
    mask = torch.as_tensor(fluid, dtype=torch.bool, device=first.device)
    if not bool(mask.any()):
        raise ValueError('No fluid cells')
    total = torch.zeros(3, dtype=torch.float64, device=first.device)
    squares = torch.zeros_like(total)
    count = 0
    with torch.no_grad():
        for frame in sorted(TRAIN_FRAMES):
            value = observed[frame]
            if value.shape != (*mask.shape, 3):
                raise ValueError('TRAIN field shape differs from fluid mask')
            cells = value[mask].to(dtype=torch.float64)
            if not bool(torch.isfinite(cells).all()):
                raise ValueError('Nonfinite normalizer input')
            total += cells.sum(0)
            squares += cells.square().sum(0)
            count += cells.shape[0]
        mean = total/count
        variance = (squares/count-mean.square()).clamp_min(0.)
        std = variance.sqrt().clamp_min(1e-4)
    return dict(mean=mean.cpu().tolist(), std=std.cpu().tolist(),
        population_fluid_points=count, frame_count=len(TRAIN_FRAMES),
        source='Only canonicalized original TRAIN frames; fluid cells only',
        pressure_gauge='Per-frame fluid-mean pressure removed before statistics',
        std_floor=1e-4)


def training_target(case, frame, y, x):
    frame = int(frame)
    if frame not in TRAIN_FRAMES:
        raise ValueError('Withheld frame cannot be a training target')
    if frame not in case['observed']:
        raise ValueError('TRAIN frame not loaded')
    return case['observed'][frame][y, x]


def known_inputs(case):
    """Prediction receives prefix and physical conditions, never scorer data."""
    return dict(raw=case['raw'], geometry=case['geometry'], fluid=case['fluid'],
        record={'reynolds': case['record']['reynolds']})


def load_case(record, device='cuda'):
    if record['proposed_split'] != 'train':
        raise ValueError('V24 feasibility uses existing TRAIN cases only')
    field = Path(record['field'])
    if fingerprint(field) != record['field_fingerprint']:
        raise ValueError('Reference field fingerprint changed')
    geometry_path = Path(record['cache'])/'geometry.npz'
    if file_digest(geometry_path) != record['hashes'][str(geometry_path)]:
        raise ValueError('Geometry cache changed')
    state_path = Path(record['state_cache'])
    if file_digest(state_path) != record['state_cache_sha256']:
        raise ValueError('Known original frame2,3 prefix changed')
    with np.load(geometry_path, allow_pickle=False) as z:
        encoded = np.array(z['encoded'], dtype=np.float32, copy=True)
        fluid_np = np.array(z['fluid'], dtype=bool, copy=True)
        interior = np.array(z['interior_yx'], dtype=np.int64, copy=True)
        boundary = np.array(z['boundary'], dtype=np.float32, copy=True)
    if encoded.ndim != 3 or encoded.shape[0] != 2 or not np.isfinite(encoded).all():
        raise ValueError('Invalid two-channel geometry encoding')
    if fluid_np.ndim != 2 or not fluid_np.any():
        raise ValueError('Invalid fluid mask')
    if interior.ndim != 2 or interior.shape[1] != 2 or not len(interior):
        raise ValueError('Empty or invalid physics interior')
    y, x = interior.T
    if (y < 0).any() or (x < 0).any() or (y >= fluid_np.shape[0]).any() or (x >= fluid_np.shape[1]).any() or not fluid_np[y, x].all():
        raise ValueError('Physics points outside fluid')
    if boundary.ndim != 2 or boundary.shape[1] != 2 or not len(boundary) or not np.isfinite(boundary).all():
        raise ValueError('Invalid obstacle boundary')
    with np.load(state_path, allow_pickle=False) as z:
        raw = np.array(z['features'], dtype=np.float32, copy=True)
    if raw.shape != (*fluid_np.shape, 6) or not np.isfinite(raw).all():
        raise ValueError('Invalid known state-cache features')
    reference = np.load(field, mmap_mode='r', allow_pickle=False)
    observed = read_training_observations(reference, fluid_np, device)
    fluid = torch.as_tensor(fluid_np, device=device)
    interior_gpu = torch.zeros_like(fluid)
    interior_gpu[y, x] = True
    case = dict(record=record, geometry=torch.as_tensor(encoded[None], device=device),
        fluid=fluid, fluid_np=fluid_np, interior=interior, interior_gpu=interior_gpu,
        boundary=boundary, fluid_yx=np.column_stack(np.where(fluid_np)),
        raw=torch.as_tensor(raw, device=device), observed=observed,
        normalization=normalization_from_observed(observed, fluid), reference=reference)
    if fingerprint(field) != record['field_fingerprint']:
        raise ValueError('Reference changed while reading TRAIN observations')
    return case
