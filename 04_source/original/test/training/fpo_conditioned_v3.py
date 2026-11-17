"""Conditional, data-assisted Navier--Stokes PINN; physical coordinate derivatives.

Not a causal time-stepper or a solver supplied only with initial conditions.
Geometry and Re are inputs; reference flow fields never enter its forward pass.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from fpo_common_v3 import ROOT, WORK, PILOT, REV, save, digest, verify, coords


def residual(q, pred, reynolds):
    def grad(v):
        return torch.autograd.grad(v, q, torch.ones_like(v), create_graph=True)[0]
    u, v, p = pred.unbind(1)
    gu, gv, gp = grad(u), grad(v), grad(p)
    lu = grad(gu[:, 0])[:, 0] + grad(gu[:, 1])[:, 1]
    lv = grad(gv[:, 0])[:, 0] + grad(gv[:, 1])[:, 1]
    return torch.stack((gu[:, 2] + u * gu[:, 0] + v * gu[:, 1] + gp[:, 0] - lu / reynolds,
                        gv[:, 2] + u * gv[:, 0] + v * gv[:, 1] + gp[:, 1] - lv / reynolds,
                        gu[:, 0] + gv[:, 1]), 1)


class ConditionalPINN(nn.Module):
    def __init__(self, config, mean, std):
        super().__init__()
        self.conditional = config['conditional']
        frequencies = config['frequencies']
        self.register_buffer('frequencies', torch.tensor(frequencies, dtype=torch.float32))
        self.register_buffer('lo', torch.tensor([4., 6., 0.]))
        self.register_buffer('span', torch.tensor([16., 4., 11.95]))
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.as_tensor(std, dtype=torch.float32))
        latent = config.get('latent', 32)
        if self.conditional:
            self.encoder = nn.Sequential(
                nn.Conv2d(2, 8, 3, padding=1), nn.Tanh(), nn.AvgPool2d(2),
                nn.Conv2d(8, 16, 3, padding=1), nn.Tanh(), nn.AvgPool2d(2),
                nn.AdaptiveAvgPool2d((4, 16)), nn.Flatten(),
                nn.Linear(16 * 4 * 16, latent), nn.Tanh())
        last = 3 + 6 * len(frequencies) + (latent + 1 if self.conditional else 0)
        layers = []
        for _ in range(config['depth']):
            layers += [nn.Linear(last, config['width']), nn.Tanh()]
            last = config['width']
        layers.append(nn.Linear(last, 3))
        self.net = nn.Sequential(*layers)

    def context(self, geometry, reynolds):
        if not self.conditional:
            return None
        z = self.encoder(geometry)
        # Dimensionless Re is input; no case ID, reference field, or validation label.
        re = torch.as_tensor(reynolds, dtype=z.dtype, device=z.device).reshape(-1, 1)
        return torch.cat((z, torch.log(re) / 8.), dim=1)

    def forward(self, q, context=None):
        s = (q - self.lo) / self.span
        features = [2 * s - 1]
        if self.frequencies.numel():
            phase = 2 * torch.pi * s.unsqueeze(-1) * self.frequencies
            features += [torch.sin(phase).flatten(1), torch.cos(phase).flatten(1)]
        if self.conditional:
            if context is None or context.shape[0] != 1:
                raise ValueError('One geometry context is required for a point batch')
            features.append(context.expand(q.shape[0], -1))
        return self.net(torch.cat(features, dim=1)) * self.std + self.mean


def errors(pred, ref):
    p = np.asarray(pred, dtype=np.float64).copy()
    r = np.asarray(ref, dtype=np.float64).copy()
    p[:, 2] -= p[:, 2].mean()
    r[:, 2] -= r[:, 2].mean()
    numerator = np.sum((p - r) ** 2, axis=0)
    denominator = np.sum(r * r, axis=0)
    return dict(velocity_relative_l2=float(np.sqrt(numerator[:2].sum() / max(denominator[:2].sum(), 1e-24))),
                pressure_relative_l2=float(np.sqrt(numerator[2] / max(denominator[2], 1e-24))))


def load_case(record, mmap=True):
    path = Path(record['cache'])
    with np.load(path / 'geometry.npz', allow_pickle=False) as z:
        geometry = z['encoded'].copy()
        fluid = z['fluid'].copy()
        boundary = z['boundary'].copy()
        interior_yx = z['interior_yx'].copy()
    with np.load(path / 'samples.npz', allow_pickle=False) as z:
        samples = {key: z[key].copy() for key in z.files}
    return dict(record=record, geometry=geometry, fluid=fluid, boundary=boundary,
                interior_yx=interior_yx, samples=samples,
                field=np.load(record['field'], mmap_mode='r', allow_pickle=False) if mmap else None)
