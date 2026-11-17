"""Small, explicitly adapted architecture cores for a technical TRAIN/VAL pilot.

Independent mathematical implementations. Author TFNO is supplied by neuralop;
the archived PINN is loaded separately. No claim of a published-model replication.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def chebyshev(x, degree):
    terms = [torch.ones_like(x), x]
    for _ in range(2, degree + 1):
        terms.append(2 * x * terms[-1] - terms[-2])
    return torch.stack(terms[:degree + 1], -1)


class ChebLayer(nn.Module):
    def __init__(self, inputs, outputs, degree=3):
        super().__init__()
        self.degree = degree
        self.weight = nn.Parameter(torch.randn(outputs, inputs, degree + 1) / math.sqrt(inputs * (degree + 1)))
        self.bias = nn.Parameter(torch.zeros(outputs))

    def forward(self, x):
        basis = chebyshev(torch.tanh(x), self.degree).flatten(1)
        return F.linear(basis, self.weight.flatten(1), self.bias)


class ChebKAN(nn.Module):
    def __init__(self, inputs, width=32, degree=3):
        super().__init__()
        self.layers = nn.Sequential(ChebLayer(inputs, width, degree), ChebLayer(width, width, degree), ChebLayer(width, 3, degree))

    def forward(self, x):
        return self.layers(x)


class PiNet(nn.Module):
    """Coupled CP-style polynomial recurrence: h <- h * (1 + U_l x)."""
    def __init__(self, inputs, width=96, degree=3):
        super().__init__()
        self.factors = nn.ModuleList(nn.Linear(inputs, width) for _ in range(degree))
        self.output = nn.Linear(width, 3)

    def forward(self, x):
        h = self.factors[0](x)
        for factor in self.factors[1:]:
            h = h * (1 + factor(x))
        return self.output(h)


class DendriteNet(nn.Module):
    """DD-style recurrence: h <- (W_l h) * embedded_input; no hidden activation."""
    def __init__(self, inputs, width=96, degree=3):
        super().__init__()
        self.embed = nn.Linear(inputs, width)
        self.layers = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(degree - 1))
        self.output = nn.Linear(width, 3)

    def forward(self, x):
        base = self.embed(x)
        h = base
        for layer in self.layers:
            h = layer(h) * base
        return self.output(h)


class RealNPU(nn.Module):
    """Heim et al. eq.14/15 with imaginary weights zero; affine output adaptation."""
    def __init__(self, inputs, width=32, eps=1e-7):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(width, inputs))
        nn.init.xavier_uniform_(self.weight)
        self.gate = nn.Parameter(torch.full((inputs,), .5))
        self.output = nn.Linear(width, 3)

    def powers(self, x):
        gate = self.gate.clamp(0, 1)
        radius = gate * (x.abs() + self.eps) + (1 - gate)
        angle = (x < 0).to(x.dtype) * gate
        return torch.exp(F.linear(radius.log(), self.weight)) * torch.cos(math.pi * F.linear(angle, self.weight))

    def forward(self, x):
        return self.output(self.powers(x))

    def project_parameters(self):
        with torch.no_grad():
            self.gate.clamp_(0, 1)


class RBF(nn.Module):
    def __init__(self, inputs, width=96):
        super().__init__()
        self.centres = nn.Parameter(torch.randn(width, inputs) * .3)
        self.log_width = nn.Parameter(torch.zeros(width))
        self.output = nn.Linear(width, 3)

    def forward(self, x):
        distances = (x.square().sum(1, keepdim=True) + self.centres.square().sum(1)[None] - 2 * x @ self.centres.T).clamp_min(0) / x.shape[1]
        widths = F.softplus(self.log_width) + 1e-4
        return self.output(torch.exp(-distances / (2 * widths.square())))


class ELM(nn.Module):
    def __init__(self, inputs, width=32):
        super().__init__()
        self.register_buffer('weight', torch.randn(width, inputs) / math.sqrt(inputs))
        self.register_buffer('bias', torch.rand(width) * 2 - 1)
        self.register_buffer('beta', torch.zeros(width + 1, 3))

    def basis(self, x):
        return torch.cat((torch.ones_like(x[:, :1]), torch.tanh(F.linear(x, self.weight, self.bias))), 1)

    def forward(self, x):
        return self.basis(x) @ self.beta


class SCN(nn.Module):
    def __init__(self, inputs, units=0):
        super().__init__()
        self.register_buffer('weight', torch.zeros(units, inputs))
        self.register_buffer('bias', torch.zeros(units))
        self.register_buffer('beta', torch.zeros(units + 1, 3))

    def basis(self, x):
        return torch.cat((torch.ones_like(x[:, :1]), torch.tanh(F.linear(x, self.weight, self.bias))), 1)

    def forward(self, x):
        return self.basis(x) @ self.beta


class CascadeCorrelation(nn.Module):
    def __init__(self, inputs, units=0):
        super().__init__()
        self.inputs = inputs
        self.units = nn.ModuleList()
        for i in range(units):
            self.add(nn.Linear(inputs + i, 1))
        self.register_buffer('beta', torch.zeros(inputs + units + 1, 3))

    def add(self, unit):
        unit.requires_grad_(False)
        self.units.append(unit)

    def basis(self, x):
        h = x
        for unit in self.units:
            h = torch.cat((h, torch.tanh(unit(h))), 1)
        return torch.cat((torch.ones_like(x[:, :1]), h), 1)

    def forward(self, x):
        return self.basis(x) @ self.beta


def ridge_fit(h, y, ridge=1e-4):
    """FP64 regularized normal equations; intercept is not penalized."""
    a, b = h.double(), y.double()
    regularizer = torch.eye(a.shape[1], device=a.device, dtype=a.dtype) * ridge
    regularizer[0, 0] = 0
    beta = torch.linalg.solve(a.T @ a + regularizer, a.T @ b)
    if not torch.isfinite(beta).all():
        raise FloatingPointError('Nonfinite least-squares coefficients')
    return beta.to(h.dtype)


def fit_elm(model, x, y, ridge=1e-4):
    with torch.no_grad():
        model.beta = ridge_fit(model.basis(x), y, ridge)
    return [{'units': len(model.weight), 'mse': float((model(x) - y).square().mean())}]


def fit_scn(model, x, y, width=32, pool=32, ridge=1e-4):
    """SC-III-style residual acceptance for every output; ridge is explicit adaptation."""
    history = []
    with torch.no_grad():
        model.beta = ridge_fit(model.basis(x), y, ridge)
        for step in range(width):
            residual = y - model(x)
            energy = residual.square().sum(0).clamp_min(1e-20)
            accepted = None
            attempts = 0
            for radius in [.5, 1., 2., 4.]:
                w = (torch.rand(pool, x.shape[1], device=x.device) * 2 - 1) * radius / math.sqrt(x.shape[1])
                b = (torch.rand(pool, device=x.device) * 2 - 1) * radius
                h = torch.tanh(F.linear(x, w, b))
                squared_projection = (h.T @ residual).square() / h.square().sum(0).clamp_min(1e-20)[:, None]
                for r in [.9, .99, .999, .9999, .99999]:
                    attempts += pool
                    xi = squared_projection - (1 - r) * energy[None]
                    valid = (xi > 0).all(1)
                    if valid.any():
                        score = xi.sum(1).masked_fill(~valid, -torch.inf)
                        k = int(score.argmax())
                        accepted = (w[k:k+1], b[k:k+1], r, float(xi[k].min()))
                        break
                if accepted is not None:
                    break
            if accepted is None:
                history.append({'units': step, 'stop': 'no_admissible_candidate', 'candidate_tests': attempts})
                break
            w, b, r, minimum_xi = accepted
            model.weight = torch.cat((model.weight, w))
            model.bias = torch.cat((model.bias, b))
            model.beta = ridge_fit(model.basis(x), y, ridge)
            history.append({'units': step + 1, 'mse': float((model(x) - y).square().mean()), 'r': r,
                            'minimum_xi': minimum_xi, 'candidate_tests': attempts})
    return history


def fit_cascade(model, x, y, width=32, pool=8, candidate_steps=30, ridge=1e-4):
    history = []
    with torch.no_grad():
        model.beta = ridge_fit(model.basis(x), y, ridge)
    for step in range(width):
        with torch.no_grad():
            expanded = model.basis(x)[:, 1:].detach()
            residual = (y - model(x)).detach()
            residual -= residual.mean(0)
        candidate = nn.Linear(expanded.shape[1], pool).to(x.device)
        optimizer = torch.optim.Adam(candidate.parameters(), lr=.01)
        for _ in range(candidate_steps):
            h = torch.tanh(candidate(expanded))
            hc = h - h.mean(0)
            correlation = (hc.T @ residual / len(x)).abs().sum(1)
            loss = -correlation.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite CasCor candidate objective')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            h = torch.tanh(candidate(expanded))
            score = ((h - h.mean(0)).T @ residual / len(x)).abs().sum(1)
            k = int(score.argmax())
            unit = nn.Linear(expanded.shape[1], 1).to(x.device)
            unit.weight.copy_(candidate.weight[k:k+1])
            unit.bias.copy_(candidate.bias[k:k+1])
            model.add(unit)
            model.beta = ridge_fit(model.basis(x), y, ridge)
            history.append({'units': step + 1, 'mse': float((model(x) - y).square().mean()),
                            'candidate_correlation': float(score[k]), 'candidate_gradient_steps': candidate_steps})
    return history


def make_head(name, inputs, config):
    if name == 'chebkan': return ChebKAN(inputs, config['kan_width'], config['degree'])
    if name == 'pinet': return PiNet(inputs, config['width'], config['degree'])
    if name == 'dendrite': return DendriteNet(inputs, config['width'], config['degree'])
    if name == 'realnpu': return RealNPU(inputs, config['npu_width'])
    if name == 'rbf': return RBF(inputs, config['width'])
    if name == 'elm': return ELM(inputs, config['constructive_width'])
    if name == 'scn': return SCN(inputs)
    if name == 'cascor': return CascadeCorrelation(inputs)
    raise ValueError(name)


def project_field(value, fluid):
    mask = fluid.to(value.dtype)
    pressure = value[:, :, 2] - (value[:, :, 2] * mask).sum() / mask.sum()
    return torch.cat((value[:, :, :2], pressure[:, :, None]), 2) * mask[:, :, None]


class FixedGeometryFeatures(nn.Module):
    """Technical pilot only: deterministic geometry pooling, no pretrained encoder."""
    def __init__(self, normalization, size=512):
        super().__init__()
        self.size = size
        self.register_buffer('geometry_mean', torch.tensor(normalization['geometry_mean']).float()[:, None, None])
        self.register_buffer('geometry_std', torch.tensor(normalization['geometry_std']).float()[:, None, None])
        v = (torch.arange(size).float() + .5) / size
        yy, xx = torch.meshgrid(v, v, indexing='ij')
        q = torch.stack((xx.flatten(), yy.flatten()), 1)
        phase = 2 * math.pi * q[:, :, None] * torch.tensor([1., 2., 4., 8., 16.])
        self.register_buffer('coordinates', q, persistent=False)
        self.register_buffer('static', torch.cat((2 * q - 1, phase.sin().flatten(1), phase.cos().flatten(1)), 1), persistent=False)
        self.inputs = self.static.shape[1] + 32 + 2 + 1

    def forward(self, known, indices=None):
        geometry = (known['geometry'] - self.geometry_mean) / self.geometry_std
        pooled = F.adaptive_avg_pool2d(geometry[None], (4, 4)).flatten()
        local = geometry.permute(1, 2, 0).reshape(-1, 2)
        static = self.static
        if indices is not None:
            static = static[indices]
            local = local[indices]
        reynolds = known['re'].log().reshape(1) / 8
        context = torch.cat((pooled, reynolds))[None].expand(len(local), -1)
        return torch.cat((static, context, local), 1)


class PointPipeline(nn.Module):
    def __init__(self, name, normalization, config):
        super().__init__()
        self.features = FixedGeometryFeatures(normalization)
        self.core = make_head(name, self.features.inputs, config)
        self.register_buffer('mean', torch.tensor(normalization['target_mean']).float())
        self.register_buffer('std', torch.tensor(normalization['target_std']).float())

    def forward(self, known):
        normalized = self.core(self.features(known))
        physical = (normalized * self.std + self.mean).reshape(512, 512, 3)
        return project_field(physical, known['fluid'])


class AuthorTFNOPipeline(nn.Module):
    """Author README architecture; pinned neuraloperator2 API and explicit defaults."""
    def __init__(self, normalization, config):
        super().__init__()
        from neuralop.models import TFNO
        self.core = TFNO(n_modes=tuple(config['n_modes']), n_layers=config['n_layers'],
                         hidden_channels=config['hidden_channels'], in_channels=3, out_channels=3,
                         projection_channel_ratio=4, lifting_channel_ratio=2,
                         factorization='Tucker', rank=.1, implementation='factorized',
                         positional_embedding='grid', fno_block_precision='full')
        self.register_buffer('geometry_mean', torch.tensor(normalization['geometry_mean']).float()[:, None, None])
        self.register_buffer('geometry_std', torch.tensor(normalization['geometry_std']).float()[:, None, None])
        self.register_buffer('mean', torch.tensor(normalization['target_mean']).float())
        self.register_buffer('std', torch.tensor(normalization['target_std']).float())

    def forward(self, known):
        geometry = (known['geometry'] - self.geometry_mean) / self.geometry_std
        re = (known['re'].log() / 8).expand(1, 512, 512)
        raw = self.core(torch.cat((re, geometry), 0)[None])[0].permute(1, 2, 0)
        return project_field(raw * self.std + self.mean, known['fluid'])
