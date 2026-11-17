"""Scientific core checks; CPU-only and independent of the FlowBench labels."""
import io
import json
import argparse
import torch
from models import (chebyshev, PiNet, DendriteNet, RealNPU, ELM, SCN, CascadeCorrelation,
                    fit_elm, fit_scn, fit_cascade, make_head, project_field)


def run():
    torch.set_num_threads(2)
    torch.manual_seed(71001)
    checked = []
    x = torch.tensor([[.5, 1., -1., 0.]])
    expected = torch.tensor([[[-1., 1., -1., 0.]]]).reshape(1, 4)
    torch.testing.assert_close(chebyshev(x, 3)[:, :, 3], expected)
    checked.append('Chebyshev T3 polynomial values')
    xs = torch.linspace(-1, 1, 19)[:, None]
    pi, dd = PiNet(1, 1, 3), DendriteNet(1, 1, 3)
    with torch.no_grad():
        for model in [pi, dd]:
            for p in model.parameters(): p.fill_(1)
        for factor in pi.factors: factor.bias.zero_()
        pi.output.bias.zero_()
        dd.embed.bias.zero_();dd.output.bias.zero_()
    torch.testing.assert_close(pi(xs), (xs * (1 + xs).square()).expand(-1, 3))
    torch.testing.assert_close(dd(xs), xs.pow(3).expand(-1, 3))
    checked.append('PiNet and DD reproduce distinct analytic polynomials')
    power = RealNPU(2, 1)
    with torch.no_grad():
        power.gate.fill_(1);power.weight.copy_(torch.tensor([[1., -1.]]))
    arithmetic = torch.tensor([[2., 4.], [-2., 4.], [2., -4.], [-2., -4.]])
    torch.testing.assert_close(power.powers(arithmetic)[:, 0], arithmetic[:, 0] / arithmetic[:, 1], atol=2e-6, rtol=2e-6)
    checked.append('RealNPU signed division')
    cfg = dict(width=12, kan_width=8, degree=3, npu_width=8, constructive_width=8)
    for name in ['chebkan', 'pinet', 'dendrite', 'realnpu', 'rbf']:
        model = make_head(name, 5, cfg)
        z = torch.randn(17, 5, requires_grad=True)
        result = model(z)
        assert result.shape == (17, 3) and torch.isfinite(result).all()
        result.square().mean().backward()
        assert z.grad is not None and torch.isfinite(z.grad).all()
        checked.append(name + ' finite forward/backward')
    x = torch.randn(192, 3)
    y = torch.stack((torch.sin(x[:, 0]), x[:, 0] * x[:, 1], x[:, 2].square()), 1)
    for name in ['elm', 'scn', 'cascor']:
        model = make_head(name, 3, cfg)
        if name == 'elm': history = fit_elm(model, x, y)
        elif name == 'scn': history = fit_scn(model, x, y, width=6, pool=24)
        else: history = fit_cascade(model, x, y, width=4, pool=4, candidate_steps=12)
        assert history and torch.isfinite(model(x)).all()
        if name == 'scn': assert len(model.weight) > 0
        if name == 'cascor': assert all(not p.requires_grad for unit in model.units for p in unit.parameters())
        memory = io.BytesIO();torch.save(model.state_dict(), memory);memory.seek(0)
        state = torch.load(memory, weights_only=True)
        if name == 'scn': restored = SCN(3, len(model.weight))
        elif name == 'cascor': restored = CascadeCorrelation(3, len(model.units))
        else: restored = ELM(3, cfg['constructive_width'])
        restored.load_state_dict(state)
        torch.testing.assert_close(restored(x), model(x))
        checked.append(name + ' fitting and checkpoint roundtrip')
    field = torch.randn(9, 9, 3)
    fluid = torch.ones(9, 9, dtype=torch.bool);fluid[3:5, 3:5] = False
    gauged = project_field(field, fluid)
    assert gauged[~fluid].count_nonzero() == 0
    assert abs(float(gauged[:, :, 2][fluid].mean())) < 1e-6
    shifted = field.clone();shifted[:, :, 2] += 100
    torch.testing.assert_close(project_field(shifted, fluid), gauged, atol=2e-5, rtol=2e-5)
    checked.append('prediction-only gauge and solid mask invariance')
    import numpy as np
    from data_metrics import field_metrics
    q = (np.arange(512) + .5) / 512
    yy, xx = np.meshgrid(q, q, indexing='ij')
    exact = np.stack((4 * yy * (1 - yy), np.zeros_like(xx), -8 * xx / 100), -1)
    metrics = field_metrics(exact, exact, np.ones((512, 512), dtype=bool), 100)
    assert metrics['all_cell_mse'] == 0
    assert metrics['prediction_physics']['divergence_rms'] < 1e-10
    assert metrics['prediction_physics']['steady_ns_momentum_rms'] < 1e-9
    checked.append('finite-difference diagnostics recover analytic Poiseuille solution')
    from neuralop.models import TFNO
    tfno = TFNO(n_modes=(4, 4), hidden_channels=8, in_channels=3, out_channels=3, n_layers=2,
                projection_channel_ratio=4, factorization='Tucker', rank=.1)
    result = tfno(torch.randn(1, 3, 16, 16))
    assert result.shape == (1, 3, 16, 16) and torch.isfinite(result).all()
    result.square().mean().backward()
    checked.append('pinned TFNO finite forward/backward')
    return dict(status='CORE_CHECKS_PASS', checks=checked, count=len(checked))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--data-config')
    args = parser.parse_args()
    result = run()
    if args.data_config:
        from data_metrics import read, audit_data
        result['data_audit'] = audit_data(read(args.data_config))
    print(json.dumps(result, indent=2))
