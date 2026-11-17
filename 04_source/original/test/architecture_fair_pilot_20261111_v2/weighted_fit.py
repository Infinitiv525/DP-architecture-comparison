"""Full-row weighted streaming least squares and constructive work units.

The three 64-output blocks carry their native physical relative-error weights.
No data point is dropped. Normal equations are accumulated in FP64; the stored
model remains FP32. Ridge is explicit and the intercept is unpenalized.
"""
import math
import time
import torch
from torch import nn
from torch.nn import functional as F
from common import patch_loss


class WeightedNormalEquations:
    def __init__(self, columns, outputs=192, device='cpu'):
        assert outputs==192
        self.a=torch.zeros(3,columns,columns,dtype=torch.float64,device=device)
        self.b=torch.zeros(columns,outputs,dtype=torch.float64,device=device)
        self.cases=0; self.rows=0

    def add(self, basis, target, weight):
        h=basis.double(); y=target.double(); w=weight.double()
        gram=h.T@h
        for c in range(3):
            sl=slice(c*64,(c+1)*64)
            self.a[c].add_(gram,alpha=float(w[c]))
            self.b[:,sl].add_(h.T@y[:,sl],alpha=float(w[c]))
        self.cases+=1; self.rows+=len(h)

    def solve(self, ridge=1e-8):
        if self.cases==0: raise ValueError('No training data in weighted fit')
        a=self.a/self.cases; b=self.b/self.cases
        penalty=torch.eye(a.shape[-1],dtype=a.dtype,device=a.device)*ridge
        penalty[0,0]=0
        beta=torch.empty_like(b)
        residuals=[]
        for c in range(3):
            sl=slice(c*64,(c+1)*64); matrix=a[c]+penalty
            factor,info=torch.linalg.cholesky_ex(matrix)
            if int(info)==0: value=torch.cholesky_solve(b[:,sl],factor)
            else: value=torch.linalg.solve(matrix,b[:,sl])
            beta[:,sl]=value
            residuals.append(float((matrix@value-b[:,sl]).norm()/b[:,sl].norm().clamp_min(1e-20)))
        if not torch.isfinite(beta).all(): raise FloatingPointError('Nonfinite weighted fit')
        return beta.float(),residuals


def fit_output(core, iterator, ridge=1e-8):
    """Iterator factory must visit every complete TRAIN case once, in order."""
    equations=None
    with torch.no_grad():
        for x,y,w in iterator():
            h=core.basis(x)
            if equations is None: equations=WeightedNormalEquations(h.shape[1],device=x.device)
            equations.add(h,y,w)
        beta,residuals=equations.solve(ridge)
        core.beta=beta
    return dict(cases=equations.cases,rows=equations.rows,normal_equation_relative_residual=residuals,ridge=ridge)


def scn_candidate(core, iterator, pool=16, seed=0):
    """Weighted SC-III admissibility over every native output and TRAIN row.

    Residuals with numerically zero weighted energy impose no inequality.
    One accepted node is returned, never an untested random replacement.
    """
    generator=torch.Generator(device=core.weight.device).manual_seed(seed)
    tests=0
    with torch.no_grad():
        for radius in [.5,1.,2.,4.]:
            w=(torch.rand(pool,665,device=core.weight.device,generator=generator)*2-1)*radius/math.sqrt(665)
            b=(torch.rand(pool,device=w.device,generator=generator)*2-1)*radius
            cross=torch.zeros(pool,192,dtype=torch.float64,device=w.device)
            norm=torch.zeros(pool,3,dtype=torch.float64,device=w.device)
            energy=torch.zeros(192,dtype=torch.float64,device=w.device)
            for x,y,weight in iterator():
                h=torch.tanh(F.linear(x,w,b)).double(); residual=(y-core(x)).double()
                weights=weight.repeat_interleave(64).double()
                cross+=h.T@(residual*weights)
                norm+=h.square().sum(0)[:,None]*weight.double()[None]
                energy+=(residual.square()*weights).sum(0)
            projection=cross.square()/norm.repeat_interleave(64,1).clamp_min(1e-30)
            active=energy>1e-20
            if not active.any(): return None,dict(stop='zero_residual',candidate_tests=tests)
            for relaxation in [.9,.99,.999,.9999,.99999,.9999999]:
                tests+=pool
                xi=projection-(1-relaxation)*energy[None]
                valid=(xi[:,active]>0).all(1)
                if valid.any():
                    scores=xi[:,active].sum(1).masked_fill(~valid,-torch.inf)
                    k=int(scores.argmax())
                    return (w[k:k+1],b[k:k+1]),dict(radius=radius,relaxation=relaxation,
                        minimum_xi=float(xi[k,active].min()),candidate_tests=tests)
    return None,dict(stop='no_admissible_candidate',candidate_tests=tests)


def append_scn(core, node):
    weight,bias=node
    core.weight=torch.cat((core.weight,weight)); core.bias=torch.cat((core.bias,bias))


def cascade_candidate(core, iterator, pool=8, epochs=1, learning_rate=.01):
    """A CasCor candidate trained on all rows, with frozen prior units.

    Adam follows case minibatches. Each candidate epoch visits ALL complete
    TRAIN fields. The objective is per-case weighted residual covariance;
    final candidate selection aggregates that covariance over all TRAIN cases.
    This explicit minibatch adaptation differs from the original full-batch
    CasCor optimizer. Global output refitting still uses the common field loss.
    """
    device=core.beta.device; dimensions=core.beta.shape[0]-1
    candidate=nn.Linear(dimensions,pool,device=device)
    optimizer=torch.optim.Adam(candidate.parameters(),lr=learning_rate)
    cases=rows=0
    for _ in range(epochs):
        for x,y,weight in iterator():
            with torch.no_grad():
                expanded=core.basis(x)[:,1:]
                residual=y-core(x); residual=residual-residual.mean(0)
                weighted=residual*weight.repeat_interleave(64)
            h=torch.tanh(candidate(expanded)); centered=h-h.mean(0)
            correlation=(centered.T@weighted).abs().sum(1)
            loss=-correlation.mean()
            if not torch.isfinite(loss): raise FloatingPointError('CasCor candidate covariance nonfinite')
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            cases+=1; rows+=len(x)
    score=torch.zeros(pool,device=device)
    with torch.no_grad():
        for x,y,weight in iterator():
            expanded=core.basis(x)[:,1:]; residual=y-core(x); residual-=residual.mean(0)
            h=torch.tanh(candidate(expanded)); h-=h.mean(0)
            score+=(h.T@(residual*weight.repeat_interleave(64))).abs().sum(1)
        k=int(score.argmax()); unit=nn.Linear(dimensions,1,device=device)
        unit.weight.copy_(candidate.weight[k:k+1]); unit.bias.copy_(candidate.bias[k:k+1])
    return unit,dict(candidate_correlation=float(score[k]),candidate_epochs=epochs,
        candidate_train_case_visits=cases,candidate_train_row_visits=rows)
