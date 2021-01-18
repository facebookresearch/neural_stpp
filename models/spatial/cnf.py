# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint
import diffeq_layers


def divergence_bf(f, y, training, **unused_kwargs):
    sum_diag = 0.
    for i in range(f.shape[1]):
        retain_graph = training or i < (f.shape[1] - 1)
        sum_diag += torch.autograd.grad(f[:, i].sum(), y, create_graph=training, retain_graph=retain_graph)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()


def divergence_approx(f, y, training, nblocks=1, e=None, **unused_kwargs):
    assert e is not None
    dim = f.shape[1]
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=training, retain_graph=training)[0][:, :dim].contiguous()
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx, e_dzdx


class TimeVariableCNF(nn.Module):

    start_time = 0.0
    end_time = 1.0

    def __init__(self, func, dim, tol=1e-6, method="dopri5", nonself_connections=False, energy_regularization=0.0, jacnorm_regularization=0.0):
        super().__init__()
        self.func = func
        self.dim = dim
        self.tol = tol
        self.method = method
        self.nonself_connections = nonself_connections
        self.energy_regularization = energy_regularization
        self.jacnorm_regularization = jacnorm_regularization
        self.nfe = 0

    def integrate(self, t0, t1, x, logpx, tol=None, method=None, norm=None, intermediate_states=0):
        """
        Args:
            t0: (N,)
            t1: (N,)
            x: (N, ...)
            logpx: (N,)
        """
        self.nfe = 0

        tol = tol or self.tol
        method = method or self.method
        e = torch.randn_like(x)[:, :self.dim]
        energy = torch.zeros(1).to(x)
        jacnorm = torch.zeros(1).to(x)
        initial_state = (t0, t1, e, x, logpx, energy, jacnorm)

        if intermediate_states > 1:
            tt = torch.linspace(self.start_time, self.end_time, intermediate_states).to(t0)
        else:
            tt = torch.tensor([self.start_time, self.end_time]).to(t0)

        solution = odeint_adjoint(
            self,
            initial_state,
            tt,
            rtol=tol,
            atol=tol,
            method=method,
        )

        if intermediate_states > 1:
            y = solution[3]
            _, _, _, _, logpy, energy, jacnorm = tuple(s[-1] for s in solution)
        else:
            _, _, _, y, logpy, energy, jacnorm = tuple(s[-1] for s in solution)

        regularization = (
            self.energy_regularization * (energy - energy.detach()) +
            self.jacnorm_regularization * (jacnorm - jacnorm.detach())
        )

        return y, logpy + regularization  # hacky method to introduce regularization.

    def forward(self, s, state):
        """Solves the same dynamics but uses a dummy variable that always integrates [0, 1]."""
        self.nfe += 1
        t0, t1, e, x, logpx, _, _ = state

        ratio = (t1 - t0) / (self.end_time - self.start_time)
        t = (s - self.start_time) * ratio + t0

        vjp = None
        with torch.enable_grad():
            x = x.requires_grad_(True)

            dx = self.func(t, x)
            dx = dx * ratio.reshape(-1, *([1] * (x.ndim - 1)))

            if self.nonself_connections:
                dx_div = self.func(t, x, rm_nonself_grads=True)
                dx_div = dx_div * ratio.reshape(-1, *([1] * (x.ndim - 1)))
            else:
                dx_div = dx

            # Use brute force trace for testing.
            if not self.training:
                div = divergence_bf(dx_div[:, :self.dim], x, self.training)
            else:
                vjp = torch.autograd.grad(dx_div[:, :self.dim], x, e, create_graph=self.training, retain_graph=self.training)[0]
                vjp = vjp[:, :self.dim]
                div = torch.sum(vjp * e, dim=1)

            # Debugging code for checking gradient connections.
            # Need to send T and N to self from attncnf.
            # if self.training and hasattr(self, "T"):
            #     grads = torch.autograd.grad(dx_div.reshape(self.T, self.N, -1)[5, 0, 0], x, retain_graph=True)[0]
            #     print(grads.reshape(self.T, self.N, -1)[4:6, 0, :])

        if not self.training:
            dx = dx.detach()
            div = div.detach()

        d_energy = torch.sum(dx * dx).reshape(1) / x.shape[0]

        if self.training:
            d_jacnorm = torch.sum(vjp * vjp).reshape(1) / x.shape[0]
        else:
            d_jacnorm = torch.zeros(1).to(x)

        return torch.zeros_like(t0), torch.zeros_like(t1), torch.zeros_like(e), dx, -div, d_energy, d_jacnorm

    def extra_repr(self):
        return f"method={self.method}, tol={self.tol}, energy={self.energy_regularization}, jacnorm={self.jacnorm_regularization}"


def rms_norm(tensor):
    return tensor.pow(2).mean().sqrt()


def ignore_norm(tensor):
    return torch.zeros(1).to(tensor)


def wrap_norm(norm_fns, shapes):
    def _norm(tensor):
        total = 0
        out = []
        for i, shape in enumerate(shapes):
            next_total = total + shape.numel()
            if i < len(norm_fns):
                out.append(norm_fns[i](tensor[total:next_total]))
            else:
                out.append(ignore_norm(tensor[total:next_total]))
            total = next_total
        assert total == tensor.numel(), "Shapes do not total to the full size of the tensor."
        return max(out)
    return _norm


def max_rms_norm(shapes):
    def _norm(tensor):
        total = 0
        out = []
        for shape in shapes:
            next_total = total + shape.numel()
            out.append(rms_norm(tensor[total:next_total]))
            total = next_total
        assert total == tensor.numel(), "Shapes do not total to the full size of the tensor."
        return max(out)
    return _norm


def rms_norm_first_elem(shapes):
    def _norm(tensor):
        total = 0
        out = []
        for i, shape in enumerate(shapes):
            next_total = total + shape.numel()
            if i == 0:
                out.append(rms_norm(tensor[total:next_total]))
            else:
                out.append(ignore_norm(tensor[total:next_total]))
            total = next_total
        assert total == tensor.numel(), "Shapes do not total to the full size of the tensor."
        return max(out)
    return _norm


ACTFNS = {
    "softplus": (lambda dim: nn.Softplus()),
    "swish": (lambda dim: diffeq_layers.TimeDependentSwish(dim)),
}

LAYERTYPES = {
    "concatsquash": diffeq_layers.ConcatSquashLinear,
    "concat": diffeq_layers.ConcatLinear_v2,
}


def build_fc_odefunc(dim=2, hidden_dims=[64, 64, 64], out_dim=None, nonzero_dim=None, actfn="softplus", layer_type="concatsquash",
                     zero_init=True, actfirst=False):
    assert layer_type in LAYERTYPES.keys(), f"layer_type must be one of {LAYERTYPES.keys()} but was given {layer_type}"
    layer_fn = LAYERTYPES[layer_type]

    nonzero_dim = dim if nonzero_dim is None else nonzero_dim
    out_dim = out_dim or dim
    if hidden_dims:
        dims = [dim] + list(hidden_dims)
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(layer_fn(d_in, d_out))
            layers.append(ACTFNS[actfn](d_out))
        layers.append(layer_fn(hidden_dims[-1], out_dim))
    else:
        layers = [layer_fn(dim, out_dim)]

    if actfirst and len(layers) > 1:
        layers = layers[1:]

    if nonzero_dim < dim:
        # zero out weights for auxiliary inputs.
        layers[0]._layer.weight.data[:, nonzero_dim:].fill_(0)

    if zero_init:
        for m in layers[-1].modules():
            if isinstance(m, nn.Linear):
                m.weight.data.fill_(0)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    return diffeq_layers.SequentialDiffEq(*layers)
