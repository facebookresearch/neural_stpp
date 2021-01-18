# Copyright (c) Facebook, Inc. and its affiliates.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad


class PlanarFlow(nn.Module):

    def __init__(self, nd, activation=torch.tanh):
        super(PlanarFlow, self).__init__()
        self.nd = nd
        self.register_buffer("one", torch.ones(1))
        self.activation = activation

        self.register_parameter('_u', nn.Parameter(torch.randn(self.nd)))
        self.register_parameter('w', nn.Parameter(torch.randn(self.nd)))
        self.register_parameter('b', nn.Parameter(torch.randn(1)))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.nd)
        self._u.data.uniform_(-stdv, stdv)
        self.w.data.uniform_(-stdv, stdv)
        self.b.data.fill_(0)

    @property
    def u(self):
        def m(a):
            return F.softplus(a) - 1.0
        wu = torch.dot(self._u, self.w)
        return self._u + (m(wu) + wu) * self.w / (torch.norm(self.w, p=2)**2 + 1e-8)

    def forward(self, x, logpx=None, reverse=False, **kwargs):
        if reverse:
            raise ValueError(f"{self.__class__.__name__} does not support reverse.")

        with torch.enable_grad():
            x.requires_grad_(True)
            h = self.activation(
                torch.mm(x, self.w.view(self.nd, 1)) + self.b
            )
        f = x + self.u.expand_as(x) * h
        if logpx is not None:
            logpy = logpx - self._logdetgrad(h, x)
            return f, logpy
        else:
            return f

    def _logdetgrad(self, h, x):
        """Computes |det df/dz|"""
        psi = grad(h, x, grad_outputs=self.one.expand_as(h).type_as(h).detach(),
                   create_graph=True, only_inputs=True)[0]
        u_dot_psi = torch.mm(psi, self.u.view(self.nd, 1)).squeeze(-1)
        detgrad = 1 + u_dot_psi
        return torch.log(detgrad + 1e-8)


class RadialFlow(nn.Module):

    def __init__(self, nd, hypernet=False):
        super().__init__()
        self.nd = nd
        self.hypernet = hypernet

        if not hypernet:
            self.register_parameter('z0', nn.Parameter(torch.randn(self.nd)))
            self.register_parameter('log_alpha', nn.Parameter(torch.randn(1)))
            self.register_parameter('_beta', nn.Parameter(torch.randn(1)))

    @property
    def beta(self):
        return -torch.exp(self.log_alpha) + F.softplus(self._beta)

    def forward(self, x, logpx=None, reverse=False, z0=None, log_alpha=None, beta=None, **kwargs):
        if reverse:
            raise ValueError(f"{self.__class__.__name__} does not support reverse.")

        if self.hypernet:
            assert z0 is not None and log_alpha is not None and beta is not None
            beta = (-torch.exp(log_alpha) + F.softplus(beta))
        else:
            z0 = self.z0
            log_alpha = self.log_alpha
            beta = (-torch.exp(log_alpha) + F.softplus(self._beta))

        z0 = z0.expand_as(x)
        r = torch.norm(x - z0, dim=-1, keepdim=True)
        h = 1 / (torch.exp(log_alpha) + r)
        f = x + beta * h * (x - z0)

        if logpx is not None:
            logdetgrad = (self.nd - 1) * torch.log(1 + beta * h) + \
                torch.log(1 + beta * h - beta * r / (torch.exp(log_alpha) + r) ** 2)
            logpy = logpx - logdetgrad.reshape(-1)
            return f, logpy
        else:
            return f


class HypernetworkRadialFlow(nn.Module):

    def __init__(self, nd, cond_dim, nflows=1):
        super().__init__()
        self.nd = nd
        self.nflows = nflows

        self.radial_flows = nn.ModuleList([RadialFlow(nd, hypernet=True) for _ in range(nflows)])

        self.hypernet = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, (self.nd + 2) * nflows),
        )

    def forward(self, x, logpx=None, reverse=False, cond=None, **kwargs):
        hyper_out = self.hypernet(cond)
        out = (x, logpx)
        for i in range(self.nflows):
            start_ind = (self.nd + 2) * i

            z0 = hyper_out[:, start_ind:start_ind + self.nd]
            log_alpha = hyper_out[:, start_ind + self.nd:start_ind + self.nd + 1] - 6.0
            beta = hyper_out[:, start_ind + self.nd + 1:start_ind + self.nd + 2] - 6.0
            out = self.radial_flows[i](*out, reverse=reverse, z0=z0, log_alpha=log_alpha, beta=beta)
        return out


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta)))
