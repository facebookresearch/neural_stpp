# Copyright (c) Facebook, Inc. and its affiliates.

import math
import torch
import torch.nn as nn

import flow_layers
from .cnf import TimeVariableCNF, build_fc_odefunc, max_rms_norm


class AuxODEFunc(nn.Module):

    def __init__(self, func, dim, aux_dim, aux_odefunc, time_offset):
        super().__init__()
        self.func = func
        self.dim = dim
        self.aux_dim = aux_dim
        self.aux_odefunc = aux_odefunc
        self.time_offset = time_offset

    def forward(self, t, state):
        x, h = state[:, :self.dim], state[:, self.dim:]
        a = h[:, -self.aux_dim:]  # Only use a subset of h if aux_dim < h dim
        dx = self.func(t, torch.cat([x, a], dim=1))
        dh = self.aux_odefunc(t - self.time_offset, h)
        return torch.cat([dx, dh], dim=1)


class JumpCNF(nn.Module):

    time_offset = 2.0

    def __init__(self, dim=2, hidden_dims=[64, 64, 64], aux_dim=0, aux_odefunc=None, layer_type="concat", actfn="softplus", zero_init=True, tol=1e-4, otreg_strength=0.0):
        super().__init__()

        self.aux_dim = aux_dim
        func = build_fc_odefunc(dim + self.aux_dim, hidden_dims, out_dim=dim, nonzero_dim=dim, layer_type=layer_type, actfn=actfn, zero_init=zero_init)

        if self.aux_dim > 0:
            assert aux_odefunc is not None
            odefunc = AuxODEFunc(func, dim, aux_dim, aux_odefunc, self.time_offset)
        else:
            odefunc = func

        assert isinstance(odefunc, nn.Module)

        self.cnf = TimeVariableCNF(odefunc, dim, tol=tol, method="dopri5", energy_regularization=otreg_strength, jacnorm_regularization=otreg_strength)

        self.inst_flow = flow_layers.HypernetworkRadialFlow(dim, cond_dim=1 + dim + aux_dim, nflows=4)

        self.z_mean = nn.Parameter(torch.zeros(1, dim))
        self.z_logstd = nn.Parameter(torch.zeros(1, dim))

    def logprob(self, event_times, spatial_locations, input_mask=None, aux_state=None):
        return self._cond_logliks(event_times, spatial_locations, input_mask, aux_state)

    def _cond_logliks(self, event_times, spatial_locations, input_mask=None, aux_state=None):
        """
        Args:
            event_times: (N, T)
            spatial_locations: (N, T, D)
            input_mask: (N, T) or None
            aux_state: (N, T, D_a)

        Returns:
            A tensor of shape (N, T) containing the conditional log probabilities.
        """

        if input_mask is None:
            input_mask = torch.ones_like(event_times)

        assert event_times.shape == input_mask.shape
        assert event_times.shape[:2] == spatial_locations.shape[:2]
        if aux_state is not None:
            assert event_times.shape[:2] == aux_state.shape[:2]

        N, T, D = spatial_locations.shape
        self.cnf.nfe = 0

        input_mask = input_mask.bool()

        if aux_state is not None:
            aux_state = aux_state

        event_times = self.time_offset + event_times
        event_times = torch.cat([torch.zeros(N, 1).to(event_times), event_times], dim=1)  # (N, 1 + T)

        input_mask = torch.cat([torch.ones(N, 1).to(input_mask), input_mask], dim=1)

        for i in range(T):

            # Mask out the integration if either t0 or t1 has input_mask == 0.
            t0 = event_times[:, -i - 1].mul(input_mask[:, -i - 1]).mul(input_mask[:, -i - 2]).reshape(N, 1).expand(N, i + 1).reshape(-1)
            t1 = event_times[:, -i - 2].mul(input_mask[:, -i - 1]).mul(input_mask[:, -i - 2]).reshape(N, 1).expand(N, i + 1).reshape(-1)

            if i == 0:
                xs = spatial_locations[:, -1].reshape(N, 1, D)
                dlogps = torch.zeros(N, 1).to(xs)
            else:
                xs = torch.cat([
                    spatial_locations[:, -i - 1].reshape(N, 1, D),
                    xs,
                ], dim=1)
                dlogps = torch.cat([
                    torch.zeros(N, 1).to(xs),
                    dlogps,
                ], dim=1)

            xs = xs.reshape(-1, D)
            dlogps = dlogps.reshape(-1)

            norm_fn = None
            if aux_state is not None:
                D_a = aux_state.shape[-1]
                auxs = aux_state[:, -i - 1:, :].expand(N, i + 1, D_a).reshape(-1, D_a)
                inputs = [xs, auxs]
                norm_fn = max_rms_norm([a.shape for a in inputs])
                xs = torch.cat(inputs, dim=1)

            xs, dlogps = self.cnf.integrate(t0, t1, xs, dlogps, method="dopri5" if i < T - 1 and self.training else "dopri5", norm=norm_fn)

            xs, auxs = xs[:, :D], xs[:, D:]

            # Apply instantaneous flow
            if i < T - 1:
                obs_x = spatial_locations[:, -i - 2].reshape(N, 1, D).expand(N, i + 1, D).reshape(-1, D)
                obs_t = event_times[:, -i - 2].reshape(N, 1).expand(N, i + 1).reshape(-1, 1)
                cond = torch.cat([obs_t, obs_x, auxs[:, -self.aux_dim:]], dim=1)  # (N * (i + 1), 1 + D + D_a)
                xs, dlogps = self.inst_flow(xs, logpx=dlogps, cond=cond)

            xs = xs.reshape(N, i + 1, D)
            dlogps = dlogps.reshape(N, i + 1)
            dlogps = torch.where(input_mask[:, -i - 1:], dlogps, torch.zeros_like(dlogps))

        logpz = gaussian_loglik(xs, self.z_mean.expand_as(xs), self.z_logstd.expand_as(xs)).sum(-1)  # (N, T)
        logpx = logpz - dlogps

        return torch.where(input_mask[:, 1:], logpx, torch.zeros_like(logpx))

    def spatial_conditional_logprob_fn(self, t, event_times, spatial_locations, aux_state=None):
        """
        Args:
            t: scalar
            event_times: (T,)
            spatial_locations: (T, D)
            aux_state: (T + 1, D_a)

        Returns a function that takes locations (N, D) and returns (N,) the logprob at time t.
        """
        T, D = spatial_locations.shape

        def loglikelihood_fn(s):
            bsz = s.shape[0]
            bsz_event_times = event_times[None].expand(bsz, T)
            bsz_event_times = torch.cat([bsz_event_times, torch.ones(bsz, 1).to(bsz_event_times) * t], dim=1)
            bsz_spatial_locations = spatial_locations[None].expand(bsz, T, D)
            bsz_spatial_locations = torch.cat([bsz_spatial_locations, s.reshape(bsz, 1, D)], dim=1)

            if aux_state is not None:
                bsz_aux_state = aux_state.reshape(1, T + 1, -1).expand(bsz, -1, -1)
            else:
                bsz_aux_state = None

            return self.logprob(bsz_event_times, bsz_spatial_locations, input_mask=None, aux_state=bsz_aux_state).sum(1)

        return loglikelihood_fn

    def vector_field_fn(self, t, event_times, spatial_locations, aux_state=None):
        """
        Args:
            t: scalar
            event_times: (T,)
            spatial_locations: (T, D)
            aux_state: (T + 1, D_a)

        Returns a function that takes locations (N, D) and returns the (N, D) vector field at time t.
        """

        T, D = spatial_locations.shape

        def vecfield_fn(s):
            bsz = s.shape[0]

            xs = s.reshape(bsz, D)
            if aux_state is not None:
                D_a = aux_state.shape[-1]
                auxs = aux_state[-1, :].reshape(1, D_a).expand(bsz, D_a)
                inputs = [xs, auxs]
                xs = torch.cat(inputs, dim=1)

            dx = self.cnf.func(t, xs)[:, :D]
            return dx

        return vecfield_fn


def gaussian_loglik(z, mean, log_std):
    mean = mean + torch.tensor(0.)
    log_std = log_std + torch.tensor(0.)
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)


def zero_init_linear(d_in, d_out):
    m = nn.Linear(d_in, d_out, bias=False)
    m.weight.data.fill_(0.)
    return m


class ConditionalSequential(nn.Module):

    def __init__(self, input_layers, cond_layers, actfn_layers):
        super().__init__()
        assert len(input_layers) == len(cond_layers) and len(input_layers) - 1 == len(actfn_layers)
        self.input_layers = nn.ModuleList(input_layers)
        self.cond_layers = nn.ModuleList(cond_layers)
        self.actfn_layers = nn.ModuleList(actfn_layers)

    def forward(self, x, cond):
        for il, cl, act in zip(self.input_layers, self.cond_layers, self.actfn_layers):
            x = il(x) + cl(cond)
            x = act(x)
        x = self.input_layers[-1](x) + self.cond_layers[-1](cond)
        return x
