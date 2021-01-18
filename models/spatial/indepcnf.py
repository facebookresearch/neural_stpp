# Copyright (c) Facebook, Inc. and its affiliates.

import math
import torch
import torch.nn as nn

from .cnf import TimeVariableCNF, build_fc_odefunc


class IndependentCNF(nn.Module):

    time_offset = 2.0

    def __init__(self, dim=2, hidden_dims=[64, 64, 64], layer_type="concat", actfn="softplus",
                 zero_init=True, tol=1e-6, otreg_strength=0.0, squash_time=False):
        super().__init__()
        self.squash_time = squash_time

        func = build_fc_odefunc(dim=dim, hidden_dims=hidden_dims, layer_type=layer_type, actfn=actfn, zero_init=zero_init)
        self.cnf = TimeVariableCNF(func, dim, tol=tol, energy_regularization=otreg_strength, jacnorm_regularization=otreg_strength)

        self.z_mean = nn.Parameter(torch.zeros(1, dim))
        self.z_logstd = nn.Parameter(torch.zeros(1, dim))

    def logprob(self, event_times, spatial_locations, input_mask=None):
        """
        Args:
            event_times: (N, T)
            spatial_locations: (N, T, D)
            input_mask: (N, T)

        Returns:
            logprob: (N,)
        """
        N, T, D = spatial_locations.shape

        if input_mask is None:
            input_mask = torch.ones_like(event_times)

        # Flatten since independent.
        event_times = event_times.reshape(N * T)
        spatial_locations = spatial_locations.reshape(N * T, D)

        if self.squash_time:
            t0 = torch.zeros_like(event_times)
            t1 = torch.zeros_like(event_times) + self.time_offset
        else:
            t0 = event_times + self.time_offset
            t1 = torch.zeros_like(event_times)

        self.cnf.nfe = 0
        z, delta_logp = self.cnf.integrate(t0, t1, spatial_locations, torch.zeros_like(event_times))
        logpz = gaussian_loglik(z, self.z_mean, self.z_logstd).sum(-1)
        logpx = logpz - delta_logp  # (N * T)
        return (logpx.reshape(N, T) * input_mask)  # (N,)

    def spatial_conditional_logprob_fn(self, t, event_times, spatial_locations):
        """
        Args:
            t: scalar
            event_times: (T,)
            spatial_locations: (T, D)

        Returns a function that takes locations (N, D) and returns (N,) the logprob at time t.
        """

        # Does not depend on history.
        del event_times, spatial_locations

        def loglikelihood_fn(s):
            bsz = s.shape[0]
            event_times = torch.ones(bsz, 1).to(s) * t
            return self.logprob(event_times, s.reshape(bsz, 1, -1), input_mask=None).sum(-1)

        return loglikelihood_fn

    def extra_repr(self):
        return f"squash_time={self.squash_time}"


def gaussian_loglik(z, mean, log_std):
    mean = mean + torch.tensor(0.)
    log_std = log_std + torch.tensor(0.)
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)
