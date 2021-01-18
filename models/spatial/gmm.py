# Copyright (c) Facebook, Inc. and its affiliates.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianMixtureSpatialModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.mu0 = nn.Parameter(torch.tensor(0.0))
        self.logstd0 = nn.Parameter(torch.tensor(0.0))
        self.coeff_decay = nn.Parameter(torch.tensor(0.1))
        self.spatial_logstd = nn.Parameter(torch.tensor(0.1))

    def logprob(self, event_times, spatial_locations, input_mask=None):
        """
        Args:
            event_times: (N, T)
            spatial_locations: (N, T, D)
            input_mask: (N, T)

        Returns:
            logprob: (N,)
        """

        # Assume inputs are right-padded. (i.e. each row of input_mask is [1, 1, ...,1, 0, ..., 0])
        if input_mask is None:
            input_mask = torch.ones_like(event_times)

        # Set distribution of first sample to be standard Normal.
        s0 = spatial_locations[:, 0]
        loglik0 = gaussian_loglik(s0, self.mu0, self.logstd0).sum(-1)  # (N,)

        # Pair-wise time deltas.
        dt = (event_times[:, :, None] - event_times[:, None])  # (N, T, T)

        locs = spatial_locations.unsqueeze(-2)   # (N, T, 1, D)
        means = spatial_locations.unsqueeze(-3)  # (N, 1, T, D)

        pairwise_logliks = gaussian_loglik(locs, means, self.spatial_logstd).sum(-1)  # (N, T, T)
        pairwise_logliks = fill_triu(pairwise_logliks, -1e20)

        dt_logdecay = -dt / F.softplus(self.coeff_decay)
        dt_logdecay = fill_triu(dt_logdecay, -1e20)

        # Normalize time-decay coefficients.
        dt_logdecay = dt_logdecay - torch.logsumexp(dt_logdecay, dim=-1, keepdim=True)  # (N, T, 1)
        loglik = torch.logsumexp(pairwise_logliks + dt_logdecay, dim=-1)  # (N, T)

        return torch.cat([loglik0[..., None], (loglik * input_mask)[:, 1:]], dim=1)  # (N, T)

    def spatial_conditional_logprob_fn(self, t, event_times, spatial_locations):
        """
        Args:
            t: scalar
            event_times: (T,)
            spatial_locations: (T, D)

        Returns a function that takes locations (N, D) and returns (N,) the logprob at time t.
        """

        if spatial_locations is None:
            return lambda s: gaussian_loglik(s, self.mu0[None], self.logstd0[None]).sum(-1)

        dt = t - event_times
        logweights = F.log_softmax(-dt / F.softplus(self.coeff_decay), dim=0)

        def loglikelihood_fn(s):
            loglik = gaussian_loglik(s[:, None], spatial_locations[None], self.spatial_logstd).sum(-1)
            return torch.logsumexp(loglik + logweights[None], dim=1)

        return loglikelihood_fn


def lowtri(A):
    return torch.tril(A, diagonal=-1)


def fill_triu(A, value):
    A = lowtri(A)
    A = A + torch.triu(torch.ones_like(A)) * value
    return A


def gaussian_loglik(z, mean, log_std):
    mean = mean + torch.tensor(0.)
    log_std = log_std + torch.tensor(0.)
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)
