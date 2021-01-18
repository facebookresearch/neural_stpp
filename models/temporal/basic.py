# Copyright (c) Facebook, Inc. and its affiliates.

import abc
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalPointProcess(nn.Module):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def logprob(self, event_times, input_mask, t0, t1):
        """
        Args:
            event_times: (N, T)
            input_mask: (N, T)
            t0: (N,) or (1,)
            t1: (N,) or (1,)
        """
        raise NotImplementedError


class HomogeneousPoissonPointProcess(TemporalPointProcess):

    def __init__(self):
        super().__init__()
        self.lamb = nn.Parameter(torch.randn(1) * 0.2 - 2.0)

    def logprob(self, event_times, spatial_locations, input_mask, t0, t1):
        lamb = F.softplus(self.lamb)
        compensator = (t1 - t0) * lamb
        loglik = input_mask.sum(-1) * torch.log(lamb + 1e-20) - compensator
        return loglik


class HawkesPointProcess(TemporalPointProcess):

    def __init__(self):
        super().__init__()

        self.mu = nn.Parameter(torch.randn(1) * 0.5 - 2.0)
        self.alpha = nn.Parameter(torch.randn(1) * 0.5 - 3.0)
        self.beta = nn.Parameter(torch.randn(1) * 0.5)

    def logprob(self, event_times, spatial_locations, input_mask, t0, t1):
        del spatial_locations

        mu = F.softplus(self.mu)
        alpha = F.softplus(self.alpha)
        beta = F.softplus(self.beta)

        dt = event_times[:, :, None] - event_times[:, None]  # (N, T, T)
        dt = fill_triu(-dt * beta, -1e20)
        lamb = torch.exp(torch.logsumexp(dt, dim=-1)) * alpha + mu  # (N, T)
        loglik = torch.log(lamb + 1e-8).mul(input_mask).sum(-1)  # (N,)

        log_kernel = -beta * (t1[:, None] - event_times) * input_mask + (1.0 - input_mask) * -1e20

        compensator = (t1 - t0) * mu
        compensator = compensator - alpha / beta * (torch.exp(torch.logsumexp(log_kernel, dim=-1)) - input_mask.sum(-1))

        return loglik - compensator


class SelfCorrectingPointProcess(TemporalPointProcess):

    def __init__(self):
        super().__init__()

        self.mu = nn.Parameter(torch.randn(1) * 0.5 - 2.0)
        self.beta = nn.Parameter(torch.randn(1) * 0.5)

    def logprob(self, event_times, spatial_locations, input_mask, t0, t1):
        del spatial_locations

        N, T = event_times.shape

        mu = F.softplus(self.mu)
        beta = F.softplus(self.beta)

        betaN = beta * torch.arange(T).reshape(1, T).expand(N, T).to(beta)  # (N, T)

        loglik = mu * event_times - betaN  # (N, T)
        loglik = loglik.mul(input_mask).sum(-1)  # (N,)

        t0_i = t0.reshape(-1).expand(N)
        N_i = torch.zeros(N).to(event_times)
        compensator = torch.zeros(N).to(event_times)
        for i in range(T):
            t1_i = torch.where(input_mask[:, i].bool(), event_times[:, i], t0_i)
            compensator = compensator + torch.exp(-beta * N_i) / mu * (torch.exp(mu * t1_i) - torch.exp(mu * t0_i))

            t0_i = torch.where(input_mask[:, i].bool(), event_times[:, i], t0_i)
            N_i = N_i + input_mask[:, i]
        compensator = compensator + torch.exp(-beta * N_i) / mu * (torch.exp(mu * t1) - torch.exp(mu * t0_i))

        return loglik - compensator  # (N,)


def lowtri(A):
    return torch.tril(A, diagonal=-1)


def fill_triu(A, value):
    A = lowtri(A)
    A = A + torch.triu(torch.ones_like(A)) * value
    return A
