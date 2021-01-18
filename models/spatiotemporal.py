# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from models.spatial import JumpCNF, SelfAttentiveCNF, ConditionalGMM
from models.temporal import NeuralPointProcess


class SpatiotemporalModel(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def forward(self, event_times, spatial_locations, input_mask, t0, t1):
        """
        Args:
            event_times: (N, T)
            spatial_locations: (N, T, D)
            input_mask: (N, T)
            t0: () or (N,)
            t1: () or (N,)
        """
        pass

    @abstractmethod
    def spatial_conditional_logprob_fn(self, t, event_times, spatial_locations, t0, t1):
        pass


class CombinedSpatiotemporalModel(SpatiotemporalModel):

    def __init__(self, spatial_model, temporal_model):
        super().__init__()
        self.spatial_model = spatial_model
        self.temporal_model = temporal_model

    def forward(self, event_times, spatial_locations, input_mask, t0, t1):
        space_loglik = self._spatial_logprob(event_times, spatial_locations, input_mask)
        time_loglik = self._temporal_logprob(event_times, spatial_locations, input_mask, t0, t1)
        return space_loglik, time_loglik

    def spatial_conditional_logprob_fn(self, t, event_times, spatial_locations, t0, t1):
        return self.spatial_model.spatial_conditional_logprob_fn(t, event_times, spatial_locations)

    def _spatial_logprob(self, event_times, spatial_locations, input_mask):
        return self.spatial_model.logprob(event_times, spatial_locations, input_mask)

    def _temporal_logprob(self, event_times, spatial_locations, input_mask, t0, t1):
        return self.temporal_model.logprob(event_times, spatial_locations, input_mask, t0, t1)


class SharedHiddenStateSpatiotemporalModel(SpatiotemporalModel, metaclass=ABCMeta):

    def __init__(self, dim=2, hidden_dims=[64, 64, 64], tpp_hidden_dims=[8, 20], tpp_cond=False, tpp_style="split",
                 actfn="softplus", tpp_actfn="softplus", zero_init=True, share_hidden=False, solve_reverse=False, tpp_otreg_strength=0.0, tol=1e-6, **kwargs):
        super().__init__()
        tpp_hidden_dims = [h for h in tpp_hidden_dims]
        self.temporal_model = NeuralPointProcess(
            cond_dim=dim, hidden_dims=tpp_hidden_dims, cond=tpp_cond, style=tpp_style, actfn=tpp_actfn, hdim=tpp_hidden_dims[0] // 2,
            separate=2 if not share_hidden else 1, tol=tol, otreg_strength=tpp_otreg_strength)
        self._build_spatial_model(dim, hidden_dims, actfn, zero_init, aux_dim=tpp_hidden_dims[0] // 2,
                                  aux_odefunc=self.temporal_model.hidden_state_dynamics if solve_reverse else zero_diffeq,
                                  tol=tol, **kwargs)

    @abstractmethod
    def _build_spatial_model(self, dim, hidden_dims, actfn, zero_init, aux_dim, aux_odefunc, **kwargs):
        pass

    def forward(self, event_times, spatial_locations, input_mask, t0, t1):
        intensities, Lambda, hidden_states = self.temporal_model.integrate_lambda(event_times, spatial_locations, input_mask, t0, t1)
        time_loglik = torch.sum(torch.log(intensities + 1e-8) * input_mask, dim=1) - Lambda
        hidden_states = hidden_states[:, 1:-1]  # Remove first (t=t0) and last (t=t1) hidden states.
        space_loglik = self.spatial_model.logprob(event_times, spatial_locations, input_mask, aux_state=hidden_states)
        return space_loglik, time_loglik

    def spatial_conditional_logprob_fn(self, t, event_times, spatial_locations, t0, t1):
        hidden_state_times = torch.cat([event_times, torch.tensor(t).reshape(-1).to(event_times)]).reshape(1, -1)
        _, _, hidden_states = self.temporal_model.integrate_lambda(hidden_state_times, spatial_locations[None], input_mask=None, t0=t0, t1=None)
        hidden_states = hidden_states[:, 1:]  # Remove first (t=t0) hidden state.
        return self.spatial_model.spatial_conditional_logprob_fn(t, event_times, spatial_locations, aux_state=hidden_states)

    def vector_field_fn(self, t, event_times, spatial_locations, t0, t1):
        hidden_state_times = torch.cat([event_times, torch.tensor(t).reshape(-1).to(event_times)]).reshape(1, -1)
        _, _, hidden_states = self.temporal_model.integrate_lambda(hidden_state_times, spatial_locations[None], input_mask=None, t0=t0, t1=None)
        hidden_states = hidden_states[0, 1:]  # Remove first (t=t0) hidden state.
        return self.spatial_model.vector_field_fn(t, event_times, spatial_locations, aux_state=hidden_states)

    def sample_spatial(self, nsamples, event_times, spatial_locations, input_mask, t0, t1):
        intensities, Lambda, hidden_states = self.temporal_model.integrate_lambda(event_times, spatial_locations, input_mask, t0, t1)
        hidden_states = hidden_states[:, 1:-1]  # Remove first (t=t0) and last (t=t1) hidden states.
        samples = self.spatial_model.sample_spatial(nsamples, event_times, spatial_locations, input_mask, aux_state=hidden_states)
        return samples


class JumpCNFSpatiotemporalModel(SharedHiddenStateSpatiotemporalModel):

    def _build_spatial_model(self, dim, hidden_dims, actfn, zero_init, aux_dim, aux_odefunc, **kwargs):
        self.spatial_model = JumpCNF(
            dim=dim, hidden_dims=hidden_dims, actfn=actfn, zero_init=zero_init, aux_dim=aux_dim, aux_odefunc=aux_odefunc, **kwargs,
        )


class SelfAttentiveCNFSpatiotemporalModel(SharedHiddenStateSpatiotemporalModel):

    def _build_spatial_model(self, dim, hidden_dims, actfn, zero_init, aux_dim, aux_odefunc, **kwargs):
        self.spatial_model = SelfAttentiveCNF(dim=dim, hidden_dims=hidden_dims, actfn=actfn, zero_init=zero_init, aux_dim=aux_dim, **kwargs)


class JumpGMMSpatiotemporalModel(SharedHiddenStateSpatiotemporalModel):

    def _build_spatial_model(self, dim, hidden_dims, actfn, zero_init, aux_dim, aux_odefunc, n_mixtures=5, **kwargs):
        self.spatial_model = ConditionalGMM(dim=dim, hidden_dims=hidden_dims, actfn=actfn, aux_dim=aux_dim, n_mixtures=n_mixtures)


def zero_diffeq(t, h):
    return torch.zeros_like(h)
