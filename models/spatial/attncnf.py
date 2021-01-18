# Copyright (c) Facebook, Inc. and its affiliates.

import math
import torch
import torch.nn as nn

from .attention import EventTimeEncoding, MultiheadAttention, L2MultiheadAttention
from .cnf import TimeVariableCNF, build_fc_odefunc, max_rms_norm


class SelfAttentiveODEFunc(nn.Module):

    def __init__(self, dim, hidden_dims, aux_dim, actfn, time_offset, nblocks=2, l2_attn=False, layer_type="concat"):
        super().__init__()
        self.dim = dim
        self.aux_dim = aux_dim
        self.time_offset = time_offset

        mid_idx = int(math.ceil(len(hidden_dims) / 2))

        self.embed_dim = hidden_dims[mid_idx]
        self.embedding = build_fc_odefunc(
            self.dim + self.aux_dim, hidden_dims[:mid_idx],
            out_dim=self.embed_dim, layer_type=layer_type, actfn=actfn, zero_init=False)

        if l2_attn:
            mha = L2MultiheadAttention
        else:
            mha = MultiheadAttention

        self.self_attns = nn.ModuleList([mha(self.embed_dim, num_heads=4) for _ in range(nblocks)])
        self.attn_actnorms = nn.ModuleList([ActNorm(self.embed_dim) for _ in range(nblocks)])
        self.fcs = nn.ModuleList([
            nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim * 4), nn.Softplus(), nn.Linear(self.embed_dim * 4, self.embed_dim))
            for _ in range(nblocks)
        ])
        self.fc_actnorms = nn.ModuleList([ActNorm(self.embed_dim) for _ in range(nblocks)])
        self.attn_gates = nn.ModuleList([TanhGate() for _ in range(nblocks)])
        self.fc_gates = nn.ModuleList(TanhGate() for _ in range(nblocks))

        self.output_proj = build_fc_odefunc(self.embed_dim, hidden_dims[mid_idx:], out_dim=self.dim, layer_type=layer_type, actfn=actfn, zero_init=True)

    def set_shape(self, shape):
        self.shape = shape
        # self.attn_weights = []

    def _create_self_attn_mask(self, T):
        return torch.triu(torch.ones(T, T), diagonal=1) * -1e12

    def _update_attn_weights(self, attn_weights):
        self.attn_weights.append(attn_weights.detach().cpu())

    def forward(self, t, state, rm_nonself_grads=False):
        T, N, _ = self.shape
        x, a = state[:, :self.dim], state[:, max(self.dim + 1, state.shape[-1] - self.aux_dim):]
        x = torch.cat([x, a], dim=-1)
        x = self.embedding(t, x)
        x = x.reshape(T, N, self.embed_dim)

        attn_mask = self._create_self_attn_mask(T).to(x)
        for norm0, self_attn, gate0, norm1, fc, gate1 in zip(self.attn_actnorms, self.self_attns, self.attn_gates, self.fc_actnorms, self.fcs, self.fc_gates):
            h, attn_weights = self_attn(norm0(x), attn_mask=attn_mask, rm_nonself_grads=rm_nonself_grads)
            # self._update_attn_weights(attn_weights)
            x = x + gate0(h)
            x = x + gate1(fc(norm1(x)))

        dx = self.output_proj(t, x.reshape(-1, self.embed_dim))
        dh = torch.zeros_like(state[:, self.dim:])
        return torch.cat([dx, dh], dim=1)


class SelfAttentiveCNF(nn.Module):

    time_offset = 2.0

    def __init__(self, dim=2, hidden_dims=[64, 64, 64], aux_dim=0, layer_type="concat", actfn="softplus", zero_init=True, l2_attn=False, tol=1e-4, otreg_strength=0.0, lowvar_trace=True):
        super().__init__()

        self.dim = dim
        self.aux_dim = aux_dim

        mid_idx = int(math.ceil(len(hidden_dims) / 2))
        self.t_embedding_dim = hidden_dims[mid_idx]
        self.t_embedding = EventTimeEncoding(self.t_embedding_dim)

        self.odefunc = SelfAttentiveODEFunc(dim, hidden_dims, aux_dim + self.t_embedding_dim, actfn, self.time_offset, l2_attn=l2_attn, layer_type=layer_type)
        self.cnf = TimeVariableCNF(self.odefunc, dim, tol=tol, method="dopri5",
                                   nonself_connections=lowvar_trace, energy_regularization=otreg_strength, jacnorm_regularization=otreg_strength)

        base_odefunc = build_fc_odefunc(dim=dim, hidden_dims=hidden_dims, layer_type=layer_type, actfn=actfn, zero_init=zero_init)
        self.base_cnf = TimeVariableCNF(base_odefunc, dim, tol=1e-6, method="dopri5", energy_regularization=1e-4, jacnorm_regularization=1e-4)

        self.base_dist_params = nn.Sequential(
            nn.Linear(aux_dim + self.t_embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, dim * 2),
        )
        self.base_dist_params[-1].weight.data.fill_(0)
        self.base_dist_params[-1].bias.data.fill_(0)

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

        if aux_state is not None:
            aux_state = aux_state

        N, T, D = spatial_locations.shape
        spatial_locations = spatial_locations.clone().requires_grad_(True)

        t_embed = self.t_embedding(event_times) / math.sqrt(self.t_embedding_dim)

        if aux_state is not None:
            inputs = [spatial_locations, aux_state, t_embed]
        else:
            inputs = [spatial_locations, t_embed]

        # attention layer uses (T, N, D) ordering.
        inputs = [inp.transpose(0, 1) for inp in inputs]
        norm_fn = max_rms_norm([a.shape for a in inputs])
        x = torch.cat(inputs, dim=-1)

        self.odefunc.set_shape(x.shape)

        x = x.reshape(T * N, -1)
        event_times = event_times.transpose(0, 1).reshape(T * N)

        t0 = event_times + self.time_offset
        t1 = torch.zeros_like(event_times) + self.time_offset

        z, delta_logp = self.cnf.integrate(t0, t1, x, torch.zeros_like(event_times), norm=norm_fn)
        z = z[:, :self.dim]  # (T * N, D)

        base_t = torch.zeros_like(event_times)
        z, delta_logp = self.base_cnf.integrate(t1, base_t, z, delta_logp)

        if aux_state is not None:
            cond_inputs = [aux_state[:, :, -self.aux_dim:], t_embed]
        else:
            cond_inputs = [t_embed]
        cond = torch.cat(cond_inputs, dim=-1)  # (N, T, -1)
        cond = torch.where(input_mask[..., None].expand_as(cond).bool(), cond, torch.zeros_like(cond))
        cond = cond.transpose(0, 1).reshape(T * N, -1)

        z_params = self.base_dist_params(cond)

        z_mean, z_logstd = torch.split(z_params, D, dim=-1)

        logpz = gaussian_loglik(z, z_mean, z_logstd).sum(-1)
        logpx = logpz - delta_logp  # (T * N)

        logpx = logpx.reshape(T, N).transpose(0, 1)  # (N, T)

        return torch.where(input_mask.bool(), logpx, torch.zeros_like(logpx))

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


def gaussian_loglik(z, mean, log_std):
    mean = mean + torch.tensor(0.)
    log_std = log_std + torch.tensor(0.)
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)


def gaussian_sample(mean, log_std):
    mean = mean + torch.tensor(0.)
    log_std = log_std + torch.tensor(0.)
    z = torch.randn_like(mean) * torch.exp(log_std) + mean
    return z


class TanhGate(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.tanh(self.weight) * x


class ActNorm(nn.Module):

    def __init__(self, num_features):
        super(ActNorm, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.register_buffer('initialized', torch.tensor(0))

    def forward(self, x, logpx=None):
        if not self.initialized:
            with torch.no_grad():
                # compute batch statistics
                x_ = x.reshape(-1, x.shape[-1])
                batch_mean = torch.mean(x_, dim=0)
                batch_var = torch.var(x_, dim=0)

                # for numerical issues
                batch_var = torch.max(batch_var, torch.tensor(0.2).to(batch_var))

                self.bias.data.copy_(-batch_mean)
                self.weight.data.copy_(-0.5 * torch.log(batch_var))
                self.initialized.fill_(1)

        bias = self.bias.expand_as(x)
        weight = self.weight.expand_as(x)

        y = (x + bias) * torch.exp(weight)

        if logpx is None:
            return y
        else:
            return y, logpx - self._logdetgrad(x)

    def inverse(self, y, logpy=None):
        assert self.initialized
        bias = self.bias.expand_as(y)
        weight = self.weight.expand_as(y)

        x = y * torch.exp(-weight) - bias

        if logpy is None:
            return x
        else:
            return x, logpy + self._logdetgrad(x)

    def _logdetgrad(self, x):
        return self.weight.view(*self.shape).expand(*x.size()).contiguous().view(x.size(0), -1).sum(1, keepdim=True)

    def __repr__(self):
        return ('{name}({num_features})'.format(name=self.__class__.__name__, **self.__dict__))
