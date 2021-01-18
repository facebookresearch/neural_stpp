# Copyright (c) Facebook, Inc. and its affiliates.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

import diffeq_layers
from .basic import TemporalPointProcess


class IntensityODEFunc(nn.Module):

    def __init__(self, hdim, dstate_fn, intensity_fn):
        super().__init__()
        self.hdim = hdim
        self.dstate_fn = dstate_fn
        self.intensity_fn = intensity_fn

    def forward(self, t, state):
        Lambda, tpp_state = state
        intensity = self.get_intensity(tpp_state).reshape(-1)
        return intensity, self.dstate_fn(t, tpp_state)

    def get_intensity(self, tpp_state):
        return torch.sigmoid(self.intensity_fn(tpp_state[..., :self.hdim]) - 2.0) * 50


class SplitHiddenStateODEFunc(nn.Module):

    def __init__(self, dstate_net, update_net):
        super().__init__()
        self.dstate_net = dstate_net
        self.update_net = update_net

    def forward(self, t, tpp_state):
        dstate = self.dstate_net(t, tpp_state)
        c, h = torch.split(tpp_state, tpp_state.shape[1] // 2, dim=1)
        dcdt, dhdt = torch.split(dstate, tpp_state.shape[1] // 2, dim=1)
        dcdt = dcdt - (dcdt * c).sum(dim=-1, keepdim=True) / (c * c).sum(dim=-1, keepdim=True) * c
        dhdt = -F.softplus(dhdt) * h
        return torch.cat([dcdt, dhdt], dim=1)

    def update_state(self, t, tpp_state, cond=None):
        if cond is not None:
            inputs = torch.cat([tpp_state, cond], dim=1)
        else:
            inputs = tpp_state

        upd_c, upd_h = torch.split(self.update_net(t, inputs), tpp_state.shape[1] // 2, dim=1)
        update = torch.cat([torch.zeros_like(upd_c), upd_h], dim=1)
        return tpp_state + update


class SimpleHiddenStateODEFunc(nn.Module):

    def __init__(self, dstate_net, update_net):
        super().__init__()
        self.dstate_net = dstate_net
        self.update_net = update_net

    def forward(self, t, tpp_state):
        return torch.tanh(self.dstate_net(t, tpp_state))

    def update_state(self, t, tpp_state, cond=None):
        if cond is not None:
            inputs = torch.cat([tpp_state, cond], dim=1)
        else:
            inputs = tpp_state
        return self.update_net(t, inputs)


class GRUHiddenStateODEFunc(nn.Module):

    def __init__(self, dstate_net, update_net):
        super().__init__()
        self.dstate_net = dstate_net
        self.update_net = update_net

    def forward(self, t, tpp_state):
        return self.dstate_net(t, tpp_state)

    def update_state(self, t, tpp_state, cond=None):
        if cond is None:
            bsz = tpp_state.shape[0]
            cond = torch.zeros(bsz, 0).to(tpp_state)

        return self.update_net(cond, tpp_state)


class HiddenStateODEFuncList(nn.Module):

    def __init__(self, *odefuncs):
        super().__init__()
        self.odefuncs = nn.ModuleList(odefuncs)

    def forward(self, t, tpp_state):
        states = torch.split(tpp_state, tpp_state.shape[-1] // len(self.odefuncs), dim=-1)
        ds = []
        for s, func in zip(states, self.odefuncs):
            ds.append(func(t, s))
        return torch.cat(ds, dim=-1)

    def update_state(self, t, tpp_state, cond=None):
        states = torch.split(tpp_state, tpp_state.shape[-1] // len(self.odefuncs), dim=-1)
        upds = []
        for s, func in zip(states, self.odefuncs):
            upds.append(func.update_state(t, s, cond))
        return torch.cat(upds, dim=-1)


class NeuralPointProcess(TemporalPointProcess):

    dynamics_dict = {
        "split": SplitHiddenStateODEFunc,
        "simple": SimpleHiddenStateODEFunc,
        "gru": GRUHiddenStateODEFunc,
    }

    def __init__(self, cond_dim=0, hidden_dims=[64, 64, 64], cond=False, style="split", actfn="softplus", hdim=None, separate=1, tol=1e-6, otreg_strength=0.1):
        super().__init__()
        if not cond:
            cond_dim = 0
        self.cond = cond
        self.cond_dim = cond_dim
        self.hdim = hidden_dims[0] if hdim is None else hdim
        assert self.hdim % 2 == 0
        self._init_state = nn.Parameter(torch.randn(hidden_dims[0]) / math.sqrt(hidden_dims[0]))

        dynamics = []
        for i in range(separate):
            dstate_net = construct_diffeqnet(hidden_dims[0] // separate, hidden_dims[1:], hidden_dims[0] // separate, time_dependent=False, actfn=actfn, zero_init=True)
            if style in ["split", "simple"]:
                update_net = construct_diffeqnet(hidden_dims[0] // separate + cond_dim, hidden_dims[1:], hidden_dims[0] // separate, time_dependent=False, actfn="celu", gated=True, zero_init=False)
            elif style in ["gru"]:
                update_net = nn.GRUCell(cond_dim, hidden_dims[0] // separate)
            dynamics.append(self.dynamics_dict[style](dstate_net, update_net))

        self.hidden_state_dynamics = HiddenStateODEFuncList(*dynamics)

        intensity_net = nn.Sequential(
            nn.Linear(self.hdim, self.hdim * 4),
            nn.Softplus(),
            nn.Linear(self.hdim * 4, 1),
        )
        intensity_odefunc = IntensityODEFunc(self.hdim, self.hidden_state_dynamics, intensity_net)
        self.ode_solver = TimeVariableODE(intensity_odefunc, atol=tol, rtol=tol, energy_regularization=otreg_strength)

    def logprob(self, event_times, spatial_locations, input_mask, t0, t1):
        intensities, Lambda, _ = self.integrate_lambda(event_times, spatial_locations, input_mask, t0, t1)
        log_intensities = torch.log(intensities + 1e-8)
        log_intensities = torch.where(input_mask.bool(), log_intensities, torch.zeros_like(log_intensities))
        logprob = torch.sum(log_intensities, dim=1) - Lambda
        return logprob

    def get_intensity(self, state):
        return self.ode_solver.func.get_intensity(state)

    def integrate_lambda(self, event_times, spatial_location, input_mask, t0, t1, nlinspace=1):
        """
        Args:
            event_times: (N, T)
            spatial_location: (N, T, D)
            input_mask: (N, T)
            t0: (N,) or (1,)
            t1: (N,) or (1,)
        """
        N, T = event_times.shape

        if not self.cond:
            # disable dependence on spatial sample.
            spatial_location = None

        if input_mask is None:
            input_mask = torch.ones_like(event_times)

        input_mask = input_mask.bool()

        init_state = self._init_state[None].expand(N, -1)
        state = (
            torch.zeros(N).to(init_state),  # Lambda(t_0)
            init_state,
        )

        t0 = t0 if torch.is_tensor(t0) else torch.tensor(t0)
        t0 = t0.expand(N).to(event_times)

        self.ode_solver.nfe = 0
        intensities = []
        prejump_hidden_states = []
        for i in range(T):

            # Set t1 = t0 if the input is masked out at time t1.
            t1_i = torch.where(input_mask[:, i], event_times[:, i], t0)
            state_traj = self.ode_solver.integrate(t0, t1_i, state, nlinspace=nlinspace, method="dopri5" if self.training else "dopri5")

            hiddens = state_traj[1]  # (1 + nlinspace, N, D)
            if i > 0:
                hiddens = hiddens[1:]
            # set hidden states to zero if input is masked out at the next time step.
            hiddens = torch.where(input_mask[:, i].reshape(1, -1, 1).expand_as(hiddens), hiddens, torch.zeros_like(hiddens))
            prejump_hidden_states.append(hiddens)

            state = tuple(s[-1] for s in state_traj)
            Lambda, tpp_state = state
            intensities.append(self.get_intensity(tpp_state).reshape(-1))

            if i < T - 1 or t1 is not None:
                cond = spatial_location[:, i] if spatial_location is not None else None
                updated_tpp_state = self.hidden_state_dynamics.update_state(event_times[:, i], tpp_state, cond=cond)
                tpp_state = torch.where(input_mask[:, i].reshape(-1, 1).expand_as(tpp_state), updated_tpp_state, tpp_state)
                state = (Lambda, tpp_state)

            # Track t0 as the last valid event time.
            t0 = torch.where(input_mask[:, i], event_times[:, i], t0)

        if t1 is not None:
            # Integrate from last time sample to t1.
            t1 = t1 if torch.is_tensor(t1) else torch.tensor(t1)
            t1 = t1.expand(N).to(event_times)
            state_traj = self.ode_solver.integrate(t0, t1, state, nlinspace=nlinspace, method="dopri5" if self.training else "dopri5")

            hiddens = state_traj[1][1:]
            prejump_hidden_states.append(hiddens)

            state = tuple(s[-1] for s in state_traj)

        Lambda, _ = state  # (N,)
        intensities = torch.stack(intensities, dim=1)  # (N, T)
        prejump_hidden_states = torch.cat(prejump_hidden_states, dim=0).transpose(0, 1)  # (N, T * nlinspace, D)
        return intensities, Lambda, prejump_hidden_states


ACTFNS = {
    "softplus": lambda dim: diffeq_layers.diffeq_wrapper(nn.Softplus()),
    "swish": lambda dim: diffeq_layers.diffeq_wrapper(Swish(dim)),
    "celu": lambda dim: diffeq_layers.diffeq_wrapper(nn.CELU()),
    "relu": lambda dim: diffeq_layers.diffeq_wrapper(nn.ReLU(inplace=True))
}


def construct_diffeqnet(input_dim, hidden_dims, output_dim, time_dependent=False, actfn="softplus", zero_init=False, gated=False):

    linear_fn = diffeq_layers.IgnoreLinear if time_dependent else diffeq_layers.ConcatLinear_v2

    if gated:
        linear_fn = GatedLinear

    layers = []
    if len(hidden_dims) > 0:
        dims = [input_dim] + list(hidden_dims)
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(linear_fn(d_in, d_out))
            layers.append(ActNorm(d_out))
            if not gated:
                layers.append(ACTFNS[actfn](d_out))
        layers.append(linear_fn(hidden_dims[-1], output_dim))
    else:
        layers.append(linear_fn(input_dim, output_dim))

    # Initialize to zero.
    if zero_init:
        for m in layers[-1].modules():
            if isinstance(m, nn.Linear):
                m.weight.data.fill_(0)
                if m.bias is not None:
                    m.bias.data.fill_(0)
    diffeqnet = diffeq_layers.SequentialDiffEq(*layers)

    return diffeqnet


class Sine(nn.Module):

    def forward(self, x):
        return torch.sin(x)


class Swish(nn.Module):

    def __init__(self, dim=1):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5] * dim))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta)))

    def extra_repr(self):
        return f'{self.beta.nelement()}'


class GatedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.gate = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x) * torch.sigmoid(self.gate(x))


class TimeVariableODE(nn.Module):

    start_time = 0.0
    end_time = 1.0

    def __init__(self, func, atol=1e-6, rtol=1e-6, method="dopri5", energy_regularization=0.01):
        super().__init__()
        self.func = func
        self.atol = atol
        self.rtol = rtol
        self.method = method
        self.energy_regularization = energy_regularization
        self.nfe = 0

    def integrate(self, t0, t1, x0, nlinspace=1, method=None):
        assert nlinspace > 0
        method = method or self.method

        solution = odeint(
            self,
            (t0, t1, torch.zeros(1).to(x0[0]), *x0),
            torch.linspace(self.start_time, self.end_time, nlinspace + 1).to(t0),
            rtol=self.rtol,
            atol=self.atol,
            method=method,
        )
        _, _, energy, *xs = solution
        reg = energy * self.energy_regularization
        return WrapRegularization.apply(reg, *xs)

    def forward(self, s, state):
        """Solves the same dynamics but uses a dummy variable that always integrates [0, 1]."""
        self.nfe += 1
        t0, t1, _, *x = state

        ratio = (t1 - t0) / (self.end_time - self.start_time)
        t = (s - self.start_time) * ratio + t0

        with torch.enable_grad():
            x = tuple(x_.requires_grad_(True) for x_ in x)
            dx = self.func(t, x)
            dx = tuple(dx_ * ratio.reshape(-1, *([1] * (dx_.ndim - 1))) for dx_ in dx)

            d_energy = sum(torch.sum(dx_ * dx_) for dx_ in dx) / sum(x_.numel() for x_ in x)

        if not self.training:
            dx = tuple(dx_.detach() for dx_ in dx)

        return tuple([torch.zeros_like(t0), torch.zeros_like(t1), d_energy, *dx])

    def extra_repr(self):
        return f"method={self.method}, atol={self.atol}, rtol={self.rtol}, energy={self.energy_regularization}"


class WrapRegularization(torch.autograd.Function):

    @staticmethod
    def forward(ctx, reg, *x):
        ctx.save_for_backward(reg)
        return x

    @staticmethod
    def backward(ctx, *grad_x):
        reg, = ctx.saved_variables
        return (torch.ones_like(reg), *grad_x)


def rms_norm(tensor):
    return tensor.pow(2).mean().sqrt()


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


class ActNorm(nn.Module):

    def __init__(self, num_features, init_scale=1.0):
        super(ActNorm, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.init_scale = init_scale
        self.register_buffer('initialized', torch.tensor(0))

    def forward(self, x):
        if not self.initialized:
            with torch.no_grad():
                # compute batch statistics
                x_ = x.reshape(-1, x.shape[-1])
                batch_mean = torch.mean(x_, dim=0)
                batch_var = torch.var(x_, dim=0)

                # for numerical issues
                batch_var = torch.max(batch_var, torch.tensor(0.2).to(batch_var))

                self.bias.data.copy_(-batch_mean)
                self.weight.data.copy_(-0.5 * torch.log(batch_var) + math.log(self.init_scale))
                self.initialized.fill_(1)

        bias = self.bias.expand_as(x)
        weight = self.weight.expand_as(x)

        # y = (x + bias) * torch.exp(weight)
        y = (x + bias) * F.softplus(weight)

        return y

    def __repr__(self):
        return ('{name}({num_features})'.format(name=self.__class__.__name__, **self.__dict__))
