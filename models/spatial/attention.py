# Copyright (c) Facebook, Inc. and its affiliates.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def update_attn_weights(attn_weights, attn_multiplier):
    if attn_multiplier is not None:
        attn_weights = attn_weights * attn_multiplier[..., None]
        attn_weights = attn_weights / attn_weights.sum(1, keepdim=True)
    return attn_weights


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attn_mask=None, rm_nonself_grads=False, attn_multiplier=None):
        """
        Args:
            x: (T, N, D)
            attn_mask: (T, T) added to pre-softmax logits.
        """

        T, N, _ = x.shape

        q, k, v = map(lambda a: a.reshape(T, N, self.num_heads, self.head_dim), torch.split(self.in_proj(x), self.embed_dim, dim=-1))
        attn_logits = torch.einsum('tbhd,sbhd->tsbh', q, k) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_mask = attn_mask[..., None, None]
            attn_logits += attn_mask
        attn_weights = F.softmax(attn_logits, dim=1)  # (T, S, N, H)
        attn_weights = update_attn_weights(attn_weights, attn_multiplier)

        attn = torch.einsum("tsbh,sbhd->tbhd", attn_weights, v).reshape(T, N, -1)

        if rm_nonself_grads:
            # Construct self-only gradient paths wrt keys and queries.
            attn_logits_keyonly = torch.einsum('tbhd,sbhd->tsbh', q.detach(), k) / math.sqrt(self.head_dim)
            attn_logits_queryonly = torch.einsum('tbhd,sbhd->tsbh', q, k.detach()) / math.sqrt(self.head_dim)

            attn_logits_keyonly = SelfonlyGradients.apply(attn_logits_keyonly)
            attn_logits = attn_logits_queryonly + (attn_logits_keyonly - attn_logits_keyonly.detach())
            if attn_mask is not None:
                attn_logits += attn_mask
            attn_weights = F.softmax(attn_logits, dim=1)
            attn_weights = update_attn_weights(attn_weights, attn_multiplier)

            # Zero out the nonself weights.
            selfonly_mask = ~(torch.triu(torch.ones(T, T), diagonal=1) + torch.tril(torch.ones(T, T), diagonal=-1)).bool()
            selfonly_attn_weights = attn_weights * selfonly_mask[..., None, None].to(attn_weights.device)
            # Self-only gradient path wrt values.
            attn_vpath = torch.einsum("tsbh,sbhd->tbhd", selfonly_attn_weights.detach(), v).reshape(T, N, -1)
            attn_spath = torch.einsum("tsbh,sbhd->tbhd", attn_weights, v.detach()).reshape(T, N, -1)

            modified_attn = attn_spath + (attn_vpath - attn_vpath.detach())

            attn = attn.detach() + (modified_attn - modified_attn.detach())

        attn = self.out_proj(attn)
        return attn, attn_weights.detach()


class L2MultiheadAttention(nn.Module):
    """ Kim et al. "The Lipschitz Constant of Self-Attention" https://arxiv.org/abs/2006.04710 """

    def __init__(self, embed_dim, num_heads):
        super(L2MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_weight = nn.Parameter(torch.empty(embed_dim, num_heads, self.head_dim))
        self.v_weight = nn.Parameter(torch.empty(embed_dim, num_heads, self.head_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_weight.view(self.embed_dim, self.embed_dim))
        nn.init.xavier_uniform_(self.v_weight.view(self.embed_dim, self.embed_dim))

    def forward(self, x, attn_mask=None, rm_nonself_grads=False, attn_multiplier=None):
        """
        Args:
            x: (T, N, D)
            attn_mask: (T, T) added to pre-softmax logits.
        """

        T, N, _ = x.shape

        q = k = torch.einsum("tbm,mhd->tbhd", x, self.q_weight)
        squared_dist = (torch.einsum('tbhd,tbhd->tbh', q, q).unsqueeze(1)
                        + torch.einsum('sbhd,sbhd->sbh', k, k).unsqueeze(0)
                        - 2 * torch.einsum('tbhd,sbhd->tsbh', q, k))
        attn_logits = -squared_dist / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_mask = attn_mask[..., None, None]
            attn_logits += attn_mask
        attn_weights = F.softmax(attn_logits, dim=1)  # (T, S, N, H)
        attn_weights = update_attn_weights(attn_weights, attn_multiplier)
        A = torch.einsum("mhd,nhd->hmn", self.q_weight, self.q_weight) / math.sqrt(self.head_dim)
        XA = torch.einsum("tbm,hmn->tbhn", x, A)
        PXA = torch.einsum("tsbh,sbhm->tbhm", attn_weights, XA)

        if rm_nonself_grads:
            # Construct self-only gradient paths wrt keys and queries.
            q_detach = q.detach()
            k_detach = k.detach()
            attn_logits_keyonly = -(torch.einsum('tbhd,tbhd->tbh', q_detach, q_detach).unsqueeze(1)
                                    + torch.einsum('sbhd,sbhd->sbh', k, k).unsqueeze(0)
                                    - 2 * torch.einsum('tbhd,sbhd->tsbh', q_detach, k)) / math.sqrt(self.head_dim)
            attn_logits_queryonly = -(torch.einsum('tbhd,tbhd->tbh', q, q).unsqueeze(1)
                                      + torch.einsum('sbhd,sbhd->sbh', k_detach, k_detach).unsqueeze(0)
                                      - 2 * torch.einsum('tbhd,sbhd->tsbh', q, k_detach)) / math.sqrt(self.head_dim)

            attn_logits_keyonly = SelfonlyGradients.apply(attn_logits_keyonly)
            attn_logits = attn_logits_queryonly + (attn_logits_keyonly - attn_logits_keyonly.detach())
            if attn_mask is not None:
                attn_logits += attn_mask
            attn_weights = F.softmax(attn_logits, dim=1)
            attn_weights = update_attn_weights(attn_weights, attn_multiplier)

            # Zero out the nonself weights.
            selfonly_mask = ~(torch.triu(torch.ones(T, T), diagonal=1) + torch.tril(torch.ones(T, T), diagonal=-1)).bool()
            selfonly_attn_weights = attn_weights * selfonly_mask[..., None, None].to(attn_weights.device)
            # Self-only gradient path wrt values.
            PXA_vpath = torch.einsum("tsbh,sbhm->tbhm", selfonly_attn_weights.detach(), XA)
            PXA_spath = torch.einsum("tsbh,sbhm->tbhm", attn_weights, XA.detach())

            modified_PXA = PXA_spath + (PXA_vpath - PXA_vpath.detach())
            PXA = PXA.detach() + (modified_PXA - modified_PXA.detach())

        PXAV = torch.einsum("tbhm,mhd->tbhd", PXA, self.v_weight).reshape(T, N, self.embed_dim)
        return self.out_proj(PXAV), attn_weights.detach()


class SelfonlyGradients(torch.autograd.Function):

    @staticmethod
    def forward(ctx, attn_logits):
        return attn_logits

    @staticmethod
    def backward(ctx, grads):
        # (T, T, N, H) -> (N, H, T)
        grads = torch.diagonal(grads, dim1=0, dim2=1)
        # (N, H, T, T) -> (T, T, N, H)
        grads = torch.diag_embed(grads).permute(2, 3, 0, 1)
        return grads


class EventTimeEncoding(nn.Module):

    def __init__(self, dim):
        super(EventTimeEncoding, self).__init__()
        self.dim = dim
        div_term = torch.exp(torch.arange(0, self.dim, 2).float() * (-math.log(10000.0) / self.dim))
        self.register_buffer("div_term", div_term)

    def forward(self, event_times):
        N, T = event_times.shape
        pe = torch.zeros(N, T, self.dim).to(event_times)
        pe[:, :, 0::2] = torch.sin(event_times[..., None] * self.div_term)
        pe[:, :, 1::2] = torch.cos(event_times[..., None] * self.div_term)
        return pe


def test_einsum_op():

    def squared_dist_fn(x, y):
        return torch.norm(x - y)**2

    def squared_dist_fn2(x, y):
        return dot_product_fn(x, x) + dot_product_fn(y, y) - 2 * dot_product_fn(x, y)

    def dot_product_fn(x, y):
        return torch.sum(x * y)

    T, N, H, D = 4, 1, 1, 10

    q = torch.randn(T, N, H, D)
    k = torch.randn(T, N, H, D)

    squared_dist = torch.zeros(T, T, N, H)
    squared_dist2 = torch.zeros(T, T, N, H)
    dot_product = torch.zeros(T, T, N, H)
    for t in range(T):
        for s in range(T):
            for n in range(N):
                for h in range(H):
                    squared_dist[t, s] = squared_dist_fn(q[t, n, h], k[s, n, h])
                    dot_product[t, s] = dot_product_fn(q[t, n, h], k[s, n, h])
                    squared_dist2[t, s] = squared_dist_fn2(q[t, n, h], k[s, n, h])

    einsum_sqdist = torch.einsum('tbhd,tbhd->tbh', q, q).unsqueeze(1) + torch.einsum('sbhd,sbhd->sbh', k, k).unsqueeze(0) - 2 * torch.einsum('tbhd,sbhd->tsbh', q, k)
    einsum_dotproduct = torch.einsum('tbhd,sbhd->tsbh', q, k)

    print("squared dist", squared_dist.reshape(T, T))
    print("squared dist 2", squared_dist2.reshape(T, T))
    print("einsum squared dist", einsum_sqdist.reshape(T, T))

    print("dot product", dot_product.reshape(T, T))
    print("einsum dot product", einsum_dotproduct.reshape(T, T))


def test_attn_mask():

    torch.set_default_dtype(torch.float64)

    T, N, D = 8, 1, 20

    attn_mask = torch.triu(torch.ones(T, T), diagonal=1) * -1e12

    x = torch.randn(T * N * D).requires_grad_(True)
    mha = L2MultiheadAttention(D, 1)

    y = mha(x.reshape(T, N, D), attn_mask=attn_mask)
    yhat = mha(x.reshape(T, N, D), attn_mask=attn_mask, rm_nonself_grads=True)
    print(torch.norm(y - yhat))

    # Construct full Jacobian.
    def func(x):
        return mha(x.reshape(T, N, D), attn_mask=attn_mask).reshape(-1)

    jac = torch.autograd.functional.jacobian(func, x)

    # Exact diagonal block of Jacobian.
    jac = jac.reshape(T, D, T, D)
    blocks = []
    for i in range(T):
        blocks.append(jac[i, :, i, :])
    jac_block_diag = torch.block_diag(*blocks)

    # Simulated diagonal block of Jacobian.
    def selfonly_func(x):
        return mha(x.reshape(T, N, D), attn_mask=attn_mask, rm_nonself_grads=True).reshape(-1)
    simulated_jac_block_diag = torch.autograd.functional.jacobian(selfonly_func, x)

    print(torch.norm(simulated_jac_block_diag - jac_block_diag))

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(jac_block_diag)
    axs[1].imshow(simulated_jac_block_diag)
    axs[2].imshow(torch.abs(simulated_jac_block_diag - jac_block_diag))
    plt.savefig("jacobian.png")


if __name__ == "__main__":
    test_attn_mask()
