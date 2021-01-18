# Copyright (c) Facebook, Inc. and its affiliates.

import torch.nn as nn


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, *layers):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layers)

    def forward(self, x, logpx=None, reverse=False, inds=None, **kwargs):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        if logpx is None:
            for i in inds:
                x = self.chain[i](x, reverse=reverse, **kwargs)
            return x
        else:
            for i in inds:
                x, logpx = self.chain[i](x, logpx, reverse=reverse, **kwargs)
            return x, logpx
