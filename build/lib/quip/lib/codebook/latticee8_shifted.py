"""
Rounds to the grid [-1.5, -0.5, 0.5, 1.5]
"""

import torch
import math
from torch import nn
from functools import cache
from quip.lib.utils.matmul_had import matmul_hadU_cuda, matmul_hadUt_cuda

_E8S_CODESZ = 8
_E8S_OFFSET = 1 / 4


def get_grid():
    intr = torch.arange(-4, 4)
    d8 = torch.cartesian_prod(*[intr] * _E8S_CODESZ).float()
    d8m2 = (d8.sum(dim=-1) % 2 == 0)
    d8n = d8.norm(dim=-1)**2 <= 10
    d8 = d8[torch.where(d8m2 * d8n)[0]]

    return torch.concat([d8 - _E8S_OFFSET, d8 + _E8S_OFFSET], dim=0)


_E8S_CACHED = get_grid()
_E8S_NORM_CACHED = torch.diag(_E8S_CACHED @ _E8S_CACHED.T)


class E8S_codebook(nn.Module):

    def __init__(self, build_truncated=True):
        super(E8S_codebook, self).__init__()
        self.opt_scale = 1  # the actual scale is ~1/0.99
        self.codesz = _E8S_CODESZ
        self.idx_dtype = torch.int16
        self.idx_offset = -2**15
        self.register_buffer('grid', _E8S_CACHED)
        self.register_buffer('grid_norm', _E8S_NORM_CACHED)

        '''
        self.to('cuda')
        samples = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(8), torch.eye(8)).rsample([20, 100000]).cuda()
        for s in torch.arange(0.8, 1.2, 0.01):
            mean = 0
            for i in range(len(samples)):
                mean += ((self.quantize(samples[i]*s, False)/s - samples[i]).norm(dim=-1)**2).mean()
            print(s, mean/len(samples))
        exit()
        '''


    def round(self, X, grid, grid_norm):
        assert X.shape[-1] == self.codesz
        Xqidx = (2 * X @ grid.T - grid_norm).argmax(-1)
        return grid[Xqidx], Xqidx

    def quantize(self, X, return_idx=True):
        vals, idx = self.round(X, self.grid, self.grid_norm)
        if return_idx:
            return vals, (idx + self.idx_offset).to(self.idx_dtype)
        return vals

    def by_idxs(self, idxs):
        return self.grid[idxs.int() - self.idx_offset]


class QuantizedE8SLinear(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.codebook = E8S_codebook(build_truncated=False).to(device)
        self.codebook.grid = self.codebook.grid.to(torch.float16)

    def forward(self,
                input,
                Qidxs,
                SU,
                SV,
                Wscale,
                rank=-1,
                A=None,
                B=None,
                rescale_WH=False,
                scaleWH=None):
        (m, n) = Qidxs.shape

        x = input.view(-1, n * _E8S_CODESZ).to(torch.float32)
        if rescale_WH:
            x /= scaleWH
        x = x * SU
        x = matmul_hadUt_cuda(x)

        if rank > 0:
            Bx = x @ B.t().to(torch.float32)
            ABx = Bx @ A.t().to(torch.float32)

        num_scale = 1024
        x = x / num_scale
        x = x.to(torch.float16)

        W_decompressed = self.codebook.by_idxs(Qidxs).reshape(-1, n * _E8S_CODESZ)
        z = x @ W_decompressed.t()

        x = z.to(torch.float32)
        x = x * (Wscale * num_scale)

        if rank > 0:
            x = x + ABx.to(torch.float32)

        x = matmul_hadU_cuda(x)
        x = x * SV

        output = x.view(*input.shape[:-1], m)

        return output
