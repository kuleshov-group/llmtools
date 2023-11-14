"""
Rounds to the grid [-1.5, -0.5, 0.5, 1.5]
"""

import torch
import math
from torch import nn
from functools import cache
from quip.lib.utils.matmul_had import matmul_hadU_cuda, matmul_hadUt_cuda

_E81B_CODESZ = 8


def get_grid():
    intr = torch.arange(-2, 2)
    hintr = intr + 1 / 2

    gintr = torch.cartesian_prod(*[intr] * _E81B_CODESZ)
    ghintr = torch.cartesian_prod(*[hintr] * _E81B_CODESZ)

    ge8 = torch.concat([gintr, ghintr], dim=0)
    ge8m2 = (ge8.sum(dim=-1) % 2 == 0)
    ge8n = ge8.norm(dim=-1)**2 <= 3

    e8 = ge8[torch.where(ge8m2 * ge8n)[0]]
    extra = torch.tensor([
        [1, 1, 1, 0, 0, 1, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 1, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 0, 1, 0, 1, 0],
        [1, 0, 0, 1, 1, 0, 1, 0],
        [0, 0, 1, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 1, 0],
        [1, 1, 1, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 1, 0, 0, 1],
        [0, 0, 1, 1, 1, 0, 0, 1],
        [0, 1, 1, 1, 0, 0, 0, 1],
    ])
        
    e8 = torch.concat([e8, extra], dim=0)
    return e8


_E81B_CACHED = get_grid()
_E81B_NORM_CACHED = torch.diag(_E81B_CACHED @ _E81B_CACHED.T)

class E81B_codebook(nn.Module):

    def __init__(self, build_truncated=True):
        super(E81B_codebook, self).__init__()
        self.opt_scale = 0.63  # the actual scale is ~1/0.99        
        self.codesz = _E81B_CODESZ
        self.idx_dtype = torch.uint8
        self.register_buffer('grid', _E81B_CACHED)
        self.register_buffer('grid_norm', _E81B_NORM_CACHED)

    def quantize(self, X, return_idx=True):
        assert X.shape[-1] == self.codesz
        Xqidx = (2 * X @ self.grid.T - self.grid_norm).argmax(-1)
        if return_idx:
            return self.grid[Xqidx], Xqidx.to(self.idx_dtype)
        return self.grid[Xqidx]
        
    def by_idxs(self, idxs):
        return self.grid[idxs.int()]


class QuantizedE81BLinear(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.codebook = E81B_codebook(build_truncated=False).to(device)
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

        x = input.view(-1, n * _E81B_CODESZ).to(torch.float32)
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

        W_decompressed = self.codebook.by_idxs(Qidxs).reshape(-1, n * _E81B_CODESZ)
        z = x @ W_decompressed.t()

        x = z.to(torch.float32)
        x = x * (Wscale * num_scale)

        if rank > 0:
            x = x + ABx.to(torch.float32)

        x = matmul_hadU_cuda(x)
        x = x * SV

        output = x.view(*input.shape[:-1], m)

        return output
