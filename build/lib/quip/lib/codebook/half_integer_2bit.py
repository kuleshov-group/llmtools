"""
Rounds to the grid [-1.5, -0.5, 0.5, 1.5]
"""

import torch
from torch import nn

from lib.utils.matmul_had import matmul_hadU_cuda, matmul_hadUt_cuda


def get_grid():
    return torch.tensor([[-1.5], [-0.5], [0.5], [1.5]])


_HI_CODESZ = 1


class half_integer_2bit(nn.Module):

    def __init__(self):
        super(half_integer_2bit, self).__init__()
        self.register_buffer('grid', get_grid())
        self.register_buffer('grid_norm', torch.diag(self.grid @ self.grid.T))
        self.opt_scale = 1.0  # the actual number is close to 1.0
        self.codesz = _HI_CODESZ
        self.idx_dtype = torch.uint8

    def quantize(self, X, return_idx=True):
        assert X.shape[-1] == self.codesz
        Xqidx = (2 * X @ self.grid.T - self.grid_norm).argmax(-1)
        if return_idx:
            return self.grid[Xqidx], Xqidx.to(self.idx_dtype)
        return self.grid[Xqidx]

    def by_idxs(self, idxs):
        return self.grid[idxs.int()]


class HalfInteger2BitLinear(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.codebook = half_integer_2bit().to(torch.float16).to(device)

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

        x = input.view(-1, n * _HI_CODESZ).to(torch.float32)
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

        W_decompressed = self.codebook.by_idxs(Qidxs).reshape(-1, n * _HI_CODESZ)
        z = x @ W_decompressed.t()

        x = z.to(torch.float32)
        x = x * (Wscale * num_scale)

        if rank > 0:
            x = x + ABx.to(torch.float32)

        x = matmul_hadU_cuda(x)
        x = x * SV

        output = x.view(*input.shape[:-1], m)

        return output
