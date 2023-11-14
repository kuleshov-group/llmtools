"""
Rounds to the grid [-1.5, -0.5, 0.5, 1.5]
"""

import torch
from torch import nn

from quip.lib.utils.matmul_had import matmul_hadU_cuda, matmul_hadUt_cuda


def get_grid():
    base = torch.arange(-3, 4).float() - 0.5
    grid = torch.cartesian_prod(*[base] * 8)
    gn = grid.norm(dim=-1)**2
    return grid[torch.where(gn <= 10)[0]]


_HI8C_CODESZ = 8
_HI8C_GRID = get_grid()
_HI8C_GRID_NORM = torch.diag(_HI8C_GRID @ _HI8C_GRID.T)


class half_integer_2bit_8col(nn.Module):

    def __init__(self, build_truncated=True):
        super(half_integer_2bit_8col, self).__init__()
        self.register_buffer('grid', _HI8C_GRID)
        self.register_buffer('grid_norm', _HI8C_GRID_NORM)

        self.opt_scale = 1.04
        self.codesz = _HI8C_CODESZ
        self.idx_dtype = torch.int16
        self.idx_offset = -2**15

        if build_truncated:
            # should have 227 entries
            self.register_buffer('grid_part', torch.unique(torch.abs(_HI8C_GRID), dim=0))
            self.register_buffer('grid_part_norm', torch.diag(self.grid_part @ self.grid_part.T))
            self.register_buffer('int_map', 2**torch.arange(_HI8C_CODESZ))

    def round(self, X, grid, grid_norm):
        assert X.shape[-1] == self.codesz
        Xqidx = (2 * X @ grid.T - grid_norm).argmax(-1)
        return grid[Xqidx], Xqidx

    def quantize(self, X, return_idx=True):
        flips = self.to_int(X < 0) << 8
        mask = 1 - 2 * (X < 0).to(torch.float32)

        X_part = torch.abs(X)
        roundout, Xqidx = self.round(X_part, self.grid_part, self.grid_part_norm)
        vals = roundout * mask
        if not return_idx:
            return vals
        return vals, (flips + Xqidx + self.idx_offset).to(self.idx_dtype)

    def to_int(self, mask):
        return (self.int_map.unsqueeze(0) * mask.int()).sum(dim=-1)

    def to_mask(self, int):
        return 1 - 2 * ((self.int_map & int.unsqueeze(-1)) > 0)

    def by_idxs(self, idxs):
        idxs = idxs.int() - self.idx_offset
        mask = self.to_mask(idxs >> 8)
        return self.grid_part[idxs & 255] * mask


class HalfInteger2Bit8ColLinear(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.codebook = half_integer_2bit_8col().to(torch.float16).to(device)

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

        x = input.view(-1, n * _HI8C_CODESZ).to(torch.float32)
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

        W_decompressed = self.codebook.by_idxs(Qidxs).reshape(-1, n * _HI8C_CODESZ)
        z = x @ W_decompressed.t()

        x = z.to(torch.float32)
        x = x * (Wscale * num_scale)

        if rank > 0:
            x = x + ABx.to(torch.float32)

        x = matmul_hadU_cuda(x)
        x = x * SV

        output = x.view(*input.shape[:-1], m)

        return output
