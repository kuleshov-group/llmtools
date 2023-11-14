"""
Rounds to the grid [-1.5, -0.5, 0.5, 1.5]
"""

import torch
import math
from torch import nn
from functools import cache
from quip.lib.utils.matmul_had import matmul_hadU_cuda, matmul_hadUt_cuda

_E8_CODESZ = 8


def get_grid():
    intr = torch.arange(-4, 4)
    hintr = intr + 1 / 2

    gintr = torch.cartesian_prod(*[intr] * _E8_CODESZ)
    ghintr = torch.cartesian_prod(*[hintr] * _E8_CODESZ)

    ge8 = torch.concat([gintr, ghintr], dim=0)
    ge8m2 = (ge8.sum(dim=-1) % 2 == 0)
    ge8n = ge8.norm(dim=-1)**2 <= 10

    e8 = ge8[torch.where(ge8m2 * ge8n)[0]]
    return e8


_E8_CACHED = get_grid()
_E8_NORM_CACHED = torch.diag(_E8_CACHED @ _E8_CACHED.T)


class E8_codebook(nn.Module):

    def __init__(self, build_truncated=True):
        super(E8_codebook, self).__init__()
        self.opt_scale = 1  # the actual scale is ~1/0.99
        self.codesz = _E8_CODESZ
        self.idx_dtype = torch.int16
        self.idx_offset = -2**15

        grid = _E8_CACHED
        self.register_buffer('grid', grid)
        self.register_buffer('grid_norm', _E8_NORM_CACHED)

        if build_truncated:
            idxs = torch.where(
                ((grid[:, 1:] < 0).sum(dim=-1) <= 1) * \
                (grid[:, 1:].min(dim=-1).values >= -0.5)
            )[0]
            grid_part = grid[idxs]
            self.register_buffer('grid_part', grid_part)
            self.register_buffer('grid_part_norm', torch.diag(grid_part @ grid_part.T))
            self.register_buffer('int_map', 2**torch.arange(_E8_CODESZ))
            allcombo_idx, idx_map = self.iterate_mask()
            self.register_buffer('allcombo_idx', allcombo_idx)
            self.register_buffer('idx_map', idx_map)

    def to_int(self, mask):
        return (self.int_map.unsqueeze(0) * mask.int()).sum(dim=-1)

    def to_mask(self, int):
        return ((self.int_map & int.unsqueeze(-1)) > 0) * 2 - 1

    def round(self, X, grid, grid_norm):
        assert X.shape[-1] == self.codesz
        Xqidx = (2 * X @ grid.T - grid_norm).argmax(-1)
        return grid[Xqidx], Xqidx

    def iterate_mask(self, device=0):
        bmask = 2**torch.arange(8)
        flips = torch.stack([((torch.tensor([i]) & bmask) > 0).int()
                             for i in range(2**8)]).to(device)
        raw_idx = torch.where(flips.sum(dim=-1) % 2 == 0)[0]
        flips = 1 - 2 * flips[raw_idx]
        idx_map = torch.zeros(2**8, dtype=torch.int32)
        for i in range(len(raw_idx)):
            idx_map[raw_idx[i]] = i
        allcombo = flips.unsqueeze(1) * self.grid_part.unsqueeze(0).to(device)
        allcombo_idx = torch.zeros(allcombo.shape[0:2], dtype=self.idx_dtype)
        for i in range(len(allcombo)):
            allcombo_idx[i] = (
                self.round(allcombo[i], self.grid.to(device), self.grid_norm.to(device))[1] +
                self.idx_offset).to(self.idx_dtype)
        return allcombo_idx.cpu(), idx_map.cpu()

    def quantize(self, X, return_idx=True):
        X_part = torch.abs(X)
        X_odd = torch.where((X < 0).sum(dim=-1) % 2 != 0)[0]
        X_part[X_odd, 0] = -X_part[X_odd, 0]
        mask = 1 - 2 * (X < 0).to(torch.float32)
        mask[X_odd, 0] = -mask[X_odd, 0]
        roundout, Xqidx = self.round(X_part, self.grid_part, self.grid_part_norm)
        vals = roundout * mask

        if not return_idx:
            return vals

        real_idx = self.allcombo_idx[self.idx_map[self.to_int((1 - mask) / 2)], Xqidx]
        return vals, real_idx

    def by_idxs(self, idxs):
        return self.grid[idxs.int() - self.idx_offset]


class QuantizedE8Linear(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.codebook = E8_codebook(build_truncated=False).to(device)
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

        x = input.view(-1, n * _E8_CODESZ).to(torch.float32)
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

        W_decompressed = self.codebook.by_idxs(Qidxs).reshape(-1, n * _E8_CODESZ)
        z = x @ W_decompressed.t()

        x = z.to(torch.float32)
        x = x * (Wscale * num_scale)

        if rank > 0:
            x = x + ABx.to(torch.float32)

        x = matmul_hadU_cuda(x)
        x = x * SV

        output = x.view(*input.shape[:-1], m)

        return output
