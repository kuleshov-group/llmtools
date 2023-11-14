"""
D8^ = D8 + 1/2 intersected with ball of radius sqrt(10)
|D8^| has 227 entries
We then add 29 entries from the set of vectors with 5 3/2 and 3 1/2
The total codebook is all 2^7 flips of these 256 entries (2^15) +- 1/4
which makes 2^16 entries.
This corresponds to a subset of E8 + 1/4
"""

import torch
import math
from torch import nn
from functools import cache
import itertools
from quip.lib.utils.matmul_had import matmul_hadU_cuda, matmul_hadUt_cuda

_E8P_CODESZ = 8
_INT_MAP = 2**(torch.arange(_E8P_CODESZ).flip(0))


def int2mask(i, int_map):
    return ((i & int_map) > 0).int()


def mask2int(mask, int_map):
    return (int_map.unsqueeze(0) * mask.int()).sum(dim=-1)


def get_abs_grid():
    intr = torch.arange(-4, 4)
    d8 = torch.cartesian_prod(*[intr] * _E8P_CODESZ).float() + 1 / 2
    d8m2 = (d8.sum(dim=-1) % 2 == 0)
    d8n = d8.norm(dim=-1)**2 <= 10
    d8abs = torch.unique(d8[sorted(torch.where(d8m2 * d8n)[0])].abs(), dim=0)

    norm12 = torch.tensor([
        [3, 1, 1, 1, 3, 3, 3, 3],
        [1, 3, 1, 1, 3, 3, 3, 3],
        [1, 1, 3, 1, 3, 3, 3, 3],
        [1, 1, 1, 3, 3, 3, 3, 3],
        [3, 3, 3, 1, 3, 3, 1, 1],
        [3, 3, 3, 1, 3, 1, 3, 1],
        [3, 3, 3, 1, 1, 3, 3, 1],
        [3, 3, 3, 1, 3, 1, 1, 3],
        [3, 3, 3, 1, 1, 3, 1, 3],
        [3, 3, 3, 1, 1, 1, 3, 3],
        [3, 3, 1, 3, 3, 3, 1, 1],
        [3, 3, 1, 3, 3, 1, 3, 1],
        [3, 3, 1, 3, 1, 3, 3, 1],
        [3, 3, 1, 3, 3, 1, 1, 3],
        [3, 3, 1, 3, 1, 3, 1, 3],
        [3, 3, 1, 3, 1, 1, 3, 3],
        [3, 1, 3, 3, 3, 3, 1, 1],
        [3, 1, 3, 3, 3, 1, 3, 1],
        [3, 1, 3, 3, 1, 3, 3, 1],
        [3, 1, 3, 3, 3, 1, 1, 3],
        [3, 1, 3, 3, 1, 3, 1, 3],
        [1, 3, 3, 3, 1, 1, 3, 3],
        [1, 3, 3, 3, 3, 3, 1, 1],
        [1, 3, 3, 3, 3, 1, 3, 1],
        [1, 3, 3, 3, 1, 3, 3, 1],
        [1, 3, 3, 3, 3, 1, 1, 3],
        [1, 3, 3, 3, 1, 3, 1, 3],
        [1, 3, 3, 3, 1, 1, 3, 3],
        [3, 3, 1, 1, 3, 3, 3, 1],
    ]) / 2
    return torch.concat([d8abs, norm12], dim=0)


def get_full_grid(abs_grid):
    """
    idx format:
        - first 8 bits = which of the 256 entries in the abs grid
        - next 7 bits = which of the right 7 dims to negate (8th can be inferred)
        - last bit = +1/4 if true else -1/4
    """
    is_even_flips = abs_grid.sum(dim=-1) % 2 == 0
    abs_idxs = torch.arange(len(abs_grid)) << _E8P_CODESZ
    entries = [[], []]
    idxs = [[], []]
    for i in range(2**(_E8P_CODESZ - 1)):
        mask = int2mask(i, _INT_MAP)
        mask_even = (mask.sum(dim=-1) % 2 == 0)
        mask = mask.unsqueeze(0).repeat(len(abs_grid), 1)
        mask[:, 0] = mask_even != is_even_flips
        mask = 1 - 2 * mask
        entries[0].append(abs_grid * mask + 1 / 4)
        idxs[0].append(abs_idxs + (i << 1) + 1)
        entries[1].append(abs_grid * mask - 1 / 4)
        idxs[1].append(abs_idxs + (i << 1))

    for i in range(2):
        entries[i] = torch.concat(entries[i], dim=0)
        idxs[i] = torch.concat(idxs[i], dim=0)
    entries = torch.concat(entries, dim=0)
    idxs = torch.concat(idxs, dim=0)
    return entries, idxs


_E8P_ABS_CACHED = get_abs_grid()
_E8P_GRID, _E8P_GRID_IDX = get_full_grid(_E8P_ABS_CACHED)

class E8P12_codebook(nn.Module):

    def __init__(self, build_truncated=True):
        super(E8P12_codebook, self).__init__()
        self.opt_scale = 1#.03#/1.09
        self.codesz = _E8P_CODESZ
        self.idx_dtype = torch.int16
        self.idx_offset = -2**15

        self.register_buffer('grid_abs', _E8P_ABS_CACHED)
        self.register_buffer('grid_abs_even', self.grid_abs.sum(dim=-1) % 2 == 0)
        self.register_buffer('int_map', _INT_MAP)
        self.register_buffer('grid', _E8P_GRID)
        self.register_buffer('grid_idx_map', (_E8P_GRID_IDX + self.idx_offset).to(self.idx_dtype))
        idx_lut = torch.zeros(_E8P_GRID_IDX.shape).int()
        idx_lut[_E8P_GRID_IDX] = torch.arange(len(_E8P_GRID_IDX)).int()
        self.register_buffer('grid_idx_inv', idx_lut)
        
        if build_truncated:
            self.register_buffer('grid_norm', torch.diag(self.grid @ self.grid.T))            
            grid_part = self.grid[:len(self.grid) // 2] - 1 / 4
            idxs = torch.where(
                ((grid_part[:, 1:] < 0).sum(dim=-1) <= 1) * \
                (grid_part[:, 1:].min(dim=-1).values >= -0.5)
            )[0]
            grid_part = grid_part[idxs]
            self.register_buffer('grid_part', grid_part)
            self.register_buffer('grid_part_norm', torch.diag(grid_part @ grid_part.T))
            allcombo_idx, idx_map = self.iterate_mask()
            self.register_buffer('allcombo_idx', allcombo_idx)
            self.register_buffer('idx_map', idx_map)

            '''
            self.to('cuda')
            samples = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(8), torch.eye(8)).rsample([2000000]).cuda()
            for s in torch.arange(0.8, 1.2, 0.01):
                print(s, ((self.quantize(samples*s, False)/s - samples).norm(dim=-1)**2).mean())
            exit()
            '''


    def iterate_mask(self, device=0):
        flips = torch.stack([((torch.tensor([i]) & self.int_map) > 0).int()
                             for i in range(2**_E8P_CODESZ)]).to(device)
        raw_idx = torch.where(flips.sum(dim=-1) % 2 == 0)[0]
        flips = 1 - 2 * flips[raw_idx]
        idx_map = torch.zeros(2**_E8P_CODESZ, dtype=torch.int32)
        for i in range(len(raw_idx)):
            idx_map[raw_idx[i]] = i
        allcombo = flips.unsqueeze(1) * self.grid_part.unsqueeze(0).to(device)
        allcombo_idx = torch.zeros(allcombo.shape[0:2]).int()
        for i in range(len(allcombo)):
            allcombo_idx[i] = self.round(allcombo[i], self.grid.to(device), self.grid_norm.to(device))[1]
        return allcombo_idx.cpu(), idx_map.cpu()

    def round(self, X, grid, grid_norm):
        assert X.shape[-1] == self.codesz
        Xqidx = (2 * X @ grid.T - grid_norm).argmax(-1)
        return grid[Xqidx], Xqidx

    def fast_quantize_part(self, X):
        X_part = torch.abs(X)
        X_odd = torch.where((X < 0).sum(dim=-1) % 2 != 0)[0]
        X_part[X_odd, 0] = -X_part[X_odd, 0]
        mask = 1 - 2 * (X < 0).to(torch.float32)
        mask[X_odd, 0] = -mask[X_odd, 0]
        roundout, Xqidx = self.round(X_part, self.grid_part, self.grid_part_norm)
        vals = roundout * mask
        real_idx = self.allcombo_idx[self.idx_map[mask2int((1 - mask) / 2, self.int_map)], Xqidx]
        err = (X - vals).norm(dim=-1)
        return vals, real_idx, err

    def quantize(self, X, return_idx=True):
        X_plus = X + 1 / 4  # quantize X to D8^ - 1/4
        X_minus = X - 1 / 4  # quantize X to D8^ + 1/4

        plus_vals, plus_idx, plus_err = self.fast_quantize_part(X_plus)
        minus_vals, minus_idx, minus_err = self.fast_quantize_part(X_minus)
        plus_idx = plus_idx + 2**15

        which = plus_err < minus_err
        final_vals = torch.where(which.unsqueeze(-1), plus_vals - 1 / 4, minus_vals + 1 / 4)

        if return_idx:
            final_idxs = self.grid_idx_map[torch.where(which, plus_idx, minus_idx)]
            return final_vals, final_idxs
        
        return final_vals

    def by_idxs(self, idxs):
        return self.grid[self.grid_idx_inv[idxs.int() - self.idx_offset]]

class QuantizedE8P12Linear(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.codebook = E8P12_codebook(build_truncated=False).to(device).to(torch.float16)

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

        x = input.view(-1, n * _E8P_CODESZ).to(torch.float32)
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


        W_decompressed = self.codebook.by_idxs(Qidxs).reshape(-1, n * _E8P_CODESZ)
        z = x @ W_decompressed.t()

        x = z.to(torch.float32)
        x = x * (Wscale * num_scale)

        if rank > 0:
            x = x + ABx.to(torch.float32)

        x = matmul_hadU_cuda(x)
        x = x * SV

        output = x.view(*input.shape[:-1], m)

        return output
