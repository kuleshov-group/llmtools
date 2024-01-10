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
import quiptools_cuda

from quip.lib.linear.autograd import AutogradQuipE8, AutogradOrthoMult

_E8P_CODESZ = 8

def get_norm12():
    # 29 elements of norm 12 in E8 + 1/4
    return torch.tensor([
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
        [1, 1, 3, 3, 1, 3, 3, 3],
        [3, 3, 1, 1, 3, 3, 3, 1],
    ]) / 2


def get_packed_abs_grid():
    intr = torch.arange(-4, 4)
    d8 = torch.cartesian_prod(*[intr] * 8).float() + 1 / 2
    d8m2 = (d8.sum(dim=-1) % 2 == 0)
    d8n = d8.norm(dim=-1)**2 <= 10
    d8abs = torch.unique(d8[sorted(torch.where(d8m2 * d8n)[0])].abs(), dim=0)
    norm12 = get_norm12()
    cba = torch.concat([d8abs, norm12], dim=0)
    cba = cba[:, [0, 2, 4, 6, 1, 3, 5, 7]]
    cba[:,7] *= (1 - 2 * (cba.sum(1) % 2))
    cba = cba * 2 + 8
    cba = cba.to(torch.int32)
    acc = cba[:,0]
    for i in range(7):
        acc = acc | (cba[:,(i+1)] << ((i+1)*4))
    return acc


def get_abs_grid():
    intr = torch.arange(-4, 4)
    d8 = torch.cartesian_prod(*[intr] * _E8P_CODESZ).float() + 1 / 2
    d8m2 = (d8.sum(dim=-1) % 2 == 0)
    d8n = d8.norm(dim=-1)**2 <= 10
    d8abs = torch.unique(d8[sorted(torch.where(d8m2 * d8n)[0])].abs(), dim=0)
    norm12 = get_norm12()
    cba = torch.concat([d8abs, norm12], dim=0)
    return cba


def get_full_grid(packed_abs_grid):
    synth_codebook = torch.zeros(1 << 16, 8)
    parity_idx = []
    shuffle_map = [0,4,1,5,2,6,3,7] 
    for c in range(1 << 16):
        signs = c & 255
        abs = c >> 8
        parity = 0
        for i in range(8):
            parity = parity ^ ((signs >> i) & 1)
        signs = signs ^ parity
        abs_code = packed_abs_grid[abs].item()
        for i in range(8):
            ii = shuffle_map[i]
            synth_codebook[c,i] = (((abs_code >> (4 * ii)) & 15) - 8) * 0.5
            if ((signs >> ii) & 1):
                synth_codebook[c,i] *= -1
        if parity:
            synth_codebook[c,:] -= 0.25
            parity_idx.append(c)
        else:
            synth_codebook[c,:] += 0.25
    return synth_codebook, torch.arange(1 << 16), parity_idx

_E8P_PACKED_ABS_CACHED = get_packed_abs_grid()
_E8P_GRID, _E8P_GRID_IDX, _PARITY_IDX = get_full_grid(_E8P_PACKED_ABS_CACHED)


class E8P12_codebook(nn.Module):

    def __init__(self, inference=False):
        super(E8P12_codebook, self).__init__()
        self.opt_scale = 1.03
        self.codesz = _E8P_CODESZ
        self.idx_dtype = torch.int64
        self.packsz = 4
        self.pack_out = False
        self.version = 1

        self.register_buffer('grid_packed_abs', _E8P_PACKED_ABS_CACHED)

        if not inference:
            self.register_buffer('grid', _E8P_GRID)
            self.register_buffer('grid_norm', _E8P_GRID.norm(dim=-1)**2)
            grid_part = _E8P_GRID[_PARITY_IDX] + 0.25
            grid_part = grid_part[
                torch.where(
                    ((grid_part[:, :7] < 0).sum(dim=-1) <= 1) * \
                    (grid_part[:, :7].min(dim=-1).values >= -0.5)
                )[0]]
            self.register_buffer('grid_part', grid_part)
            self.register_buffer('grid_part_norm', grid_part.norm(dim=-1)**2)
            abs_grid = get_abs_grid()
            self.register_buffer('grid_abs_odd', abs_grid.sum(dim=-1) % 2 == 1)
            self.register_buffer('part_abs_map', self.round(grid_part.abs(), abs_grid, abs_grid.norm(dim=-1)**2)[1])
            self.register_buffer('bit_map', 2**torch.arange(8))
            '''
            self.to('cuda')
            samples = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(8), torch.eye(8)).rsample([2000000]).cuda()
            for s in torch.arange(0.8, 1.2, 0.01):
                print(s, ((self.quantize(samples*s, False)/s - samples).norm(dim=-1)**2).mean())
            exit()
            '''

    def round(self, X, grid, grid_norm):
        assert X.shape[-1] == self.codesz
        Xqidx = (2 * X @ grid.T - grid_norm).argmax(-1)
        return grid[Xqidx], Xqidx

    def fast_quantize_part(self, X, parity):
        X_part = torch.abs(X)
        X_odd = torch.where((X < 0).sum(dim=-1) % 2 != 0)[0]
        X_part[X_odd, 7] = -X_part[X_odd, 7]
        mask = 1 - 2 * (X < 0).to(torch.float32)
        mask[X_odd, 7] = -mask[X_odd, 7]
        roundout, Xqidx = self.round(X_part, self.grid_part, self.grid_part_norm)
        vals = roundout * mask
        err = (X - vals).norm(dim=-1)
        abs_idx = self.part_abs_map[Xqidx]
        sign_mask = (((roundout < 0) ^ (mask < 0))[:, [0, 2, 4, 6, 1, 3, 5, 7]])
        sign_mask[:, 7] = sign_mask[:, 7] ^ self.grid_abs_odd[abs_idx]
        sign_mask[:, 0] = sign_mask[:, 0] ^ parity
        mask_idx = (sign_mask * self.bit_map).sum(dim=-1).int()
        idx = (abs_idx << 8) + mask_idx
        return vals, idx, err

    def quantize(self, X, return_idx=True, **kwargs):
        X_plus = X + 1 / 4  # quantize X to D8^ - 1/4
        X_minus = X - 1 / 4  # quantize X to D8^ + 1/4

        plus_vals, plus_idx, plus_err = self.fast_quantize_part(X_plus, True)
        minus_vals, minus_idx, minus_err = self.fast_quantize_part(X_minus, False)
        
        which = plus_err < minus_err
        final_vals = torch.where(which.unsqueeze(-1), plus_vals - 1 / 4, minus_vals + 1 / 4)
        final_idx = torch.where(which, plus_idx, minus_idx)
       
        if return_idx:
            return final_vals, final_idx

        return final_vals

    def maybe_pack_idxs(self, idxs):        
        m, n = idxs.shape
        idxs = idxs.view(m//2, 2, (n*8)//16, 2).transpose(1, 2).contiguous()

        abs32 = (idxs[:, :, 0, 0] >> 8) + \
            ((idxs[:, :, 1, 0] >> 8) << 8) + \
            ((idxs[:, :, 0, 1] >> 8) << 16) + \
            ((idxs[:, :, 1, 1] >> 8) << 24)

        sign32 = torch.zeros(abs32.shape, dtype=abs32.dtype, device=abs32.device)
        for i in range(4):
            wt = idxs[:, :, i % 2, i // 2]
            for j in range(8):
                sign32 += ((wt >> j) & 1) << (4*j + i)

        output = (sign32 << 32) + abs32
        output = output.reshape(m//16, 8, n//8, 4).transpose(1, 2).contiguous()
        return output.view(m, n//4)
        
    def by_idxs(self, idxs, **kwargs):
        m, n = idxs.shape
        W_decompressed = quiptools_cuda.decompress_packed_e8p(
            idxs.view(m//16, n//2, 8, 4),
            self.grid_packed_abs
        )
        return W_decompressed



class QuantizedE8P12Linear(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.codebook = E8P12_codebook(inference=True).to(torch.float32).to(device)

    def maybe_unpack_idxs(self, idxs):
        return idxs
    
    def forward(self,
                input,
                Qidxs,
                SU,
                SV,
                Wscale,
                had_left,
                had_right,
                K_left,
                K_right,
                rank=-1,
                A=None,
                B=None,
                rescale_WH=False,
                scaleWH=None,
                **kwargs):
        #breakpoint()
        if input.isnan().any() or input.isinf().any():
            breakpoint()

        n, m = len(SU), len(SV)

        x = input.view(-1, n).to(torch.float32)
        
        if x.isnan().any() or x.isinf().any():
            breakpoint()

        if rescale_WH:
            x /= scaleWH
        x = x * SU
        if x.isnan().any() or x.isinf().any():
            breakpoint()

        #* Custom Cuda *#
        transpose = torch.ones(1).to(x.device)
        # x = AutogradOrthoMult.apply(x, transpose, had_left, had_right, K_left, K_right)
        if x.isnan().any() or x.isinf().any():
            breakpoint()
        x = matmul_hadUt_cuda(x, had_left, K_left)

        if rank > 0:
            Bx = x @ B.t().to(torch.float32)
            ABx = Bx @ A.t().to(torch.float32)

        W_decompressed = quiptools_cuda.decompress_packed_e8p(
            Qidxs.view(m//16, n//64, 8, 4),
            self.codebook.grid_packed_abs
        )
        x = (x.to(torch.float32) @ W_decompressed.T.to(torch.float32)).to(torch.float32)
        #z = AutogradQuipE8.apply(x, self.codebook.grid_packed_abs, Qidxs, m, n)

        if x.isnan().any() or x.isinf().any():
            breakpoint()

        x = x.to(torch.float32)

        if x.isnan().any() or x.isinf().any():
            breakpoint()

        x *= Wscale
        if x.isnan().any() or x.isinf().any():
            breakpoint()

        if rank > 0:
            x = x + ABx.to(torch.float32)

        #* Custom Cuda *#
        transpose = torch.zeros(1).to(x.device)
        #x = AutogradOrthoMult.apply(x, transpose, had_left, had_right, K_left, K_right)
        x = matmul_hadU_cuda(x, had_right, K_right)
        if x.isnan().any() or x.isinf().any():
            breakpoint()

        x = x * SV

        output = x.view(*input.shape[:-1], m)

        if output.isnan().any() or output.isinf().any():
            breakpoint()

        output = output.to(torch.float64)
        return output
