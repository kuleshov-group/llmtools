"""
builds a deep-hole-centered D4 codebook
this is a codebook consisting of points on the lattice in R4
    where each component is a half-integer
    and the components sum to an even number
from this lattice, we select the points that have a norm-squared of at most 9
this results in a codebook of 256 points distributed as follows
    8 with sorted abs of [1/2, 1/2, 1/2, 1/2]
    8                    [3/2, 3/2, 3/2, 3/2]
    4c2 * 8 = 48         [1/2, 1/2. 3/2, 3/2]
    4 * 8 = 32           [1/2, 1/2, 1/2, 3/2]
    4 * 8 = 32           [1/2, 3/2, 3/2, 3/2]
    4 * 8 = 32           [1/2, 1/2, 1/2, 5/2]
    4 * 3 * 8 = 96       [1/2, 1/2, 3/2, 5/2]
"""

import torch
from torch import nn
import quiptools_cuda

from quip.lib.utils.matmul_had import matmul_hadU_cuda, matmul_hadUt_cuda, matmul_hadU, matmul_hadUt

from quip.lib.linear.autograd import AutogradQuip

_D4_CODESZ = 4


def code3_signs(i3, x):
    if (i3 & (1 << 5)):
        x[2] *= -1
    if (i3 & (1 << 6)):
        x[1] *= -1
    if (sum(x) % 2 != 0):
        x[3] *= -1
    if (i3 & (1 << 7)):
        for j in range(_D4_CODESZ):
            x[j] *= -1
    assert (sum(x) % 2 == 0)
    return x


def code8_to_d4(i8):
    assert ((i8 >= 0) and (i8 < 256))
    i3 = i8 & (7 << 5)
    i8 = i8 & 31
    if i8 < 16:
        if i8 < 8:
            if i8 < 2:
                if i8 < 1:
                    return code3_signs(i3, [0.5] * _D4_CODESZ)
                else:
                    return code3_signs(i3, [1.5] * _D4_CODESZ)
            else:
                ibx = i8 >> 1
                if i8 & 1:
                    x = [0.5] * _D4_CODESZ
                    x[0] = 1.5
                    x[ibx] = 1.5
                else:
                    x = [1.5] * _D4_CODESZ
                    x[0] = 0.5
                    x[ibx] = 0.5
                return code3_signs(i3, x)
        else:
            ibx = (i8 & 3)
            if i8 < 8 + 4:
                x = [0.5] * _D4_CODESZ
                x[ibx] = 1.5
            else:
                x = [1.5] * _D4_CODESZ
                x[ibx] = 0.5
            return code3_signs(i3, x)
    else:
        if i8 < 16 + 4:
            ibx = (i8 & 3)
            x = [0.5] * _D4_CODESZ
            x[ibx] = 2.5
            return code3_signs(i3, x)
        else:
            ibx = i8 - 20
            ib4 = ibx & 3
            ib3 = ibx >> 2
            x = [0.5] * _D4_CODESZ
            x[ib4] = 1.5
            if (ib3 >= ib4):
                ib3 += 1
            x[ib3] = 2.5
            return code3_signs(i3, x)


def build_D4_CB():
    CB = torch.zeros(256, _D4_CODESZ)
    for i in range(256):
        x = code8_to_d4(i)
        for j in range(_D4_CODESZ):
            CB[i, j] = x[j]
    return CB


'''
def quantize(X, CB):
    scale = X.square().mean().sqrt() / 1.21
    X = X / scale
    Xqidx = (2 * X @ CB.t() - (CB @ CB.t()).diag()).argmax(1)
    return (CB[Xqidx, :] * scale, scale, Xqidx.to(torch.uint8))
def quantize_noscale_a(X, CB, A):
    Xqidx = (2 * X @ A @ CB.t() - (CB @ A @ CB.t()).diag()).argmax(1)
    return (CB[Xqidx, :], Xqidx.to(torch.uint8))
def quantize_full_lattice(X):
    Xround = (X + 0.5).round() - 0.5
    adjustParity = Xround.sum(1) % 2
    furthestEntry = (X - Xround).abs().argmax(1)
    furthestEntrySign = (X - Xround)[torch.arange(n), furthestEntry].sign()
    Xround[torch.arange(n), furthestEntry] += furthestEntrySign * adjustParity
    return Xround
'''


class D4_codebook(nn.Module):

    def __init__(self):
        super(D4_codebook, self).__init__()
        self.register_buffer("grid", build_D4_CB())
        self.register_buffer('grid_norm', (self.grid @ self.grid.T).diag())        
        self.codesz = _D4_CODESZ
        self.opt_scale = 1.21
        self.idx_dtype = torch.uint8

    def _quantize_noscale(self, X, return_idx=True):
        Xqidx = (2 * X @ self.grid.T - self.grid_norm).argmax(1)
        if return_idx:
            return self.grid[Xqidx, :], Xqidx.to(self.idx_dtype)
        return self.grid[Xqidx, :]

    def quantize(self, X, return_idx=True):
        assert X.shape[-1] == self.codesz
        return self._quantize_noscale(X, return_idx=return_idx)

    def by_idxs(self, idxs):
        return self.grid[idxs.int()]


class QuantizedD4Linear(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.D4_CB = build_D4_CB().to(device).to(torch.float16)

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

        x = input.view(-1, _D4_CODESZ * n).to(torch.float32)
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


        ## We wrap the following in a autograd function
        """
        # manifest the matrix
        W_decompressed = torch.zeros(m, n * _D4_CODESZ, dtype=torch.float16, device=x.device)
        quiptools_cuda.decompress(Qidxs, self.D4_CB, W_decompressed)
        z = x @ W_decompressed.t()
        """
        # manifest the matrix
        z = AutogradQuip.apply(x, self.D4_CB, Qidxs)


        x = z.to(torch.float32)
        x = x * (Wscale * num_scale)

        if rank > 0:
            x = x + ABx.to(torch.float32)

        #? For Debug ?#
        # import pickle
        # with open('debug/quip_overflow.pkl', 'wb') as f:
        #     pickle.dump([x], f)


        # torch.min(x): tensor(-255.6819, device='cuda:0'), no NaNs as well. torch.max(x): tensor(229.4429, device='cuda:0') #
        #x_cpu = matmul_hadU(x.cpu()) #* matmul_hadU without cuda works.
        x = matmul_hadU_cuda(x) ##TODO: Numerical Overflow. Produce -inf
        x = x * SV

        output = x.view(*input.shape[:-1], m)

        return output
