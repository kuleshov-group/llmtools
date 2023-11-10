import torch
import torch.nn as nn
import quiptools_cuda
from lib.utils import matmul_hadU_cuda, matmul_hadUt_cuda, dtype_from_str
from lib import codebook

class QuantizedLinear(nn.Module):

    def __init__(self, in_features, out_features, codesz, idx_dtype, outlier_channel_split=False, rank=-1, rescale_WH=False):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.outlier_channel_split = outlier_channel_split
        self.rank = rank
        self.rescale_WH = rescale_WH

        if self.outlier_channel_split:
            self.register_buffer('ocs_dupe_inds', torch.arange(in_features))

        if self.rank > 0:
            self.register_buffer('A', torch.zeros(out_features, rank))
            self.register_buffer('B', torch.zeros(rank, in_features))
        else:
            self.A = None
            self.B = None

        if self.rescale_WH:
            self.register_buffer("scaleWH", torch.ones(in_features))
        else:
            self.scaleWH = None
            
        self.register_buffer("Qidxs", torch.zeros(
            out_features, in_features // codesz, dtype=dtype_from_str(idx_dtype)))
        self.register_buffer("codebook_id", torch.tensor(0))
        self.register_buffer("SU", torch.ones(in_features))
        self.register_buffer("SV", torch.ones(out_features))
        self.register_buffer("Wscale", torch.ones(()))

        self.built_codebook_class = False

    def forward(self, input):
        if not self.built_codebook_class:
            self.codebook_class = codebook.get_quantized_class(
                self.codebook_id.item())(self.Qidxs.device)
            self.built_codebook_class = True

        if self.outlier_channel_split:
            input = input[..., self.ocs_dupe_inds]

        return self.codebook_class(
            input, self.Qidxs, self.SU, self.SV, self.Wscale,
            rank=self.rank, A=self.A, B=self.B,
            rescale_WH=self.rescale_WH, scaleWH=self.scaleWH)

