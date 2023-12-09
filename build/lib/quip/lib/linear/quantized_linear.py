import torch
import torch.nn as nn
import quiptools_cuda
from quip.lib.utils import matmul_hadU_cuda, matmul_hadUt_cuda, dtype_from_str
from quip.lib import codebook

class MultiLayerModel(nn.Module):
    def __init__(self, input_features=1024, output_features=512, hidden_features1=512, hidden_features2=256):
        super(MultiLayerModel, self).__init__()
        # Define the first linear layer
        self.linear1 = nn.Linear(input_features, hidden_features1)
        # Define the second linear layer
        self.linear2 = nn.Linear(hidden_features1, hidden_features2)
        # Define the third linear layer
        self.linear3 = nn.Linear(hidden_features2, output_features)
    
    def forward(self, x):
        # Forward pass through the first layer
        x = self.linear1(x)
        # Apply an activation function, e.g., ReLU
        x = torch.relu(x)

        num_scale = 1024
        x = x / num_scale
        # Forward pass through the second layer
        x = self.linear2(x)
        # Apply an activation function
        x = torch.relu(x)
        # Forward pass through the third layer
        x = self.linear3(x)
        return x



class QuantizedLinear(nn.Module):

    def __init__(self, in_features, out_features, codesz, idx_dtype, outlier_channel_split=False, rank=-1, rescale_WH=False, Qidxs=None, D4_CB=None):
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
            
        self.register_buffer("Qidxs", torch.ones(
            out_features, in_features // codesz, dtype=dtype_from_str(idx_dtype)))
        self.register_buffer("codebook_id", torch.tensor(0))
        self.register_buffer("SU", torch.ones(in_features))
        self.register_buffer("SV", torch.ones(out_features))
        self.register_buffer("Wscale", torch.ones(()))

        self.built_codebook_class = False

        #* QUIP-LLMTools Integration *#
        self.codesz = codesz
        self.idx_dtype = idx_dtype

        if Qidxs is not None:
            self.Qidxs = Qidxs
        if D4_CB is not None:
            self.D4_CB = D4_CB

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

