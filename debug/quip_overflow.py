import torch
from torch import nn
import quiptools_cuda

from quip.lib.utils.matmul_had import matmul_hadU_cuda, matmul_hadUt_cuda, matmul_hadU, matmul_hadUt
from quip.lib.utils.matmul_had_back import matmul_hadU_cuda_back

from quip.lib.linear.autograd import AutogradQuip
import pickle
import torch

breakpoint()

with open('/share/kuleshov/jy928/llmtools-2bit/debug/quip_overflow.pkl', 'rb') as fp:
    input_tensor = pickle.load(fp)

x = input_tensor[0]

x = matmul_hadU_cuda(x) ##TODO: Produce -inf


print(torch.max(x))
print(torch.min(x))
print(torch.isnan(x).any())

