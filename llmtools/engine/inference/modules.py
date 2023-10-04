import math
import numpy as np
import torch
import torch.nn as nn
from llmtools.engine.inference.autograd import (
    Autograd2bit, Autograd4bit, Autograd3bit
)

try:
    import quant_cuda
except:
    print('CUDA extension not installed. Inference will not work.')

# Assumes layer is perfectly divisible into 256 * 256 blocks
class QuantLinear(nn.Module): 
    def __init__(
        self, bits, groupsize, in_features, out_features, bias, is_cuda=True
    ):
        super().__init__()
        if bits not in [2,3,4,8]:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.groupsize = groupsize if groupsize != -1 else in_features
        self.maxq = 2 ** self.bits - 1

        self.register_buffer('qweight', torch.zeros((in_features // 32 * self.bits, out_features), dtype=torch.int32))
        self.register_buffer('qzeros', torch.zeros((math.ceil(in_features / self.groupsize), out_features // 32 * self.bits), dtype=torch.int32))
        self.register_buffer('scales', torch.zeros((math.ceil(in_features / self.groupsize), out_features), dtype=torch.float16))
        self.register_buffer('g_idx', torch.tensor([i // self.groupsize  for i in range(in_features)], dtype = torch.int32))
        if bias is not None:
            self.register_buffer('bias', torch.zeros((out_features),dtype=torch.float16))
        else:
            self.bias = None
        
        # is performed by unpacking the weights and using torch.matmul
        if self.bits in [2,4,8]: 
            self.register_buffer('wf',torch.tensor(list(range(0,32,self.bits)), dtype=torch.int32).unsqueeze(0),persistent=False)
        elif self.bits == 3:
            self.register_buffer('wf', torch.tensor([[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
                                                     [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
                                                     [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0],], dtype=torch.int32).reshape(1,3,12), persistent=False)
            
        self.is_cuda = is_cuda

    def pack(self, linear, scales, zeros, g_idx = None):
        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx
        
        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone().half()
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()
            
        intweight = []
        for idx in range(self.in_features):
            intweight.append(
                torch.round(
                    (linear.weight.data[:,idx] + scale_zeros[self.g_idx[idx]]) 
                    / self.scales[self.g_idx[idx]]).to(torch.int)[:,None]
            )
        intweight = torch.cat(intweight,dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        qweight = np.zeros(
            (intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32
        )
        i = 0
        row = 0
        while row < qweight.shape[0]:
            if self.bits in [2,4,8]:
                for j in range(i, i + (32//self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32//self.bits
                row += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i))
                i += 10
                qweight[row] |= intweight[i] << 30
                row += 1
                qweight[row] |= (intweight[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= intweight[i] << 31
                row += 1
                qweight[row] |= (intweight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")
                
        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight) 
        
        zeros -= 1;
        zeros = zeros.numpy().astype(np.uint32)
        qzeros = np.zeros(
            (zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32
        )
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            if self.bits in [2,4,8]:
                for j in range(i, i + (32//self.bits)):
                    qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                i += 32//self.bits
                col += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i))
                i += 10
                qzeros[:, col] |= zeros[:, i] << 30
                col += 1
                qzeros[:, col] |= (zeros[:, i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 1)
                i += 10
                qzeros[:, col] |= zeros[:, i] << 31
                col += 1
                qzeros[:, col] |= (zeros[:, i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 2)
                i += 10
                col += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")
                
        qzeros = qzeros.astype(np.int32)
        self.qzeros = torch.from_numpy(qzeros) 

    def forward(self, x):
        if self.bits == 4:
            out = Autograd4bit.apply(
                x, 
                self.qweight, 
                self.scales, 
                self.qzeros, 
                self.g_idx, 
            )
            if self.bias is not None:
                out += self.bias
        elif self.bits == 2:
            out = Autograd2bit.apply(
                x, 
                self.qweight, 
                self.scales, 
                self.qzeros, 
                self.g_idx, 
            )
            if self.bias is not None:
                out += self.bias
        elif self.bits == 3:
            out = Autograd3bit.apply(
                x, 
                self.qweight, 
                self.scales, 
                self.qzeros, 
                self.g_idx, 
                self.wf,
                self.out_features,
            )
            if self.bias is not None:
                out += self.bias
        else:
            raise NotImplementedError()
        return out
