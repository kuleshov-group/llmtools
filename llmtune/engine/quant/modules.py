import math
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import custom_bwd, custom_fwd
from . import matmult as mm

try:
    import quant_cuda
except:
    print('CUDA extension not installed. Inference will not work.')

# Assumes layer is perfectly divisible into 256 * 256 blocks
class QuantLinear(nn.Module): 
    def __init__(self, bits, groupsize, in_features, out_features, bias, kernel_switch_threshold=128, is_cuda=True):
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
        if bias:
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
            
        self.kernel_switch_threshold = kernel_switch_threshold
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
        # if torch.is_grad_enabled():
        if True:
            out = AutogradMatmul4bit.apply(
                x, 
                self.qweight, 
                self.scales, 
                self.qzeros, 
                self.g_idx, 
                self.bits, 
                self.maxq
            )
            if self.bias:
                out += self.bias
        else:
            out = classic_forward(
                x, 
                qweight=self.qweight, 
                bias=self.bias, 
                scales=self.scales, 
                qzeros=self.qzeros, 
                g_idx=self.g_idx, 
                outfeatures=self.out_features, 
                wf=self.wf,
                bits=self.bits, 
                is_cuda=self.is_cuda, 
                kernel_switch_threshold=self.kernel_switch_threshold
            )
        return out

# ----------------------------------------------------------------------------
# helpers

# class AutogradMatmul4bit(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, qweight, scales, zeros):
#         ctx.save_for_backward(qweight, scales, zeros)
#         buff = get_buffer(
#             qweight.shape, dtype=scales.dtype, device=qweight.device
#         )
#         quant_cuda.vecquant4recons(qweight, buff, scales, zeros)
#         # dtype = x.dtype
#         # x = x.float()
#         y = torch.matmul(x, buff).clone()
#         # y = y.to(dtype)
#         return y

#     @staticmethod
#     def backward(ctx, grad_output):
#         qweight, scales, zeros = ctx.saved_tensors
#         buff = get_buffer(
#             qweight.shape, dtype=scales.dtype, device=qweight.device
#         )
#         # dtype = grad_output.dtype
#         # grad_output = grad_output.float()
#         quant_cuda.vecquant4recons(qweight, buff, scales, zeros)
#         grad = torch.matmul(grad_output, buff.T)
#         # grad = grad.to(dtype)
#         return grad, None, None, None

class AutogradMatmul4bit(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, x, qweight, scales, zeros, g_idx, bits, maxq):
        ctx.save_for_backward(qweight, scales, zeros, g_idx)
        if g_idx is None:
            output = mm._matmul4bit_v1_recons(
                x, qweight, scales, zeros
            )
        else:
            output = mm._matmul4bit_v2_recons(
                x, qweight, scales, zeros, g_idx
            )
        output = output.clone()
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        qweight, scales, zeros, g_idx = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            if g_idx is None:
                grad = mm._matmul4bit_v1_recons(
                    grad_output, qweight, scales, zeros, transpose=True
                )
            else:
                grad = mm._matmul4bit_v2_recons(
                    grad_output, qweight, scales, zeros, g_idx, transpose=True
                )
        return grad, None, None, None, None, None, None

class AutogradMatmul2bit(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, x, qweight, scales, zeros, g_idx, bits, maxq):
        ctx.save_for_backward(qweight, scales, zeros, g_idx)
        output = mm._matmul2bit_v2_recons(x, qweight, scales, zeros, g_idx)
        output = output.clone()
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        qweight, scales, zeros, g_idx = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad = mm._matmul2bit_v2_recons(
                grad_output, qweight, scales, zeros, g_idx, transpose=True
            )
        return grad, None, None, None, None, None, None        

def classic_forward(
    x, qweight, bias, scales, qzeros, g_idx, outfeatures, wf=None,
    bits=4, is_cuda=True, kernel_switch_threshold=128
):
    out_shape = x.shape[:-1] + (outfeatures, )
    x = x.reshape(-1,x.shape[-1])     
    # dtype = x.dtype
    # x = x.float()
    if  is_cuda is True and (kernel_switch_threshold is False or x.shape[0] < kernel_switch_threshold):
        out = torch.zeros((x.shape[0], outfeatures), device=x.device, dtype=torch.float32)
        if bits == 2:
            quant_cuda.vecquant2matmul(x.float(), qweight, out, scales.float(), qzeros, g_idx)
        elif bits == 3:
            quant_cuda.vecquant3matmul(x.float(), qweight, out, scales.float(), qzeros, g_idx)
        elif bits == 4:
            quant_cuda.vecquant4matmul(x.float(), qweight, out, scales.float(), qzeros, g_idx)
        elif bits == 8:
            quant_cuda.vecquant8matmul(x.float(), qweight, out, scales.float(), qzeros, g_idx)
        out = out.half()
    else:
         if bits in [2,4,8]:
            zeros = torch.bitwise_right_shift(torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits), wf.unsqueeze(0)).to(torch.int16 if self.bits == 8 else torch.int8)
            torch.bitwise_and(zeros, (2 ** bits) - 1, out=zeros)
                
            zeros = zeros + 1
            zeros = zeros.reshape(scales.shape)   
                        
            weight = torch.bitwise_right_shift(torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1), wf.unsqueeze(-1)).to(torch.int16 if bits == 8 else torch.int8)
            torch.bitwise_and(weight,(2 ** bits) - 1, out=weight)
         elif bits == 3:
            zeros = qzeros.reshape(qzeros.shape[0], qzeros.shape[1]//3, 3, 1).expand(-1, -1, -1, 12)
            zeros = (zeros >> wf.unsqueeze(0))
            zeros[:,:,0,10] = (zeros[:,:,0,10]&0x3) | ((zeros[:,:,1,0] << 2)&0x4)
            zeros[:,:,1,11] = (zeros[:,:,1,11]&0x1) | ((zeros[:,:,2,0] << 1)&0x6)
            zeros = zeros & 0x7
            zeros = torch.cat([zeros[:,:,0,:11], zeros[:,:,1,1:12], zeros[:,:,2,1:11]], dim=2)
            
            zeros = zeros + 1
            zeros = zeros.reshape(scales.shape)  
            
            weight = qweight.reshape(qweight.shape[0]//3, 3, 1, qweight.shape[1]).expand(-1, -1, 12, -1)
            weight = (weight >> wf.unsqueeze(-1))&0x7
            weight[:,0,10] = (weight[:,0,10]&0x3) | ((weight[:,1,0] << 2)&0x4)
            weight[:,1,11] = (weight[:,1,11]&0x1) | ((weight[:,2,0] << 1)&0x6)
            weight = weight & 0x7
            weight = torch.cat([weight[:,0,:11], weight[:,1,1:12], weight[:,2,1:11]], dim=1)
                
         weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
                
         weights = (scales[g_idx] * (weight - zeros[g_idx]))
         out = torch.matmul(x.half(), weights)
    out = out.reshape(out_shape)
    out = out + bias if bias is not None else out
    # out = out.to(dtype)
    return out

buffer_mat_dic = {}
def get_buffer(shape_of_qweight, dtype=torch.float16, device='cuda'):
    if shape_of_qweight not in buffer_mat_dic.keys():
        buffer_mat_dic[shape_of_qweight] = torch.zeros(
            (shape_of_qweight[0] * 8, shape_of_qweight[1]), 
            dtype=dtype, device=device
        )
    return buffer_mat_dic[shape_of_qweight]