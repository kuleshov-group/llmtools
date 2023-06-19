import math
import numpy as np
import torch
import torch.nn as nn
from . import matmult as mm
from torch.cuda.amp import custom_bwd, custom_fwd

class Autograd4bit(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, x, qweight, scales, zeros, g_idx):
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

class Autograd2bit(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, x, qweight, scales, zeros, g_idx):
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

class Autograd3bit(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, x, qweight, scales, zeros, g_idx, wf):
        ctx.save_for_backward(qweight, scales, zeros, g_idx, wf)
        weight = unpack_weight(qweight, scales, qzeros, g_idx, wf, bits=3)
        output = torch.matmul(x.half(), weight)
        # output.reshape(x.shape[:-1] + (outfeatures,))
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        qweight, scales, zeros, g_idx, wf = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            weight = unpack_weight(qweight, scales, qzeros, g_idx, wf, bits=3)
            grad = torch.matmul(x.half(), weight.T)
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
        weight = unpack_weight(qweight, scales, qzeros, g_idx, wf, bits)
        out = torch.matmul(x.half(), weight)
        del weight

    out = out.reshape(out_shape)
    out = out + bias if bias is not None else out
    # out = out.to(dtype)
    return out

def unpack_weight(qweight, scales, qzeros, g_idx, wf=None, bits=4):
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

       # allocate buffers
       buffer1_shape = (qweight.shape[0]//3, 3, 1, qweight.shape[1])
       weight1 = get_buffer(buffer1_shape)
       
       weight = qweight.reshape(qweight.shape[0]//3, 3, 1, qweight.shape[1]).expand(-1, -1, 12, -1)
       # print(qweight.data_ptr(), qweight.requires_grad)
       # print(weight.data_ptr(), weight.requires_grad)
       weight2 = torch.zeros(weight.shape)
       weight = (weight >> wf.unsqueeze(-1))&0x7
       # print(weight.data_ptr(), weight.requires_grad)
       weight[:,0,10] = (weight[:,0,10]&0x3) | ((weight[:,1,0] << 2)&0x4)
       # print(weight.data_ptr(), weight.requires_grad)
       weight[:,1,11] = (weight[:,1,11]&0x1) | ((weight[:,2,0] << 1)&0x6)
       # print(weight.data_ptr(), weight.requires_grad)
       weight &= 0x7
       # print(weight.data_ptr(), weight.requires_grad)
       weight = torch.cat([weight[:,0,:11], weight[:,1,1:12], weight[:,2,1:11]], dim=1)
       # print(weight.data_ptr(), weight.requires_grad)
           
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
    # print(weight.data_ptr(), weight.requires_grad)
           
    g_idx_long = g_idx.to(torch.long)
    # weights = (scales[g_idx_long] * (weight - zeros[g_idx_long]))
    # out = torch.matmul(x.half(), weights)
    # print(weight.dtype)
    # print(zeros.dtype)
    # print(scales.dtype)
    weight -= zeros[g_idx_long]
    # print(weight.data_ptr(), weight.requires_grad)
    weight = weight.to(torch.half)
    # print(weight.data_ptr(), weight.requires_grad)
    weight *= scales[g_idx_long]
    # print(weight.data_ptr(), weight.requires_grad)
    return weight

# ----------------------------------------------------------------------------
# helpers

buffer_mat_dic = {}
def get_buffer(shape_of_qweight, dtype=torch.float16, device='cuda'):
    if shape_of_qweight not in buffer_mat_dic.keys():
        buffer_mat_dic[shape_of_qweight] = torch.zeros(
            (shape_of_qweight[0] * 8, shape_of_qweight[1]), 
            dtype=dtype, device=device
        )
    return buffer_mat_dic[shape_of_qweight]

buffer_mat_dic1 = {}
def get_buffer1(buffer_shape, dtype=torch.int32, device='cuda'):
    if buffer_shape not in buffer_mat_dic1.keys():
        buffer_mat_dic1[buffer_shape] = torch.zeros(
            buffer_shape, dtype=dtype, device=device
        )
    return buffer_mat_dic1[buffer_shape]    

buffer_mat_dic2 = {}
def get_buffer2(buffer_shape, dtype=torch.float16, device='cuda'):
    if buffer_shape not in buffer_mat_dic2.keys():
        buffer_mat_dic2[buffer_shape] = torch.zeros(
            buffer_shape, dtype=dtype, device=device
        )
    return buffer_mat_dic2[buffer_shape]        