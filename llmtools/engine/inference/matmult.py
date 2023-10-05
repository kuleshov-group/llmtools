import torch
import numpy as np
try:
    import quant_cuda
except:
    print('CUDA extension not installed. Inference will not work.')

# Global Buffer
buffer_mat_dic = {}
use_new = True
auto_switch = True
auto_switch_thd = 8
debug = False
cache_buffer = True

def get_buffer(shape_of_qweight, dtype=torch.float16, device='cuda', bits=4):
    target_shape = (shape_of_qweight[0] * (32 // bits), shape_of_qweight[1])
    if not cache_buffer:
        return torch.zeros(target_shape, dtype=dtype, device=device)
    if target_shape not in buffer_mat_dic.keys():
        buffer_mat_dic[target_shape] = torch.zeros(target_shape, dtype=dtype, device=device)
    else:
        if buffer_mat_dic[target_shape].device != device:
            buffer_mat_dic[target_shape] = buffer_mat_dic[target_shape].to(device)
        if buffer_mat_dic[target_shape].dtype != dtype:
            buffer_mat_dic[target_shape] = buffer_mat_dic[target_shape].to(dtype=dtype)
    return buffer_mat_dic[target_shape]

def _matmul4bit_v1_recons(x, qweight, scales, zeros, transpose=False):
    if debug:
        print('_matmul4bit_v1_recons')
    if not transpose:
        assert qweight.shape[0] * 8 == x.shape[-1]
    else:
        assert qweight.shape[1] == x.shape[-1]
    buffer = get_buffer(qweight.shape, dtype=scales.dtype, device=qweight.device)
    quant_cuda.vecquant4recons_v1(qweight, buffer, scales, zeros)
    # dtype = x.dtype
    # x = x.float()
    if not transpose:
        output = torch.matmul(x, buffer)
    else:
        output = torch.matmul(x, buffer.T)
    # output = output.to(dtype)
    return output


def _matmul4bit_v2_recons(x, qweight, scales, zeros, g_idx, transpose=False):
    if debug:
        print('_matmul4bit_v2_recons')
    if not transpose:
        assert qweight.shape[0] * 8 == x.shape[-1]
    else:
        assert qweight.shape[1] == x.shape[-1]
    buffer = get_buffer(qweight.shape, dtype=scales.dtype, device=qweight.device)
    quant_cuda.vecquant4recons_v2(qweight, buffer, scales, zeros, g_idx)
    if not transpose:
        output = torch.matmul(x, buffer)
    else:
        output = torch.matmul(x, buffer.T)
    return output


def _matmul2bit_v2_recons(x, qweight, scales, zeros, g_idx, transpose=False):
    if debug:
        print('_matmul2bit_v2_recons')
    if not transpose:
        assert qweight.shape[0] * 16 == x.shape[-1]
    else:
        assert qweight.shape[1] == x.shape[-1]
    buffer = get_buffer(qweight.shape, dtype=scales.dtype, device=qweight.device, bits=2)
    quant_cuda.vecquant2recons_v2(qweight, buffer, scales, zeros, g_idx)
    if not transpose:
        output = torch.matmul(x, buffer)
    else:
        output = torch.matmul(x, buffer.T)
    return output


def matmul4bit(x, qweight, scales, zeros, g_idx=None):
    raise NotImplementedError()
    # detect if zeros is int32
    if zeros.dtype != torch.int32:
        # use v1
        if use_new:
            if auto_switch:
                if np.prod(x.shape[:-1]) > auto_switch_thd:
                    output = _matmul4bit_v1_recons(x.half(), qweight, scales.half(), zeros.half())
                else:
                    output = _matmul4bit_v1(x, qweight, scales, zeros)
        else:
            output = _matmul4bit_v1(x, qweight, scales, zeros)
    else:
        if g_idx is None:
            g_idx = torch.zeros(qweight.shape[0] * 8, dtype=torch.int32, device=x.device)
        # use v2
        if use_new:
            if auto_switch:
                if np.prod(x.shape[:-1]) > auto_switch_thd:
                    output = _matmul4bit_v2_recons(x.half(), qweight, scales.half(), zeros, g_idx)
                else:
                    output = _matmul4bit_v2(x, qweight, scales, zeros, g_idx)
        else:
            output = _matmul4bit_v2(x, qweight, scales, zeros, g_idx)
    return output


def matmul3bit(x, qweight, scales, zeros, g_idx, outfeatures):
    out_shape = x.shape[:-1] + (outfeatures, )
    x = x.reshape(-1,x.shape[-1])  
    output = torch.zeros((x.shape[0], outfeatures), device=x.device, dtype=torch.float32)
    quant_cuda.vecquant3matmul(x.float(), qweight, output, scales.float(), zeros, g_idx)
    output = output.reshape(out_shape)
    return output