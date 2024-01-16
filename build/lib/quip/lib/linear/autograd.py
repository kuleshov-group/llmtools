import math
import numpy as np
import torch
import torch.nn as nn

# Add Quip CUDA dependencies
from torch.cuda.amp import custom_bwd, custom_fwd

import quiptools_cuda
from quip.lib.utils.matmul_had import matmul_hadU_cuda, matmul_hadUt_cuda


_D4_CODESZ = 4

#* Custom Gradient Pass for E8 codebook*# 
class AutogradQuipE8(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x, grid_packed_abs, Qidxs, m, n):
        m_torch = torch.tensor(m, dtype=torch.int32)
        n_torch = torch.tensor(n, dtype=torch.int32)
        ctx.save_for_backward(x, grid_packed_abs, Qidxs, m_torch, n_torch) # Saves given tensors for a future call to backward().

        # * manifest the matrix #
        W_decompressed = quiptools_cuda.decompress_packed_e8p(
                Qidxs.view(m//16, n//64, 8, 4),
                grid_packed_abs
            )

        z = (x.to(torch.float32) @ W_decompressed.T.to(torch.float32)).to(torch.float32) #(x.to(torch.float16) cause overflow!
        return z

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        x, grid_packed_abs, Qidxs, m_torch, n_torch = ctx.saved_tensors # saved tensors can be accessed through the saved_tensors attribute. breakpoint()
        m = m_torch.item()
        n = n_torch.item()

        if ctx.needs_input_grad[0]:
            # * re-manifest the matrix #
            W_decompressed = quiptools_cuda.decompress_packed_e8p(
                Qidxs.view(m//16, n//64, 8, 4),
                grid_packed_abs
            )

            grad = (grad_output.to(torch.float32) @ W_decompressed.to(torch.float32)).to(torch.float32)

        return grad, None, None, None, None, None, None


#* Custom Gradient Pass for D4 codebook*# 
class AutogradQuipD4(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, x, D4_CB, Qidxs):
        ctx.save_for_backward(Qidxs, D4_CB, x) # Saves given tensors for a future call to backward().

        (m, n) = Qidxs.shape
        ## Taken from the last else block ##
        # * manifest the matrix #
        W_decompressed = torch.zeros(m, n * _D4_CODESZ, dtype=torch.float16, device=x.device)
        #quiptools_cuda.decompress(Qidxs, D4_CB, W_decompressed) # Unpack the quantized weights
        quiptools_cuda.decompress_d4(Qidxs, D4_CB, W_decompressed) # Unpack the quantized weights #? Latest QUIP Integration
        z = x @ W_decompressed.t() # * output 

        return z

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        """
        Unsure if the scaling of the input in the forward pass has an effect that needs to be reversed or accounted for in the backward pass, 
        then it would need to require applying some form of rescaling to grad_output.
        """
        Qidxs, D4_CB, x = ctx.saved_tensors # saved tensors can be accessed through the saved_tensors attribute. breakpoint()
        
        if ctx.needs_input_grad[0]:
            (m, n) = Qidxs.shape
            # * re-manifest the matrix #
            W_decompressed = torch.zeros(m, n * _D4_CODESZ, dtype=torch.float16, device=grad_output.device)

            #quiptools_cuda.decompress(Qidxs, D4_CB, W_decompressed) # Unpack the quantized weights
            quiptools_cuda.decompress_d4(Qidxs, D4_CB, W_decompressed) # Unpack the quantized weights #? Latest QUIP Integration
            grad = grad_output @ W_decompressed


        return grad, None, None, None, None, None, None
    


class AutogradOrthoMult(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x, transpose, had_left, had_right, K_left, K_right):
        K_left_torch = torch.tensor(K_left, dtype=torch.int32)
        K_right_torch = torch.tensor(K_right, dtype=torch.int32)
        ctx.save_for_backward(transpose, had_left, had_right, K_left_torch, K_right_torch) # Saves given tensors for a future call to backward().

        #* Tranpose is either a zero tensor or ones tensor *#
        if torch.is_nonzero(transpose.any()):
            output = matmul_hadUt_cuda(x, had_left, K_left)
            y = matmul_hadUt_cuda(x, had_left, K_left)
        else:
            output = matmul_hadU_cuda(x, had_right, K_right)
            y = matmul_hadU_cuda(x, had_right, K_right)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        # breakpoint()
        transpose, had_left, had_right, K_left_torch, K_right_torch = ctx.saved_tensors # saved tensors can be accessed through the saved_tensors attribute. breakpoint()
        K_right = K_right_torch.item()
        K_left = K_left_torch.item()

        if ctx.needs_input_grad[0]:
            if torch.is_nonzero(transpose.any()):
                grad = matmul_hadU_cuda(grad_output, had_left, K_left)
            else:
                grad = matmul_hadUt_cuda(grad_output, had_right, K_right)


        return grad, None, None, None, None, None, None
    