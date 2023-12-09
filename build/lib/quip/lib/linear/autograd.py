import math
import numpy as np
import torch
import torch.nn as nn

# Add Quip CUDA dependencies
from torch.cuda.amp import custom_bwd, custom_fwd

import quiptools_cuda
from quip.lib.utils.matmul_had import matmul_hadU_cuda, matmul_hadUt_cuda


_D4_CODESZ = 4

class AutogradQuip(torch.autograd.Function):
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
    def forward(ctx, x, transpose):
        ctx.save_for_backward(transpose) # Saves given tensors for a future call to backward().

        #* Tranpose is either a zero tensor or ones tensor *#
        #breakpoint()
        if torch.is_nonzero(transpose.any()):
            x = matmul_hadUt_cuda(x)
        else:
            x = matmul_hadU_cuda(x)
        return x

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        """
        """
        transpose = ctx.saved_tensors[0] # saved tensors can be accessed through the saved_tensors attribute. breakpoint()
        
        if ctx.needs_input_grad[0]:
            if torch.is_nonzero(transpose.any()):
                grad = matmul_hadU_cuda(grad_output)
            else:
                grad = matmul_hadUt_cuda(grad_output)


        return grad, None, None, None, None, None, None
    