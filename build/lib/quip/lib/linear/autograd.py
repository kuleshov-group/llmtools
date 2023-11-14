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
    def forward(ctx, input, D4_CB, Qidxs):
        # breakpoint()

        ctx.save_for_backward(Qidxs, D4_CB) # Saves given tensors for a future call to backward().

        (m, n) = Qidxs.shape

        ## Taken from the last else block ##
        # * manifest the matrix #
        W_decompressed = torch.zeros(m, n * _D4_CODESZ, dtype=torch.float16, device=input.device)
        
        quiptools_cuda.decompress(Qidxs, D4_CB, W_decompressed) # Unpack the quantized weights
        z = input.float() @ W_decompressed.t().float() # * output 


        return z


    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        """
        Unsure if the scaling of the input in the forward pass has an effect that needs to be reversed or accounted for in the backward pass, 
        then it would need to require applying some form of rescaling to grad_output.
        """
        # breakpoint()
        Qidxs, D4_CB = ctx.saved_tensors # saved tensors can be accessed through the saved_tensors attribute.

        if ctx.needs_input_grad[0]:
            (m, n) = Qidxs.shape

            # * re-manifest the matrix #
            W_decompressed = torch.zeros(m, n * _D4_CODESZ, dtype=torch.float16, device=grad_output.device)
            quiptools_cuda.decompress(Qidxs, D4_CB, W_decompressed) # Unpack the quantized weights
            grad = grad_output.float() @ W_decompressed.float()


        return grad, None, None, None, None, None, None
    

