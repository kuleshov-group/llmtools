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
    def forward(ctx, input, D4_CB, Qidxs, SU, SV, Wscale, rank=-1, A=None, B=None, rescale_WH=False, scaleWH=None):
        # ctx.save_for_backward(SU, SV, Qidxs, Wscale) # Saves given tensors for a future call to backward().
        ctx.save_for_backward(SU, SV, Qidxs, Wscale, D4_CB, A, B, rank, rescale_WH, scaleWH) # Saves given tensors for a future call to backward().

        (m, n) = Qidxs.shape
        x = input.view(-1, _D4_CODESZ * n).to(torch.float32)
        if rescale_WH:
            x /= scaleWH
        x = x * SU
        x = matmul_hadUt_cuda(x)

        if rank > 0:
            Bx = x @ B.t().to(torch.float32)
            ABx = Bx @ A.t().to(torch.float32)

        num_scale = 1024
        x = x / num_scale
        x = x.to(torch.float16)

        ## Taken from the last else block ##
        # manifest the matrix
        W_decompressed = torch.zeros(m, n * _D4_CODESZ, dtype=torch.float16, device=x.device)
        
        quiptools_cuda.decompress(Qidxs, D4_CB, W_decompressed) # Unpack the quantized weights
        z = x @ W_decompressed.t()

        x = z.to(torch.float32)
        x = x * (Wscale * num_scale) # Scale the weights back up with Wscale and Num_scale

        if rank > 0:
            x = x + ABx.to(torch.float32)

        x = matmul_hadU_cuda(x)
        x = x * SV

        output = x.view(*input.shape[:-1], m)

        return output


    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        """
        Unsure if the scaling of the input in the forward pass has an effect that needs to be reversed or accounted for in the backward pass, 
        then it would need to require applying some form of rescaling to grad_output.
        """
        SU, SV, Qidxs, Wscale, D4_CB, A, B, rank, rescale_WH, scaleWH = ctx.saved_tensors # saved tensors can be accessed through the saved_tensors attribute.

        if ctx.needs_input_grad[0]:
            (m, n) = Qidxs.shape

            # Question: Do I need to scale grad_output based on scaleWH and num_scale?
            grad = grad_output.view(-1, _D4_CODESZ * n).to(torch.float32)
            if rescale_WH:
                grad /= scaleWH

            grad = grad * SU # Transform the grad by timing SU?
            grad = matmul_hadUt_cuda(grad)


            if rank > 0:
                Bgrad = grad @ B.to(torch.float32)
                ABgrad = Bgrad @ A.to(torch.float32)

            num_scale = 1024
            grad = grad / num_scale
            grad = grad.to(torch.float16)

            # manifest the matrix
            W_decompressed = torch.zeros(m, n * _D4_CODESZ, dtype=torch.float16, device=grad.device)
            quiptools_cuda.decompress(Qidxs, D4_CB, W_decompressed) # Unpack the quantized weights
            grad_z = grad @ W_decompressed

            grad = grad_z.to(torch.float32)
            grad = grad * (Wscale * num_scale)


            if rank > 0:
                grad = grad + ABgrad.to(torch.float32)
            
            grad = matmul_hadU_cuda(grad)
            grad = grad * SV

            grad_input = grad.view(*grad_output.shape[:-1], m)

        return grad_input, None, None, None, None, None, None
    

