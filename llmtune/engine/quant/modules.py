import numpy as np
import torch
import torch.nn as nn

try:
    import quant_cuda
except:
    print('CUDA extension not installed. Inference will not work.')

# Assumes layer is perfectly divisible into 256 * 256 blocks
class QuantLinear(nn.Module): 
    def __init__(self, bits, in_features, out_features):
        super().__init__()
        if bits not in [2,3,4,8]:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        self.bits = bits
        self.register_buffer('zeros', torch.zeros((out_features, 1)))
        self.register_buffer('scales', torch.zeros((out_features, 1)))
        self.register_buffer('bias', torch.zeros(out_features))
        self.register_buffer(
            'qweight', torch.zeros(
                (in_features // 256 * (bits * 8), out_features), 
                dtype=torch.int
            )
        )
        self.in_features = in_features
        self.out_features = out_features

    def pack(self, linear, scales, zeros):
        self.zeros = zeros * scales
        self.scales = scales.clone()
        if linear.bias is not None:
            self.bias = linear.bias.clone()  

        intweight = torch.round((linear.weight.data + self.zeros) / self.scales).to(torch.int)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        qweight = np.zeros(
            (intweight.shape[0] // 256 * (self.bits * 8), intweight.shape[1]), dtype=np.uint32
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

    def forward(self, x):
        # if torch.is_grad_enabled():
        if True:
            out = AutogradMatmul4bit.apply(
                x, self.qweight, self.scales, self.zeros
            )
            out += self.bias
        else:
            out = classic_forward(
                x, self.qweight, self.bias, self.scales, self.zeros, self.bits
            )
        return out

# ----------------------------------------------------------------------------
# helpers

class AutogradMatmul4bit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, qweight, scales, zeros):
        ctx.save_for_backward(qweight, scales, zeros)
        buff = get_buffer(
            qweight.shape, dtype=scales.dtype, device=qweight.device
        )
        quant_cuda.vecquant4recons(qweight, buff, scales, zeros)
        # dtype = x.dtype
        # x = x.float()
        y = torch.matmul(x, buff).clone()
        # y = y.to(dtype)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        qweight, scales, zeros = ctx.saved_tensors
        buff = get_buffer(
            qweight.shape, dtype=scales.dtype, device=qweight.device
        )
        # dtype = grad_output.dtype
        # grad_output = grad_output.float()
        quant_cuda.vecquant4recons(qweight, buff, scales, zeros)
        grad = torch.matmul(grad_output, buff.T)
        # grad = grad.to(dtype)
        return grad, None, None, None

def classic_forward(x, qweight, bias, y, scales, zeros, bits=4):
    outshape = list(x.shape)
    x = x.reshape(-1, x.shape[-1])
    y = bias.clone().repeat(x.shape[0],1)
    outshape[-1] = bias.numel()
    dtype = x.dtype
    x = x.float()
    if bits == 2:
        quant_cuda.vecquant2matmul(x, qweight, y, scales, zeros)
    elif bits == 3:
        quant_cuda.vecquant3matmul(x, qweight, y, scales, zeros)
    elif bits == 4:
        quant_cuda.vecquant4matmul(x, qweight, y, scales, zeros)
    elif bits == 8:
        quant_cuda.vecquant8matmul(x, qweight, y, scales, zeros)
    else:
        raise NotImplementedError("Only 2,3,4,8 bits are supported.")
    y = y.to(dtype)
    return y.reshape(outshape)

buffer_mat_dic = {}
def get_buffer(shape_of_qweight, dtype=torch.float16, device='cuda'):
    if shape_of_qweight not in buffer_mat_dic.keys():
        buffer_mat_dic[shape_of_qweight] = torch.zeros(
            (shape_of_qweight[0] * 8, shape_of_qweight[1]), 
            dtype=dtype, device=device
        )
    return buffer_mat_dic[shape_of_qweight]