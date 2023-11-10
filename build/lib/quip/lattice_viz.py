import numpy as np
import torch
import matplotlib.pyplot as plt
'''
on the x-axis we have number of bits
on the y-axis we have expected average squared error of the compression scheme on a Gaussian random vector
we have the following series:
- a continuous line representing quantizing to a scaled unbounded integer grid, then applying an entropy code to get the number of bits (the line is continuous because the scale factor varies continuously)
- a continuous line representing quantizing to a scaled unbounded half-integer grid, then applying an entropy code
- points for quantizing to finite integer grids, e.g. {-1,0,1} and {-1.5,-0.5,0.5,1.5} for 2-bit, etc (here, we're not using an entropy code)
- points for lattice quantization using the A2 lattice in 2 dimensions, selecting a number of code points corresponding to various numbers of bits (e.g. 256 code points for 4-bits-per-weight)
- points for lattice quantization using the D4 lattice in 4 dimensions
- points for lattice quantization using the E8 lattice in 8 dimensions
this will make it easy to visualize the tradeoffs using different coding schemes
cts = torch.bincount(Q.view(-1))
    p = cts / cts.sum()
    ent = (p * torch.log2(1 / p)).sum()
    glog.info(f'entropy: {ent/4} bits per weight')
'''


def calc_entropy(x):
    c = 1


def quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


def quant_unbound_emp(s, n=1e3):
    x = torch.randn(int(n))
    q = torch.round(x / s) * s
    return (x - q).square().mean()


std = torch.distributions.normal.Normal(0, 1)


def quant_unbound(s, a=1e3, halfgrid=False):
    N = torch.arange(-a, a)
    if halfgrid: N += 0.5
    '''
    decompose into (a) x^2 - (b) 2xsn + (c) (sn)^2
    (a) let f(x)=x, f'(x)=1, g(x)=-exp(-x^2/2), g'(x)=x*exp(-x^2/2)
    int_a^b x^2 phi(x)dx = (1/sqrt(2*pi)) int_a^b f(x) g'(x) dx // int by parts
    = (1/sqrt(2*pi)*[f(x)g(x)|_a^b] - int_a^b (1/sqrt(2*pi))*f'(x)g(x)dx 
    (b) let y=-x^2/2, dy=-xdx by change of var
    - int_a^b x phi(x)dx = -1/sqrt(2*pi) int_y(a)^y(b) exp(y)dy
    = =1/sqrt(2*pi) exp(y)|_y(a)^y(b), y(a)=-a^2/2
    (c) CDF 
    '''
    start = s * (N - 0.5)
    end = s * (N + 0.5)

    a = 1  # just variance
    b = (2 * s * N) / np.sqrt(
        2 * np.pi) * (torch.exp(-end.square() / 2) - torch.exp(-start.square() / 2))
    c = (s * N).square() * (std.cdf(end) - std.cdf(start))

    return a + b.sum() + c.sum()


def entropy_code(s, halfgrid=False):
    a = 1e3
    N = torch.arange(-a, a)
    if halfgrid: N += 0.5
    P = std.cdf(s * (N + 0.5)) - std.cdf(s * (N - 0.5))
    assert torch.all(P >= 0)
    idx = torch.where(P > 0)
    return -(P[idx] * torch.log2(P[idx])).sum()


# def quant_unbound_int(X, b, wscale):
#     # scale = (2 * torch.max(X.max(0)[0].abs(), X.min(0)[0].abs())) / (2**b-1)
#     scale = wscale / (2**b-1)
#     return torch.round(X / scale) * scale
#
# def quant_unbound_int_half(X, b, wscale):
#     # scale = (2 * torch.max(X.max(0)[0].abs(), X.min(0)[0].abs())) / (2**b)
#     scale = wscale / (2**b)
#     return torch.round(X / scale + 0.5) * scale


def quant_bound_int(X, b, wscale):
    scale = wscale / (2**(b - 1) - 1)
    q = torch.clamp(torch.round(X / scale), -2**(b - 1) + 1, 2**(b - 1) - 1)
    return scale * q


def quant_bound_int_half(X, b, wscale):
    scale = wscale / (2**(b - 1) - 1)
    q = torch.round(X / scale + 0.5) - 0.5
    q = torch.clamp(q, -2**(b - 1) + 0.5, 2**(b - 1) - 0.5)
    return scale * q


def calc_mse(X, hatX):
    return (X - hatX).square().mean()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    wscale = 1
    b = 2
    d, n = 100, 1000
    X = torch.randn(d, n)
    assert torch.all(X.max(0)[0] > 0) and torch.all(X.min(0)[0] < 0)
    hatX2 = quant_bound_int(X, b, wscale)
    hatY2 = quant_bound_int_half(X, b, wscale)

    print(calc_mse(X, hatX2))
    print(calc_mse(X, hatY2))
