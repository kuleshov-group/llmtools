import primefac
import scipy
import math


def butterfly_factors(n):
    pf = list(primefac.primefac(n))
    return (math.prod(pf[0::2]), math.prod(pf[1::2]))

def gen_rand_orthos(m,p):
    if (p != 2):
        return torch.tensor(scipy.stats.special_ortho_group.rvs(p, size=m)).to(torch.float32)
    X = torch.zeros(m,2,2)
    t = torch.rand(m) * (2 * math.pi) 
    sin_t = torch.sin(t)
    cos_t = torch.cos(t)
    X[:,0,0] = cos_t
    X[:,1,1] = cos_t
    X[:,0,1] = sin_t
    X[:,1,0] = -sin_t
    return X

# generates a random orthogonal butterfly matrix of dimension n, without blocking
def gen_rand_ortho_butterfly_noblock(n):
    return ([gen_rand_orthos(1, p) for p in butterfly_factors(n)], torch.randperm(n), torch.randperm(n))

# multiply by a random orthogonal butterfly matrix
def mul_ortho_butterfly(Bpp, x):
    (B, p_in, p_out) = Bpp
    assert((len(x.shape) == 1) or (len(x.shape) == 2))
    orig_dim = 2
    if (len(x.shape) == 1):
        (n,) = x.shape
        x = x.reshape(n,1)
        orig_dim = 1
    (n,q) = x.shape
    x = x[p_in,:]
    pfn = tuple(butterfly_factors(n))
    for i in range(len(pfn)):
        mpfx = math.prod(pfn[0:i])
        p = pfn[i]
        msfx = math.prod(pfn[(i+1):])
        x = x.reshape(mpfx, p, msfx, q).permute(0,2,1,3).reshape(mpfx * msfx, p, q)
        x = B[i] @ x
        x = x.reshape(mpfx, msfx, p, q).permute(0,2,1,3).reshape(n,q)
    x = x[p_out,:]
    if (orig_dim == 1):
        x = x.reshape(n)
    return x

# generates a random orthogonal butterfly matrix of dimension n
# and converts it to a dense matrix
def rand_ortho_butterfly_noblock(n):
    return mul_ortho_butterfly(gen_rand_ortho_butterfly_noblock(n), torch.eye(n))