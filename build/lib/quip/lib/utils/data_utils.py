import torch
from .matmul_had import matmul_hadU
import glog

def flat_to_sym(V, N):
    A = torch.zeros(N, N, dtype=V.dtype, device=V.device)
    idxs = torch.tril_indices(N, N, device=V.device)
    A[idxs.unbind()] = V
    A[idxs[1, :], idxs[0, :]] = V
    return A


def sym_to_flat(A):
    N = A.shape[-1]
    idxs = torch.tril_indices(N, N, device=A.device)
    return A[idxs.unbind()]


def register_H_hook(module, device):
    n = module.in_features
    H = torch.zeros(n, n, dtype=torch.float64, device=device)
    mu = torch.zeros(n, dtype=torch.float64, device=device)
    ct = 0

    def H_hook(module, x):
        nonlocal H, mu, ct, n
        x = x[0].reshape(-1, n).to(torch.float64)
        mu.add_(x.sum(dim=0))
        H.addmm_(x.T, x)
        ct += len(x)

    hook = module.register_forward_pre_hook(H_hook)

    def done():
        nonlocal H, mu, ct, hook
        hook.remove()
        return H.cpu(), mu.cpu(), ct

    return done


def block_LDL(H, b): 
    n = H.shape[0]
    assert (n % b == 0)
    m = n // b
    L = torch.linalg.cholesky(H)
    DL = torch.diagonal(L.reshape(m, b, m, b), dim1=0, dim2=2).permute(2, 0, 1)
    D = DL @ DL.permute(0, 2, 1)
    # DLinv = torch.linalg.inv(DL)
    L = L.view(n, m, b)
    for i in range(m):
        # L[:, i, :] = L[:, i, :] @ DLinv[i, :, :]
        L[:, i, :] = torch.linalg.solve(DL[i, :, :], L[:, i, :], left=False)
    L = L.reshape(n, n)
    return (L, D)


def sample_devset(dataset, tokenizer, size=128, ctx_size=2048):
    devset = torch.zeros((size, ctx_size), dtype=torch.int64)
    saved = 0
    while saved < size:
        tokens = tokenizer(dataset[torch.randint(len(dataset), (size,))]['text'],
                           return_tensors='pt',
                           truncation=True,
                           padding=True,
                           max_length=ctx_size)
        lens = tokens.attention_mask.sum(dim=-1)
        good = torch.where(lens == ctx_size)[0]
        if len(good) > 0:
            if saved + len(good) > size:
                good = good[:size - saved]
            devset[saved: saved+len(good)] = tokens.input_ids[good]
            saved += len(good)
    return devset


def load_quip(save_name, cb, args, device):
    glog.info(f"loading cached compressed layer from path \"{save_name}\"")
    dict_loaded = torch.load(save_name, map_location=torch.device('cuda', device))
    SU = dict_loaded['SU'].to(device)
    SV = dict_loaded['SV'].to(device)
    Wscale = dict_loaded['Wscale'].to(device)
    Qidxs = dict_loaded['Qidxs'].to(device)
    (m, n) = Qidxs.shape
    hatWr = cb.to(device).by_idxs(Qidxs).view(m, n * cb.codesz)
    hatWr = hatWr * Wscale
    del Wscale
    if args.lora_rank > 0:
        A = dict_loaded['A'].to(device)
        B = dict_loaded['B'].to(device)
        hatWr = hatWr + A @ B
        del A, B
    if args.incoh_mode == "had":
        hatW = (matmul_hadU((matmul_hadU(hatWr) * SU).T) * SV).T
    elif args.incoh_mode == "kron":
        hatW = SV.T @ hatWr @ SU
    else: raise NotImplementedError
    del SU, SV
    if args.rescale_WH:
        hatW = hatW / dict_loaded['scaleWH'][None, :].to(device)
    return hatW


def dtype_from_str(str):
    dtype_map = {
        'torch.int32': torch.int32,
        'torch.int16': torch.int16,
        'torch.uint8': torch.uint8,
    }
    return dtype_map[str]
