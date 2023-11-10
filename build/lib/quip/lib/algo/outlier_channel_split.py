import torch
from tqdm import tqdm
import math


def outlier_channel_split(W, H, mu, to_size):
    old_dim = W.shape[-1]
    remaining = to_size - old_dim

    W = torch.cat([W, torch.zeros(W.shape[0], remaining).to(W.device)], dim=-1)
    new_H = torch.zeros(to_size, to_size).to(H.device)
    new_H[0:H.shape[0], 0:H.shape[1]] = H
    H = new_H
    mu = torch.cat([mu, torch.zeros(remaining).to(mu.device)], dim=0)

    print('old drange', torch.max(W.flatten()) - torch.min(W.flatten()))
    extra_inds = []
    dupe_inds = list(range(old_dim))
    for i in tqdm(range(old_dim, to_size), desc='outlier channel splitting'):
        col = torch.argmax(W.abs()).item() % W.shape[-1]
        row = math.ceil(torch.argmax(W.abs()).item() // W.shape[-1])
        assert torch.allclose(W[row, col].abs(), torch.max(W.abs().flatten()))
        extra_inds.append(col)
        dupe_inds.append(dupe_inds[col])
        W[:, col] /= 2
        W[:, i] = W[:, col]
        H[i, 0:i] = H[col, 0:i]
        H[0:i, i] = H[0:i, col]
        H[i, i] = H[col, col]
        mu[i] = mu[col]
        i += 1

    print('new drange', torch.max(W.flatten()) - torch.min(W.flatten()))
    assert torch.allclose(H.cpu(), H.cpu().T)
    return W, H, mu, extra_inds, dupe_inds


def fuse_W(W, extra_inds):
    for i in range(len(extra_inds)):
        W[:, extra_inds[-(i + 1)]] += W[:, -(i + 1)]
    return W[:, :W.shape[-1] - len(extra_inds)]
