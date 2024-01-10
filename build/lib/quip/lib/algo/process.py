import torch
from tqdm import tqdm
from quip.lib import lattice, utils


def preprocess(H, mu, args):
    if not args.remove_mean:
        H.add_(mu[None, :] * mu[:, None])
    H = utils.regularize_H(H, n, args.sigma_reg)
