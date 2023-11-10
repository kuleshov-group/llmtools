import torch
from tqdm import tqdm
from lib import utils


def basic_preprocess(H, mu, n, args):
    if not args.remove_mean:
        H.add_(mu[None, :] * mu[:, None])
    H = utils.regularize_H(H, n, args.sigma_reg)
    return H, mu
