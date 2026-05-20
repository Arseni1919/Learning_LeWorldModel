import numpy as np
import torch


def signed_log(x):
    if isinstance(x, torch.Tensor):
        return x.sign() * torch.log1p(x.abs())
    return np.sign(x) * np.log1p(np.abs(x))


def signed_exp(x):
    if isinstance(x, torch.Tensor):
        return x.sign() * (torch.expm1(x.abs()))
    return np.sign(x) * np.expm1(np.abs(x))
