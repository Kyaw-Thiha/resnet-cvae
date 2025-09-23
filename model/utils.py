# ------------------------------
# Utils
# ------------------------------


from __future__ import annotations

import torch
from torch import Tensor


def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    """
    Reparameterization trick: z = mu + sigma * eps,  eps ~ N(0, I)
    Args:
        mu:     (B, z_dim)
        logvar: (B, z_dim)
    Returns:
        z:      (B, z_dim)
    """
    std: Tensor = torch.exp(0.5 * logvar)
    eps: Tensor = torch.randn_like(std)
    return mu + eps * std


def kl_normal(mu: Tensor, logvar: Tensor) -> Tensor:
    """
    KL( N(mu, diag(exp(logvar))) || N(0, I) ) per-sample.
    Args:
        mu:     (B, z_dim)
        logvar: (B, z_dim)
    Returns:
        kl:     (B,)
    """
    kl: Tensor = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1)
    return kl


def gaussian_nll(x: Tensor, mean: Tensor, sigma: float = 0.1) -> Tensor:
    """
    Fixed-σ Gaussian negative log-likelihood (up to additive constant), per-sample.
    Suitable when inputs are scaled to [-1, 1].
    Args:
        x:     (B, C, H, W)
        mean:  (B, C, H, W)
        sigma: scalar std
    Returns:
        nll:   (B,)
    """
    diff: Tensor = (x - mean).view(x.size(0), -1)
    return (diff.pow(2).sum(dim=1)) / (2.0 * (sigma**2))
