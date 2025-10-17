# ------------------------------
# Utils
# ------------------------------


from __future__ import annotations
import math

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


def gaussian_nll(
    x: Tensor,
    mean: Tensor,
    sigma: Tensor,
) -> Tensor:
    """
    Heteroscedastic Gaussian negative log-likelihood per sample.

    Args:
        x:     (B, C, H, W)
        mean:  (B, C, H, W)
        sigma: broadcastable to (B, C, H, W); can be a scalar, (C,), (1,C,1,1), or (B,C,H,W)
    Keyword Args:
        clamp_min: lower bound for sigma to avoid div-by-zero
        clamp_max: upper bound for sigma to avoid exploding variance early

    Returns:
        nll:   (B,)   # mean over all pixels & channels, per sample
    """

    # Full NLL: 0.5 * [ (x-μ)^2 / σ^2 + 2 log σ + log(2π) ]
    # Summed over pixels/channels, yielding (B,)
    var = sigma * sigma
    sq = (x - mean) ** 2
    nll = 0.5 * (sq / var + 2.0 * torch.log(sigma) + math.log(2.0 * math.pi))
    return nll.flatten(1).mean(dim=1)
