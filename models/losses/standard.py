from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor

from models.losses.base import BaseLossModule, LossContext, LossInputs, LossOutput, ensure_tensor
from models.utils import gaussian_nll, kl_normal


@dataclass
class StandardLossConfig:
    """
    Configuration for the classic β-VAE style objective.

    reconstruction_weight:
        Multiplier on the reconstruction term. Kept for completeness even though it
        defaults to 1.0.
    kl_weight:
        Static multiplier on the KL term before the scheduled `beta` factor.
    log_sigma_bounds:
        Clamp range (min, max) applied to the decoder's predicted log-σ map to keep
        the Gaussian likelihood numerically stable.
    log_sigma_l2_weight:
        Weight for a mild L2 regulariser on the log-σ map (pre-clamp) to avoid
        extreme values.
    """

    reconstruction_weight: float = 1.0
    kl_weight: float = 1.0
    log_sigma_bounds: Tuple[float, float] = (-3.0, 0.5)
    log_sigma_l2_weight: float = 1e-4


class StandardLoss(BaseLossModule):
    """
    Standard β-VAE loss. Matches the previous implementation but wrapped in a class
    so we can swap objectives without touching the training loop.
    """

    def __init__(self, config: StandardLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or StandardLossConfig()

    def __call__(self, inputs: LossInputs, context: LossContext) -> LossOutput:
        cfg = self.config
        log_sigma_min, log_sigma_max = cfg.log_sigma_bounds

        log_sigma_clamped = inputs.log_sigma.clamp(min=log_sigma_min, max=log_sigma_max)
        sigma = log_sigma_clamped.exp()

        recon_vector = gaussian_nll(inputs.x, inputs.x_hat, sigma=sigma)
        recon = recon_vector.mean() * float(cfg.reconstruction_weight)

        if cfg.log_sigma_l2_weight > 0:
            recon = recon + float(cfg.log_sigma_l2_weight) * (log_sigma_clamped**2).mean()

        kl_vector = kl_normal(inputs.mu, inputs.logvar)
        kl = kl_vector.mean()

        scaled_kl = kl * float(cfg.kl_weight) * float(context.beta)
        total = recon + scaled_kl

        metrics: Dict[str, Tensor] = {
            "beta": ensure_tensor(context.beta),
            "kl_scaled": scaled_kl.detach(),
        }
        return LossOutput(total=total, reconstruction=recon, kl=kl, metrics=metrics)

