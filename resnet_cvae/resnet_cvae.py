# ------------------------------
# CVAE wrapper
# ------------------------------

from __future__ import annotations

from typing import Tuple, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from resnet_cvae.encoder import Encoder
from resnet_cvae.decoder import Decoder
from resnet_cvae.utils import reparameterize, gaussian_nll, kl_normal


class ResNetCVAE(nn.Module):
    """
    Full class-conditional VAE with Gaussian likelihood:
      Encoder(x,y) -> mu, logvar, e
      z = reparameterize(mu, logvar)
      Decoder(z, e) -> mean image x_hat (R)
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        z_dim: int = 16,
        num_classes: int = 10,
        cond_dim: int = 16,
        use_film: bool = False,
        sigma: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder: Encoder = Encoder(
            in_ch=in_ch,
            z_dim=z_dim,
            num_classes=num_classes,
            cond_dim=cond_dim,
            use_film=use_film,
        )
        self.decoder: Decoder = Decoder(
            out_ch=out_ch,
            z_dim=z_dim,
            cond_dim=cond_dim,
            use_film=use_film,
        )
        self.sigma: float = float(sigma)  # fixed std for Gaussian likelihood

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            x: (B, C, H, W)
            y: (B,) int labels OR (B, num_classes) one-hot
        Returns:
            x_hat: (B, out_ch, H, W)
            mu:     (B, z_dim)
            logvar: (B, z_dim)
            z:      (B, z_dim)
        """
        mu, logvar, e = self.encoder(x, y)
        z: Tensor = reparameterize(mu, logvar)
        x_hat: Tensor = self.decoder(z, e)
        return x_hat, mu, logvar, z

    def loss(
        self, x: Tensor, y: Tensor, beta: float = 1.0
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            total_loss: scalar
            recon:      scalar (mean over batch)
            kl:         scalar (mean over batch)
        """
        x_hat, mu, logvar, _ = self.forward(x, y)
        recon_vec: Tensor = gaussian_nll(x, x_hat, sigma=self.sigma)  # (B,)
        kl_vec: Tensor = kl_normal(mu, logvar)  # (B,)

        recon: Tensor = recon_vec.mean()
        kl: Tensor = kl_vec.mean()
        total: Tensor = recon + beta * kl
        return total, recon, kl

    @torch.no_grad()
    def sample(
        self, n: int, y: Tensor, device: Optional[torch.device] = None
    ) -> Tensor:
        """
        Sample x ~ p(x|z,y) with z~N(0,I).
        Args:
            n:      number of samples
            y:      (n,) int labels OR (n, num_classes) one-hot
            device: device override (defaults to model device)
        Returns:
            x_hat:  (n, out_ch, 28, 28)
        """
        dev: torch.device = device or next(self.parameters()).device
        if y.dim() == 1:
            num_classes: int = self.encoder.num_classes
            y_onehot: Tensor = (
                F.one_hot(y.to(torch.long), num_classes=num_classes).float().to(dev)
            )
        else:
            y_onehot = y.float().to(dev)

        e: Tensor = self.encoder.label_embed(y_onehot)
        z_dim: int = self.encoder.fc_mu.out_features
        z: Tensor = torch.randn(n, z_dim, device=dev)
        x_hat: Tensor = self.decoder(z, e)
        return x_hat
