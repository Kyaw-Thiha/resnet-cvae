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

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
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
        x_hat, sigma_map = self.decoder(z, e)
        return x_hat, sigma_map, mu, logvar, z

    def loss(self, x: Tensor, y: Tensor, beta: float = 1.0) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            total_loss: scalar
            recon:      scalar (mean over batch)
            kl:         scalar (mean over batch)
        """
        x_hat, sigma_map, mu, logvar, _ = self.forward(x, y)
        sigma_map: Tensor = sigma_map.exp().clamp(1e-3, 1)
        recon_vec: Tensor = gaussian_nll(x, x_hat, sigma=sigma_map)  # (B,)
        kl_vec: Tensor = kl_normal(mu, logvar)  # (B,)

        # Mild L2 on log σ to avoid extreme values
        # tune 1e-5 ~ 1e-3
        reg = 1e-4 * (sigma_map**2).mean()

        recon: Tensor = recon_vec.mean() + reg
        kl: Tensor = kl_vec.mean()
        total: Tensor = recon + beta * kl
        return total, recon, kl

    def decode(self, z: Tensor, y: Tensor, cond_scale: float = 1.0) -> Tuple[Tensor, Tensor]:
        """
        Decode a latent code under a class label with adjustable conditioning strength.

        Args:
            z: (B, z_dim)
                Latent vectors to decode.
            y: (B,) (int labels) or (B, num_classes) (one-hot)
                Class labels or one-hot encodings that condition the decoder.
            cond_scale: float, default=1.0
                Multiplier on the class embedding before decoding.
                - 0.0  → “unconditional” path (no label influence)
                - 1.0  → normal conditioning
                - >1.0 → emphasize class features

        Returns:
            x_hat: (B, out_ch, H, W)
                Decoder mean image (for Gaussian likelihood).
        """
        if y.dim() == 1:
            num_classes: int = self.encoder.num_classes
            y_onehot: Tensor = F.one_hot(y.to(torch.long), num_classes=num_classes).float().to(z.device)
        else:
            y_onehot = y.float().to(z.device)
        e: Tensor = self.encoder.label_embed(y_onehot) * float(cond_scale)
        return self.decoder(z, e)

    @torch.no_grad()
    def sample(
        self,
        n: int,
        y: Tensor,
        device: Optional[torch.device] = None,
        *,
        temperature: float = 1.0,
        seed: Optional[int] = None,
        z: Optional[Tensor] = None,
        guidance_scale: float = 0.0,
        cond_scale: float = 1.0,
    ) -> Tensor:
        """
        Sample x ~ p(x|z,y) with z~N(0,I).
        Args:
            n:      number of samples
            y:      (n,) int labels OR (n, num_classes) one-hot
            device: device override (defaults to model device)
            temperature: float, default=1.0
                Scales the latent prior std:
                - <1.0  → truncation (cleaner, less diverse)
                - =1.0  → standard prior
                - >1.0  → more diversity/variation
            seed: RNG seed for reproducible latent sampling (ignored if `z` is provided).
            z: Optional[Tensor], shape (n, z_dim)
                Custom latent codes to decode instead of random sampling.
            guidance_scale: float, default=0.0
                Classifier-free guidance strength in output space:
                `x = x_cond + s * (x_cond - x_uncond)`. Set 0.0 to disable.
            cond_scale: float, default=1.0
                Multiplier for label-conditioning strength (passed to `decode`).
        Returns:
            x_hat:  (n, out_ch, 28, 28)

        Controllable sampling from p(x|z,y).
        - temperature: scales N(0,I) std for z (truncation when <1)
        - seed: reproducible latent draw (ignored if `z` is provided)
        - z: custom latent(s) of shape (n, z_dim) (bypasses sampling)
        - guidance_scale: CFG-style mix in output space (0 disables)
        - cond_scale: scales label embedding strength (1.0 normal, 0.0 ≈ unconditional)
        """
        dev = device or next(self.parameters()).device
        if y.dim() == 1:
            y = y.to(torch.long)
        y = y.to(dev)

        z_dim: int = self.encoder.fc_mu.out_features
        if z is None:
            if seed is not None:
                g = torch.Generator(device=dev)
                g.manual_seed(int(seed))
                z = torch.randn(n, z_dim, generator=g, device=dev)
            else:
                z = torch.randn(n, z_dim, device=dev)
            z = z * float(max(1e-6, temperature))  # temperature as latent std scaling
        else:
            assert z.shape == (n, z_dim), f"z must be shape (n,{z_dim})"
            z = z.to(dev)

        x_cond, sigma = self.decode(z, y, cond_scale=cond_scale)
        if guidance_scale <= 0.0:
            return x_cond

        # “Unconditional” path via zeroed conditioning (null label embedding)
        x_uncond, sigma = self.decode(z, y, cond_scale=0.0)
        s: float = float(guidance_scale)
        return x_cond + s * (x_cond - x_uncond)
