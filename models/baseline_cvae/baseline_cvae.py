# ------------------------------
# Simple, self-contained class-conditional VAE (MLP baseline)
# ------------------------------
from __future__ import annotations

from typing import Tuple, Optional
import torch
from torch import Tensor
import torch.nn as nn

from models.baseline_cvae.encoder import Encoder
from models.baseline_cvae.decoder import Decoder
from models.utils import reparameterize, gaussian_nll, kl_normal


class BaselineCVAE(nn.Module):
    """
    Wrapper that wires the simple Encoder/Decoder into a class-conditional VAE.

    API:
      - forward(x,y) -> x_hat, log_sigma_map, mu, logvar, z
      - loss(x,y,beta) -> total, recon, kl
      - decode(z,y,cond_scale) -> x_hat, log_sigma_map
      - sample(n,y,device,temperature,seed,z,guidance_scale,cond_scale) -> x_hat
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        img_size: Tuple[int, int] = (28, 28),
        z_dim: int = 16,
        num_classes: int = 10,
        cond_dim: int = 16,
        hidden: int = 256,
        init_log_sigma: float = -1.0,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            in_ch=in_ch,
            img_size=img_size,
            z_dim=z_dim,
            num_classes=num_classes,
            cond_dim=cond_dim,
            hidden=hidden,
        )
        self.decoder = Decoder(
            out_ch=out_ch,
            img_size=img_size,
            z_dim=z_dim,
            cond_dim=cond_dim,
            hidden=hidden,
            init_log_sigma=init_log_sigma,
        )
        self.num_classes = num_classes
        self.z_dim = z_dim

    def forward(self, x: Tensor, y: Tensor):
        mu, logvar, e = self.encoder(x, y)
        z = reparameterize(mu, logvar)
        x_hat, log_sigma_map = self.decoder(z, e)
        return x_hat, log_sigma_map, mu, logvar, z

    def loss(self, x: Tensor, y: Tensor, beta: float = 1.0):
        x_hat, log_sigma_map, mu, logvar, _ = self.forward(x, y)

        # Keep σ in a sane range
        log_sigma_clamped = log_sigma_map.clamp(min=-3.0, max=0.5)
        sigma = log_sigma_clamped.exp()

        recon_vec = gaussian_nll(x, x_hat, sigma=sigma)  # (B,)
        kl_vec = kl_normal(mu, logvar)  # (B,)

        reg = 1e-5 * (log_sigma_clamped**2).mean()  # tiny L2 on logσ
        recon = recon_vec.mean() + reg
        kl = kl_vec.mean()
        total = recon + beta * kl
        return total, recon, kl

    def decode(self, z: Tensor, y: Tensor, cond_scale: float = 1.0):
        """
        Decode latent with adjustable conditioning strength.
        """
        e = self.encoder.embed_labels(y) * float(cond_scale)
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
        dev = device or next(self.parameters()).device
        if y.dim() == 1:
            y = y.to(torch.long)
        y = y.to(dev)

        if z is None:
            if seed is not None:
                g = torch.Generator(device=dev)
                g.manual_seed(int(seed))
                z = torch.randn(n, self.z_dim, generator=g, device=dev)
            else:
                z = torch.randn(n, self.z_dim, device=dev)
            z = z * float(max(1e-6, temperature))
        else:
            assert z.shape == (n, self.z_dim), f"z must be shape (n,{self.z_dim})"
            z = z.to(dev)

        x_cond, _ = self.decode(z, y, cond_scale=cond_scale)
        if guidance_scale <= 0.0:
            return x_cond

        x_uncond, _ = self.decode(z, y, cond_scale=0.0)
        s = float(guidance_scale)
        return x_cond + s * (x_cond - x_uncond)
