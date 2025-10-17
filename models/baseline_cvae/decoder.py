from __future__ import annotations
from typing import Tuple
import torch
from torch import Tensor
import torch.nn as nn


class Decoder(nn.Module):
    """
    Simple class-conditional MLP decoder.
    - Takes [z || e] and predicts mean image
    - Uses a single learned global log_sigma, broadcast to map
    """

    def __init__(
        self,
        out_ch: int = 1,
        img_size: Tuple[int, int] = (28, 28),
        z_dim: int = 16,
        cond_dim: int = 16,
        hidden: int = 256,
        init_log_sigma: float = -1.0,  # σ ≈ 0.37
    ) -> None:
        super().__init__()
        self.out_ch = out_ch
        self.H, self.W = img_size
        self.z_dim = z_dim
        self.cond_dim = cond_dim

        flat_out = out_ch * self.H * self.W

        self.layer_1 = nn.Sequential(
            nn.Linear(z_dim + cond_dim, hidden),
            nn.ReLU(inplace=True),
        )
        self.layer_2 = nn.Sequential(
            nn.Linear(hidden + z_dim, hidden),
            nn.ReLU(inplace=True),
        )
        self.layer_3 = nn.Linear(hidden, flat_out)

        # Global noise scale (stable baseline)
        self.log_sigma = nn.Parameter(torch.tensor(float(init_log_sigma)))

    def forward(self, z: Tensor, e: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            z: (B, z_dim)
            e: (B, cond_dim)
        Returns:
            x_hat:         (B, out_ch, H, W)
            log_sigma_map: (B, out_ch, H, W)  # broadcast of scalar log_sigma
        """
        h = self.layer_1(torch.cat([z, e], dim=1))
        h = self.layer_2(torch.cat([h, z], dim=1))
        h = self.layer_3(h)

        B = z.size(0)
        x_hat = h.view(B, self.out_ch, self.H, self.W)
        log_sigma_map = self.log_sigma.expand_as(x_hat)
        return x_hat, log_sigma_map
