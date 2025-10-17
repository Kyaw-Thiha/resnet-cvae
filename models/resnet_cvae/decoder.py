# ------------------------------
# Decoder
# ------------------------------

from __future__ import annotations
from typing import Tuple


import torch
from torch import Tensor
import torch.nn as nn

from models.resnet_cvae.resblocks import ResBlockUp


class Decoder(nn.Module):
    """
    ResNet upsampling decoder.
    Input is concat([z, e]) then project to (7,7,64), up→14→28, then 1x1 to out_ch.
    """

    def __init__(
        self,
        out_ch: int = 1,
        z_dim: int = 16,
        cond_dim: int = 16,
        out_res: int = 32,
        use_film: bool = False,
    ) -> None:
        super().__init__()
        self.use_film: bool = use_film
        self.cond_dim: int = cond_dim

        assert out_res % 4 == 0
        self.out_start_res = out_res // 4
        # Backbone
        self.fc: nn.Linear = nn.Linear(z_dim + cond_dim, self.out_start_res * self.out_start_res * 64)
        self.block1: ResBlockUp = ResBlockUp(64, 64, use_film=use_film, cond_dim=cond_dim)  # 7->14
        self.block2: ResBlockUp = ResBlockUp(64, 32, use_film=use_film, cond_dim=cond_dim)  # 14->28

        # Heads
        # - Mean Head
        # - LogSigma Head
        self.head: nn.Conv2d = nn.Conv2d(32, out_ch, kernel_size=1, stride=1, padding=0)
        self.head_sigma: nn.Conv2d = nn.Conv2d(32, out_ch, kernel_size=1)
        if self.head_sigma.bias is not None:
            # better init for logsigma head (start around σ≈0.3 → logσ≈-1.2)
            nn.init.constant_(self.head_sigma.bias, -1.2)

    def forward(self, z: Tensor, e: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            z: (B, z_dim)
            e: (B, cond_dim)
        Returns:
            x_hat: (B, out_ch, 28, 28)
        Notes:
            Use tanh for [-1, 1]
            Use sigmoid for [0, 1]
        """
        zc: Tensor = torch.cat([z, e], dim=1)  # (B, z_dim + cond_dim)
        h: Tensor = self.fc(zc).view(-1, 64, self.out_start_res, self.out_start_res)
        h = self.block1(h, e if self.use_film else None)
        h = self.block2(h, e if self.use_film else None)

        # Heads
        x_hat: Tensor = self.head(h)
        sigma_map: Tensor = self.head_sigma(h)
        x_hat = torch.tanh(x_hat)

        return x_hat, sigma_map
