# ------------------------------
# Decoder
# ------------------------------

from __future__ import annotations


import torch
from torch import Tensor
import torch.nn as nn

from resnet_cvae.resblocks import ResBlockUp


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
        use_film: bool = False,
    ) -> None:
        super().__init__()
        self.use_film: bool = use_film
        self.cond_dim: int = cond_dim

        self.fc: nn.Linear = nn.Linear(z_dim + cond_dim, 7 * 7 * 64)
        self.block1: ResBlockUp = ResBlockUp(64, 64, use_film=use_film, cond_dim=cond_dim)  # 7->14
        self.block2: ResBlockUp = ResBlockUp(64, 32, use_film=use_film, cond_dim=cond_dim)  # 14->28
        self.out_conv: nn.Conv2d = nn.Conv2d(32, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, z: Tensor, e: Tensor) -> Tensor:
        """
        Args:
            z: (B, z_dim)
            e: (B, cond_dim)
        Returns:
            x_hat: (B, out_ch, 28, 28)
        """
        zc: Tensor = torch.cat([z, e], dim=1)  # (B, z_dim + cond_dim)
        h: Tensor = self.fc(zc).view(-1, 64, 7, 7)
        h = self.block1(h, e if self.use_film else None)
        h = self.block2(h, e if self.use_film else None)
        x_hat: Tensor = self.out_conv(h)
        return x_hat
