# ------------------------------
# Encoder
# ------------------------------

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from models.resnet_cvae.resblocks import ResBlockDown


class Encoder(nn.Module):
    """
    ResNet encoder for images (default MNIST 1x28x28).
    Outputs (mu, logvar, e) where e is the class embedding.
    """

    def __init__(
        self,
        in_ch: int = 1,
        z_dim: int = 16,
        num_classes: int = 10,
        cond_dim: int = 16,
        use_film: bool = False,
    ) -> None:
        super().__init__()
        self.use_film: bool = use_film
        self.cond_dim: int = cond_dim
        self.num_classes: int = num_classes

        # class embedding from one-hot
        self.label_embed: nn.Sequential = nn.Sequential(
            nn.Linear(num_classes, cond_dim),
            nn.SiLU(inplace=True),
        )

        # Stem
        self.stem: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
        )

        # ResNet Down: 28->14->7
        self.block1: ResBlockDown = ResBlockDown(32, 32, downsample=True, use_film=use_film, cond_dim=cond_dim)
        self.block2: ResBlockDown = ResBlockDown(32, 64, downsample=True, use_film=use_film, cond_dim=cond_dim)
        self.block3: ResBlockDown = ResBlockDown(64, 64, downsample=False, use_film=use_film, cond_dim=cond_dim)

        self.avgpool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1, 1))  # -> (B,64,1,1)
        self.fc_mu: nn.Linear = nn.Linear(64 + cond_dim, z_dim)
        self.fc_logvar: nn.Linear = nn.Linear(64 + cond_dim, z_dim)

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x: (B, C, H, W)
            y: (B,) int labels OR (B, num_classes) one-hot
        Returns:
            mu:     (B, z_dim)
            logvar: (B, z_dim)
            e:      (B, cond_dim) class embedding
        """
        if y.dim() == 1:
            y_onehot: Tensor = F.one_hot(y.to(torch.long), num_classes=self.num_classes).float()
        else:
            y_onehot = y.float()

        e: Tensor = self.label_embed(y_onehot)  # (B, cond_dim)

        h: Tensor = self.stem(x)
        h = self.block1(h, e if self.use_film else None)
        h = self.block2(h, e if self.use_film else None)
        h = self.block3(h, e if self.use_film else None)

        h = self.avgpool(h).squeeze(-1).squeeze(-1)  # (B, 64)
        hc: Tensor = torch.cat([h, e], dim=1)  # (B, 64 + cond_dim)

        mu: Tensor = self.fc_mu(hc)
        logvar: Tensor = self.fc_logvar(hc)
        return mu, logvar, e
