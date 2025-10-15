from __future__ import annotations
from typing import Tuple
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Simple class-conditional MLP encoder.
    - Flattens image
    - Concats a learned projection of one-hot labels
    - Outputs (mu, logvar) for z plus the label embedding e for reuse in the decoder
    """

    def __init__(
        self,
        in_ch: int = 1,
        img_size: Tuple[int, int] = (28, 28),
        z_dim: int = 16,
        num_classes: int = 10,
        cond_dim: int = 16,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.H, self.W = img_size
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.cond_dim = cond_dim

        flat_in = in_ch * self.H * self.W

        self.label_embed = nn.Sequential(
            nn.Linear(num_classes, cond_dim),
            nn.ReLU(inplace=True),
        )

        self.mlp = nn.Sequential(
            nn.Linear(flat_in + cond_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(hidden, z_dim)
        self.fc_logvar = nn.Linear(hidden, z_dim)

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x: (B, C, H, W)
            y: (B,) int labels OR (B, num_classes) one-hot
        Returns:
            mu:      (B, z_dim)
            logvar:  (B, z_dim)
            e:       (B, cond_dim)   # label embedding for the decoder
        """
        B = x.shape[0]
        x_flat = x.view(B, -1)

        if y.dim() == 1:
            y_oh = F.one_hot(y.to(torch.long), num_classes=self.num_classes).float().to(x.device)
        else:
            y_oh = y.float().to(x.device)

        e = self.label_embed(y_oh)  # (B, cond_dim)
        h = self.mlp(torch.cat([x_flat, e], dim=1))  # (B, hidden)
        mu = self.fc_mu(h)  # (B, z_dim)
        logvar = self.fc_logvar(h)  # (B, z_dim)
        return mu, logvar, e

    # Expose this so the wrapper can embed labels during decode/sample
    def embed_labels(self, y: Tensor) -> Tensor:
        """
        y: (B,) int labels OR (B, num_classes) one-hot
        returns e: (B, cond_dim)
        """
        if y.dim() == 1:
            y_oh = F.one_hot(y.to(torch.long), num_classes=self.num_classes).float().to(next(self.parameters()).device)
        else:
            y_oh = y.float().to(next(self.parameters()).device)
        return self.label_embed(y_oh)
