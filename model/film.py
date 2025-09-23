# ------------------------------
# FiLM (optional)
# ------------------------------
from __future__ import annotations


from torch import Tensor
import torch.nn as nn


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation:
      y = gamma(cond) * x + beta(cond), applied channel-wise.
    """

    def __init__(self, cond_dim: int, num_channels: int) -> None:
        super().__init__()
        self.num_channels: int = num_channels
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(cond_dim, num_channels * 2),
        )
        # Start near identity: gamma≈0, beta≈0 -> will be added on top of normalized features
        nn.init.zeros_(self.net[0].weight)  # type: ignore[attr-defined]
        nn.init.zeros_(self.net[0].bias)  # type: ignore[attr-defined]

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x:    (B, C, H, W)
            cond: (B, cond_dim)
        Returns:
            (B, C, H, W)
        """
        b, c, _, _ = x.shape
        gamma_beta: Tensor = self.net(cond)  # (B, 2C)
        gamma, beta = gamma_beta.chunk(2, dim=1)  # (B,C), (B,C)
        gamma = gamma.view(b, c, 1, 1)
        beta = beta.view(b, c, 1, 1)
        return gamma * x + beta
