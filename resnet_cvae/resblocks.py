# ------------------------------
# Residual Blocks
# ------------------------------

from __future__ import annotations

from typing import Optional

from torch import Tensor
import torch.nn as nn
from model.film import FiLM


class ResBlockDown(nn.Module):
    """
    Residual block with optional downsampling (stride=2 on the first conv).
    Main: Conv3x3 -> GN -> SiLU -> Conv3x3 -> GN
    Skip: 1x1 (stride matches) or Identity
    Output: SiLU(main + skip)
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        downsample: bool = True,
        norm_groups: int = 8,
        use_film: bool = False,
        cond_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        stride: int = 2 if downsample else 1

        self.conv1: nn.Conv2d = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1
        )
        self.gn1: nn.GroupNorm = nn.GroupNorm(norm_groups, out_ch)
        self.conv2: nn.Conv2d = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, stride=1, padding=1
        )
        self.gn2: nn.GroupNorm = nn.GroupNorm(norm_groups, out_ch)

        self.skip: nn.Module
        if (in_ch != out_ch) or (stride == 2):
            self.skip = nn.Conv2d(
                in_ch, out_ch, kernel_size=1, stride=stride, padding=0
            )
        else:
            self.skip = nn.Identity()

        self.act: nn.SiLU = nn.SiLU(inplace=True)

        self.use_film: bool = use_film
        self.film1: Optional[FiLM] = (
            FiLM(cond_dim, out_ch) if (use_film and cond_dim is not None) else None
        )
        self.film2: Optional[FiLM] = (
            FiLM(cond_dim, out_ch) if (use_film and cond_dim is not None) else None
        )

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        identity: Tensor = self.skip(x)

        out: Tensor = self.conv1(x)
        out = self.gn1(out)
        out = self.act(out)
        if self.film1 is not None and cond is not None:
            out = self.film1(out, cond)

        out = self.conv2(out)
        out = self.gn2(out)
        if self.film2 is not None and cond is not None:
            out = self.film2(out, cond)

        out = self.act(out + identity)
        return out


class ResBlockUp(nn.Module):
    """
    Residual block with x2 upsampling (nearest) on both main and skip paths.
    Main: Up -> Conv3x3 -> GN -> SiLU -> Conv3x3 -> GN
    Skip: Up -> 1x1
    Output: SiLU(main + skip)
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        norm_groups: int = 8,
        use_film: bool = False,
        cond_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.up: nn.Upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv1: nn.Conv2d = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=1, padding=1
        )
        self.gn1: nn.GroupNorm = nn.GroupNorm(norm_groups, out_ch)
        self.conv2: nn.Conv2d = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, stride=1, padding=1
        )
        self.gn2: nn.GroupNorm = nn.GroupNorm(norm_groups, out_ch)

        self.skip: nn.Conv2d = nn.Conv2d(
            in_ch, out_ch, kernel_size=1, stride=1, padding=0
        )
        self.act: nn.SiLU = nn.SiLU(inplace=True)

        self.use_film: bool = use_film
        self.film1: Optional[FiLM] = (
            FiLM(cond_dim, out_ch) if (use_film and cond_dim is not None) else None
        )
        self.film2: Optional[FiLM] = (
            FiLM(cond_dim, out_ch) if (use_film and cond_dim is not None) else None
        )

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        identity: Tensor = self.up(x)
        identity = self.skip(identity)

        out: Tensor = self.up(x)
        out = self.conv1(out)
        out = self.gn1(out)
        out = self.act(out)
        if self.film1 is not None and cond is not None:
            out = self.film1(out, cond)

        out = self.conv2(out)
        out = self.gn2(out)
        if self.film2 is not None and cond is not None:
            out = self.film2(out, cond)

        out = self.act(out + identity)
        return out
