from __future__ import annotations
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import SVHN
from torchvision import transforms as T


_SCALE = T.Lambda(lambda x: x * 2.0 - 1.0)  # [0,1] -> [-1,1]


class SVHNDataset(Dataset[Tuple[Tensor, Tensor]]):
    """
    SVHN RGB images (3xHxW), scaled to [-1, 1].
    Note: torchvision maps label '10' to '0'.
    """

    def __init__(self, root: str, split: str, download: bool = True) -> None:
        super().__init__()
        if split not in {"train", "test", "extra"}:
            raise ValueError("split must be 'train' | 'test' | 'extra'")
        self.ds = SVHN(root=root, split=split, download=download)
        self.transform = T.Compose([T.ToTensor(), _SCALE])

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        img, label = self.ds[idx]  # PIL.Image, int
        x: Tensor = self.transform(img)  # shape: [3, H, W]
        y: Tensor = torch.tensor(label, dtype=torch.long)
        return x, y
