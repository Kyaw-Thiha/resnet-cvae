from __future__ import annotations
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms as T


_SCALE = T.Lambda(lambda x: x * 2.0 - 1.0)  # [0,1] -> [-1,1]

_TRAIN_TF = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Lambda(lambda x: x * 2.0 - 1.0)])

_TEST_TF = T.Compose([T.ToTensor(), T.Lambda(lambda x: x * 2.0 - 1.0)])


class CIFAR10Dataset(Dataset[Tuple[Tensor, Tensor]]):
    """
    CIFAR-10 RGB images (3x32x32), scaled to [-1, 1].
    """

    def __init__(self, root: str, train: bool, download: bool = True) -> None:
        super().__init__()
        self.ds = CIFAR10(root=root, train=train, download=download)
        # self.transform = T.Compose([T.ToTensor(), _SCALE])
        self.transform = _TRAIN_TF if train else _TEST_TF

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        img, label = self.ds[idx]
        x: Tensor = self.transform(img)  # shape: [3, 32, 32]
        y: Tensor = torch.tensor(label, dtype=torch.long)
        return x, y
