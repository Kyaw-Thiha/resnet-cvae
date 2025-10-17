from __future__ import annotations


from typing import Callable, Optional, Tuple


import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision import transforms as T


class MNISTDataset(Dataset[Tuple[Tensor, Tensor]]):
    """
    Wraps torchvision MNIST and applies scaling to [-1, 1] for Gaussian likelihood.
    Returns (image, label).
    """

    def __init__(
        self,
        root: str,
        train: bool,
        download: bool = True,
        image_channels: int = 1,
        transform: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        super().__init__()
        self.ds: MNIST = MNIST(root=root, train=train, download=download)
        self.image_channels: int = image_channels

        # Default transform: ToTensor -> scale to [-1, 1]
        self.transform: Callable[[Tensor], Tensor]
        if transform is None:
            self.transform = T.Compose(
                [
                    T.ToTensor(),  # [0,1]
                    T.Lambda(lambda x: x * 2.0 - 1.0),  # [-1,1]
                ]
            )
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        img, label = self.ds[idx]
        x: Tensor = self.transform(img)
        if self.image_channels == 3 and x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        y: Tensor = torch.tensor(label, dtype=torch.long)
        return x, y
