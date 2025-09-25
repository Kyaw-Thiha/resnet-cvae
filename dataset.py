from __future__ import annotations


from typing import Callable, Optional, Sequence, Tuple


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


# class PredictLabelDataset(Dataset[Tensor]):
#     """A simple dataset that yields labels for generation (predict loop)."""
#
#     def __init__(self, num_classes: int = 10, samples_per_class: int = 8) -> None:
#         super().__init__()
#         self.num_classes: int = num_classes
#         self.samples_per_class: int = samples_per_class
#         self._labels: torch.Tensor = torch.arange(num_classes).repeat_interleave(samples_per_class)
#
#     def __len__(self) -> int:
#         return int(self._labels.numel())
#
#     def __getitem__(self, idx: int) -> Tensor:
#         return self._labels[idx]


class PredictLabelDataset(Dataset[torch.Tensor]):
    """
    Yields class labels for generation.
    - classes: sequence of class ids to include (default: range(num_classes))
    - samples_per_class: how many items for each class
    """

    def __init__(
        self,
        num_classes: int = 10,
        samples_per_class: int = 8,
        classes: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        if classes is None:
            classes = list(range(int(num_classes)))
        self.classes = list(map(int, classes))
        self.samples_per_class = int(samples_per_class)

        labels = []
        for c in self.classes:
            labels.extend([c] * self.samples_per_class)
        self._labels: torch.Tensor = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self._labels.numel())

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._labels[idx]
