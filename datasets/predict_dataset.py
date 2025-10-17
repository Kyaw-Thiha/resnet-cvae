from __future__ import annotations


from typing import Optional, Sequence, Any, Dict

import torch
from torch.utils.data import Dataset


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


class PredictWrapper(Dataset):
    def __init__(self, base: PredictLabelDataset, cfg: Dict[str, Any]) -> None:
        self.base = base
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        y = self.base[idx]
        out: Dict[str, Any] = {
            "y": y,
            "temperature": torch.tensor(self.cfg["temperature"]),
            "guidance_scale": torch.tensor(self.cfg["guidance_scale"]),
            "cond_scale": torch.tensor(self.cfg["cond_scale"]),
        }
        if self.cfg.get("seed") is not None:
            out["seed"] = torch.tensor(self.cfg["seed"])
        return out
