from __future__ import annotations
from typing import Optional

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, random_split

from dataset import MNISTDataset, PredictLabelDataset

from torch.utils.data import Dataset
import torch
from typing import Any, Dict


class MNISTDataModule(LightningDataModule):
    """
    Lightning DataModule for MNIST with train/val/test splits and an optional
    predict loader that yields labels for class-conditional generation.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 128,
        num_workers: int = 4,
        val_split: int = 5000,
        image_channels: int = 1,
        num_classes: int = 10,
        predict_samples_per_class: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        # generation controls (used by predict dataloader)
        temperature: float = 1.0,
        guidance_scale: float = 0.0,
        cond_scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.val_split = int(val_split)
        self.image_channels = int(image_channels)
        self.num_classes = int(num_classes)
        self.predict_samples_per_class = int(predict_samples_per_class)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)

        # predict controls
        self.temperature = float(temperature)
        self.guidance_scale = float(guidance_scale)
        self.cond_scale = float(cond_scale)
        self.seed = seed

        self._train = None
        self._val = None
        self._test = None
        self._predict = None

    # Called only on rank 0; downloads if necessary
    def prepare_data(self) -> None:  # type: ignore[override]
        MNISTDataset(root=self.data_dir, train=True, download=True)
        MNISTDataset(root=self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:  # type: ignore[override]
        if stage in (None, "fit"):
            full_train = MNISTDataset(
                root=self.data_dir,
                train=True,
                download=False,
                image_channels=self.image_channels,
            )
            train_len = len(full_train) - self.val_split
            self._train, self._val = random_split(full_train, [train_len, self.val_split])
        if stage in (None, "test"):
            self._test = MNISTDataset(
                root=self.data_dir,
                train=False,
                download=False,
                image_channels=self.image_channels,
            )
        if stage in (None, "predict"):
            self._predict = PredictLabelDataset(self.num_classes, self.predict_samples_per_class)

    # Dataloaders
    def train_dataloader(self) -> DataLoader:
        assert self._train is not None
        return DataLoader(
            self._train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val is not None
        return DataLoader(
            self._val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        assert self._test is not None
        return DataLoader(
            self._test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        assert self._predict is not None

        class _GenWrapper(Dataset):  # type: ignore[misc]
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

        cfg = {
            "temperature": self.temperature,
            "guidance_scale": self.guidance_scale,
            "cond_scale": self.cond_scale,
            "seed": self.seed,
        }
        wrapped = _GenWrapper(self._predict, cfg)
        return DataLoader(
            wrapped,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
