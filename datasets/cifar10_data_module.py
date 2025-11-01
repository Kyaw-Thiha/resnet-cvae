from __future__ import annotations
from typing import Optional, Sequence

from torch.utils.data import DataLoader, random_split
from lightning.pytorch import LightningDataModule

from datasets.predict_dataset import PredictLabelDataset, PredictWrapper
from datasets.cifar10_dataset import CIFAR10Dataset


class CIFAR10DataModule(LightningDataModule):
    """
    CIFAR-10 DataModule (RGB only).
    """

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 128,
        num_workers: int = 4,
        val_split: int = 5000,
        image_channels: int = 3,
        num_classes: int = 10,
        predict_samples_per_class: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        # predict controls
        predict_classes: Optional[Sequence[int]] = None,
        temperature: float = 1.0,
        guidance_scale: float = 0.0,
        cond_scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.num_classes = num_classes
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.predict_samples_per_class = predict_samples_per_class
        self.predict_classes = list(predict_classes) if predict_classes is not None else None
        self.temperature = temperature
        self.guidance_scale = guidance_scale
        self.cond_scale = cond_scale
        self.seed = seed
        self._train_size: Optional[int] = None

        self._train = None
        self._val = None
        self._test = None
        self._predict = None

    def prepare_data(self) -> None:  # type: ignore[override]
        CIFAR10Dataset(self.data_dir, train=True, download=True)
        CIFAR10Dataset(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:  # type: ignore[override]
        if stage in (None, "fit"):
            full_train = CIFAR10Dataset(self.data_dir, train=True, download=False)
            train_len = len(full_train) - self.val_split
            self._train, self._val = random_split(full_train, [train_len, self.val_split])
            self._train_size = int(train_len)

        if stage in (None, "test"):
            self._test = CIFAR10Dataset(self.data_dir, train=False, download=False)

        if stage in (None, "predict"):
            self._predict = PredictLabelDataset(
                num_classes=self.num_classes,
                samples_per_class=self.predict_samples_per_class,
                classes=self.predict_classes,
            )

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
        cfg = {
            "temperature": self.temperature,
            "guidance_scale": self.guidance_scale,
            "cond_scale": self.cond_scale,
            "seed": self.seed,
        }
        wrapped = PredictWrapper(self._predict, cfg)
        return DataLoader(
            wrapped,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    @property
    def train_dataset_size(self) -> Optional[int]:
        if self._train_size is not None:
            return self._train_size
        if self._train is not None:
            try:
                return int(len(self._train))
            except TypeError:
                return None
        return None
