from __future__ import annotations
from typing import Optional

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, random_split

from dataset import MNISTDataset, PredictLabelDataset


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
            self._train, self._val = random_split(
                full_train, [train_len, self.val_split]
            )
        if stage in (None, "test"):
            self._test = MNISTDataset(
                root=self.data_dir,
                train=False,
                download=False,
                image_channels=self.image_channels,
            )
        if stage in (None, "predict"):
            self._predict = PredictLabelDataset(
                self.num_classes, self.predict_samples_per_class
            )

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
        return DataLoader(
            self._predict,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
