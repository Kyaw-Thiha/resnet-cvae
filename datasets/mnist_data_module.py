from __future__ import annotations
from typing import Optional

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, random_split

from datasets.mnist_dataset import MNISTDataset
from datasets.predict_dataset import PredictLabelDataset, PredictWrapper


class MNISTDataModule(LightningDataModule):
    """
    DataModule for MNIST with train/val/test splits and an optional predict path.

    Predict-time controls (optional):
        temperature: Optional[float]
            Latent prior std scale for sampling (truncation/diversity).
        guidance_scale: Optional[float]
            CFG-style guidance scale (0 disables).
        cond_scale: Optional[float]
            Label-conditioning strength multiplier.
        seed: Optional[int]
            RNG seed for reproducible z when not provided explicitly.

    Notes:
        • These fields are ignored during fit/validate/test.
        • If unset (None), they are omitted from predict batches and model defaults apply.
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
        predict_classes: Optional[list[int]] = None,
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
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)

        # predict controls
        self.predict_samples_per_class = int(predict_samples_per_class)
        self.predict_classes = predict_classes
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
            self._predict = PredictLabelDataset(self.num_classes, self.predict_samples_per_class, self.predict_classes)

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
        """
        Builds a dataloader that yields dictionaries consumed by `predict_step`.

        Each item is a dict with:
          - "y": label for the sample
          - Optional controls (only included if provided on the DataModule):
              "temperature", "guidance_scale", "cond_scale", "seed"

        Rationale:
            Keeping these fields Optional avoids leaking prediction-only flags into
            training/validation loops and preserves clean configurations.
        """

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
