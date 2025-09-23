from __future__ import annotations

from typing import Tuple, Optional, cast

from lightning.pytorch.utilities.types import LRSchedulerConfig, OptimizerLRScheduler
import torch
from torch import Tensor
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from lightning.pytorch.core.module import LightningModule

from resnet_cvae.resnet_cvae import ResNetCVAE


class CVAELightning(LightningModule):
    """
    Lightning wrapper for the class-conditional ResNet CVAE with Gaussian likelihood.
    Handles training/validation/test loops and image generation during predict.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        z_dim: int = 16,
        num_classes: int = 10,
        cond_dim: int = 16,
        use_film: bool = False,
        sigma: float = 0.1,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        beta: float = 1.0,
        kl_warmup_epochs: int = 10,
        max_epochs: int = 50,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model: ResNetCVAE = ResNetCVAE(
            in_ch=in_channels,
            out_ch=out_channels,
            z_dim=z_dim,
            num_classes=num_classes,
            cond_dim=cond_dim,
            use_film=use_film,
            sigma=sigma,
        )

        # Image Classes
        self.num_classes: int = num_classes

        # Hyperparameters
        self.lr: float = lr
        self.weight_decay: float = weight_decay
        self.beta: float = beta
        self.kl_warmup_epochs: int = kl_warmup_epochs

        # Will be set by the trainer; used in KL warm-up schedule
        self._max_epochs: int = max_epochs

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.model(x, y)

    # -----------------
    # Loss components
    # -----------------
    def _beta(self) -> float:
        # Linear KL warm-up for first kl_warmup_epochs, then hold at hparams.beta
        current_epoch: int = int(self.current_epoch)
        warm: int = int(self.kl_warmup_epochs)
        target: float = float(self.beta)
        if warm <= 0:
            return target
        return float(min(1.0, (current_epoch + 1) / warm) * target)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        total, recon, kl = self.model.loss(x, y, beta=self._beta())
        self.log("train/loss", total, prog_bar=True)
        self.log("train/recon", recon)
        self.log("train/kl", kl)
        self.log("train/beta", torch.tensor(self._beta()), prog_bar=True)
        return total

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        x, y = batch
        total, recon, kl = self.model.loss(x, y, beta=float(self.beta))
        self.log("val/loss", total, prog_bar=True, sync_dist=False)
        self.log("val/recon", recon, sync_dist=False)
        self.log("val/kl", kl, sync_dist=False)

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        x, y = batch
        total, recon, kl = self.model.loss(x, y, beta=float(self.beta))
        self.log("test/loss", total, prog_bar=True)
        self.log("test/recon", recon)
        self.log("test/kl", kl)

    def on_fit_start(self) -> None:
        # sync with Trainer in case YAML/CLI changed it
        if self.trainer.max_epochs:
            self._max_epochs = int(self.trainer.max_epochs)

    # -----------------
    # Optim / Sched
    # -----------------
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.AdamW(
            self.parameters(), lr=float(self.lr), weight_decay=float(self.weight_decay)
        )

        # Optional: cosine decay after warm-up portion of epochs
        def lr_lambda(epoch: int) -> float:
            max_ep: int = int(self._max_epochs)
            warm: int = int(max(1, self.kl_warmup_epochs))
            if epoch < warm:
                return (epoch + 1) / warm
            # cosine from 1.0 -> 0.2
            import math

            t: float = (epoch - warm) / max(1, max_ep - warm)
            return 0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * t))

        scheduler = cast(
            LRSchedulerConfig,
            {
                "scheduler": LambdaLR(optimizer, lr_lambda=lr_lambda),
                "interval": "epoch",
                "frequency": 1,
            },
        )
        return [optimizer], [scheduler]

    # -----------------
    # Prediction / Generation
    # -----------------
    def predict_step(
        self,
        batch: Tensor | Tuple[Tensor, ...],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Tensor:
        """
        Expects a batch of integer labels (B,) or one-hots (B, num_classes).
        Returns generated images in [-inf, +inf] (means of Gaussian). You can post-process outside.
        """
        if isinstance(batch, tuple):
            y = batch[0]
        else:
            y = batch
        b: int = y.shape[0]
        return self.model.sample(n=b, y=y, device=self.device)

    def sample(
        self, n: int, y: Tensor, device: Optional[torch.device] = None
    ) -> Tensor:
        dev = self.device if device is None else device
        return self.model.sample(n=n, y=y, device=dev)
