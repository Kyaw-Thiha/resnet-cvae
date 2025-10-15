from __future__ import annotations

from typing import Tuple, Optional, Union, Dict, cast

from lightning.pytorch.utilities.types import LRSchedulerConfig, OptimizerLRScheduler
import torch
from torch import Tensor
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from lightning.pytorch.core.module import LightningModule

from models.baseline_cvae.baseline_cvae import BaselineCVAE
from models.resnet_cvae.resnet_cvae import ResNetCVAE


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
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        beta: float = 1.0,
        kl_warmup_epochs: int = 10,
        max_epochs: int = 50,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # self.model: ResNetCVAE = ResNetCVAE(
        #     in_ch=in_channels,
        #     out_ch=out_channels,
        #     z_dim=z_dim,
        #     num_classes=num_classes,
        #     cond_dim=cond_dim,
        #     use_film=use_film,
        # )
        self.model: BaselineCVAE = BaselineCVAE(
            in_ch=in_channels,
            out_ch=out_channels,
            z_dim=z_dim,
            num_classes=num_classes,
            cond_dim=cond_dim,
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
        B, C, H, W = x.shape
        pixel_count = C * H * W

        self.log("train/loss", total, prog_bar=True)
        self.log("train/recon", recon)
        self.log("train/kl", kl)
        self.log("train/beta", torch.tensor(self._beta()), prog_bar=True)
        self.log("train/recon_per_pixel", recon / pixel_count, sync_dist=False)
        self.log("train/kl_per_pixel", kl / pixel_count, sync_dist=False)
        return total

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        x, y = batch
        total, recon, kl = self.model.loss(x, y, beta=float(self._beta()))
        B, C, H, W = x.shape
        pixel_count = C * H * W

        self.log("val_loss", total, prog_bar=True, sync_dist=False)
        self.log("val/recon", recon, sync_dist=False)
        self.log("val/kl", kl, sync_dist=False)
        self.log("val/recon_per_pixel", recon / pixel_count, sync_dist=False)
        self.log("val/kl_per_pixel", kl / pixel_count, sync_dist=False)

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        x, y = batch
        total, recon, kl = self.model.loss(x, y, beta=float(self.beta))
        B, C, H, W = x.shape
        pixel_count = C * H * W

        self.log("test/loss", total, prog_bar=True)
        self.log("test/recon", recon)
        self.log("test/kl", kl)
        self.log("test/recon_per_pixel", recon / pixel_count, sync_dist=False)
        self.log("test/kl_per_pixel", kl / pixel_count, sync_dist=False)

    def on_fit_start(self) -> None:
        """Sync with Trainer so our LR schedule uses the true max_epochs (from YAML/CLI)."""
        if self.trainer.max_epochs:
            self._max_epochs = int(self.trainer.max_epochs)

    # -----------------
    # Optim / Sched
    # -----------------
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.AdamW(self.parameters(), lr=float(self.lr), weight_decay=float(self.weight_decay))

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
        batch: Union[Tensor, Tuple[Tensor, ...], Dict[str, Tensor]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Tensor:
        """
        Expects a batch of integer labels (B,) or one-hots (B, num_classes).
        Returns generated images in [-inf, +inf] (means of Gaussian). You can post-process outside.

        Prediction entrypoint with controllable generation.

        Supports two batch formats:
        1) Tensor: interpreted as labels `y` of shape (B,)
        2) Dict: keys may include:
            - "y": (B,) labels or (B, num_classes) one-hot  [REQUIRED]
            - "z": (B, z_dim) custom latent codes           [OPTIONAL]
            - "temperature": float                          [OPTIONAL] default 1.0
            - "guidance_scale": float                       [OPTIONAL] default 0.0
            - "cond_scale": float                           [OPTIONAL] default 1.0
            - "seed": int                                   [OPTIONAL]

        Notes:
            • If any optional keys are missing, sensible defaults are used.
            • Output are decoder means (Gaussian), suitable to rescale from [-1,1] to [0,1].

        Returns:
            (B, out_ch, H, W) tensor of generated images.
        """
        temperature: float = 1.0
        guidance_scale: float = 0.0
        cond_scale: float = 1.0
        seed: Optional[int] = None
        z: Optional[Tensor] = None

        if isinstance(batch, dict):
            y = batch["y"]
            z = batch.get("z")
            if "temperature" in batch:
                temperature = float(torch.as_tensor(batch.get("temperature", 1.0)).reshape(-1)[0])
            if "guidance_scale" in batch:
                guidance_scale = float(torch.as_tensor(batch.get("guidance_scale", 0.0)).reshape(-1)[0])
            if "cond_scale" in batch:
                cond_scale = float(torch.as_tensor(batch.get("cond_scale", 1.0)).reshape(-1)[0])
            if "seed" in batch:
                seed = int(torch.as_tensor(batch["seed"]).reshape(-1)[0].item())
                seed = int(seed) + int(batch_idx)
                # seed = int(batch["seed"])  # type: ignore[arg-type]

            if isinstance(y, torch.Tensor):
                if y.dim() == 1:
                    assert torch.all((0 <= y) & (y < self.num_classes)), f"labels out of range: {y.min()}..{y.max()}"
                elif y.dim() == 2:
                    assert y.shape[1] == self.num_classes, f"one-hot width {y.shape[1]} != num_classes {self.num_classes}"
                else:
                    raise ValueError(f"Unexpected y shape: {y.shape}")

        elif isinstance(batch, tuple):
            y = batch[0]
        else:
            y = batch

        b: int = y.shape[0]
        return self.model.sample(
            n=b,
            y=y,
            device=self.device,
            temperature=temperature,
            seed=seed,
            z=z,
            guidance_scale=guidance_scale,
            cond_scale=cond_scale,
        )

    def sample(
        self,
        n: int,
        y: Tensor,
        device: Optional[torch.device] = None,
        *,
        temperature: float = 1.0,
        seed: Optional[int] = None,
        z: Optional[Tensor] = None,
        guidance_scale: float = 0.0,
        cond_scale: float = 1.0,
    ) -> Tensor:
        """
        Thin, typed pass-through to the underlying model's controllable sampler.
        Args:
            n: int
                Number of images to sample.
            y: (n,) labels or (n, num_classes) one-hot
                Class labels for conditioning.
            device: Optional[torch.device]
                Device override; defaults to module device.
            temperature: float, default=1.0
                Latent prior std scale (truncation/diversity control).
            seed: Optional[int]
                RNG seed for reproducible sampling (ignored if `z` provided).
            z: Optional[Tensor]
                Custom latents of shape (n, z_dim).
            guidance_scale: float, default=0.0
                CFG-style guidance blend factor.
            cond_scale: float, default=1.0
                Label-conditioning strength multiplier.

        Returns:
            (n, out_ch, H, W) tensor of generated images (decoder means).
        """
        dev = self.device if device is None else device
        return self.model.sample(
            n=n,
            y=y,
            device=dev,
            temperature=temperature,
            seed=seed,
            z=z,
            guidance_scale=guidance_scale,
            cond_scale=cond_scale,
        )
