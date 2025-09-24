from __future__ import annotations
from typing import Any, Optional
from lightning.pytorch import Trainer
from lightning.pytorch.tuner.tuning import Tuner


def _get(trainer: Trainer, name: str, default: Optional[Any] = None) -> Any:
    # avoid Pyright errors on dynamic attrs
    return getattr(trainer, name, default)


def run_bs_finder_ephemeral(parent: Trainer, model, datamodule, mode: str = "power"):
    t = Trainer(
        accelerator=_get(parent, "accelerator", "auto"),
        devices=_get(parent, "devices", "auto"),
        precision=_get(parent, "precision", "32-true"),
        logger=False,  # ← disable default TensorBoard logger
        enable_checkpointing=False,
        callbacks=[],
        default_root_dir=None,
        max_epochs=1,
        enable_progress_bar=False,
    )
    # mode ∈ {"power","binsearch"}
    new_batch_size = Tuner(t).scale_batch_size(model=model, datamodule=datamodule, mode=mode)
    return new_batch_size
