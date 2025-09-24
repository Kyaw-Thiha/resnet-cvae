from __future__ import annotations
from typing import Any, Optional
from lightning.pytorch import Trainer
from lightning.pytorch.tuner.tuning import Tuner


def _get(trainer: Trainer, name: str, default: Optional[Any] = None) -> Any:
    # avoid Pyright errors on dynamic attrs
    return getattr(trainer, name, default)


def run_lr_finder_ephemeral(parent: Trainer, model, datamodule):
    # mirror only what we need; keep it safe + side-effect free
    t = Trainer(
        accelerator=_get(parent, "accelerator", "auto"),
        devices=_get(parent, "devices", "auto"),
        precision=_get(parent, "precision", "32-true"),
        logger=False,  # ← disable default TensorBoard logger
        enable_checkpointing=False,
        callbacks=[],  # no file-writing callbacks
        default_root_dir=None,  # nothing to write anyway
        max_epochs=1,
        enable_progress_bar=False,
    )
    lr_finder = Tuner(t).lr_find(model=model, datamodule=datamodule)
    return lr_finder  # lr_finder.suggestion()
