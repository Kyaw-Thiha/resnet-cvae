from __future__ import annotations
import os
from typing import Any, List, cast

from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import Logger, TensorBoardLogger

from utils.make_run_dir import make_run_dir


def _iter_loggers(logger: Any) -> List[Logger]:
    """
    Return a flat list of Logger instances.
    Works with single Logger or a logger-collection without importing its type.
    """
    if logger is None:
        return []
    if isinstance(logger, Logger):
        return [logger]

    # Heuristics for LoggerCollection across PL/Lightning versions
    if hasattr(logger, "loggers"):
        try:
            return [lg for lg in getattr(logger, "loggers") if isinstance(lg, Logger)]
        except Exception:
            pass
    if hasattr(logger, "_logger_iterable"):
        try:
            return [lg for lg in getattr(logger, "_logger_iterable") if isinstance(lg, Logger)]
        except Exception:
            pass

    try:
        return [lg for lg in logger if isinstance(lg, Logger)]
    except Exception:
        return []


class RunDirBootstrap(Callback):
    def __init__(self, base_path: str = "runs") -> None:
        self.base_path = base_path

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        run_dir = make_run_dir("fit", base=self.base_path)

        # 1) set the trainer's root for this run
        cast(Any, trainer).default_root_dir = run_dir  # type: ignore[assignment]

        # 2) REPLACE the TB logger with one that points to <run_dir>/logs
        new_tb = TensorBoardLogger(
            save_dir=os.path.join(run_dir, "logs"),
            name=None,  # no extra "name" subdir; optional
            default_hp_metric=False,
        )

        existing = trainer.logger
        if existing is None:
            cast(Any, trainer).logger = new_tb
        else:
            # Build a new collection: [new_tb] + all non-TB loggers already present
            others = [lg for lg in _iter_loggers(existing) if not isinstance(lg, TensorBoardLogger)]

            # Minimal LoggerCollection without importing its type
            class _LoggerCollection(list):
                @property
                def name(self) -> str:
                    return ",".join(getattr(l, "name", l.__class__.__name__) for l in self)

            cast(Any, trainer).logger = _LoggerCollection([new_tb] + others)

        # 3) point ModelCheckpoint(s) into <run_dir>/checkpoints/<best|interval>
        for cb in cast(Any, trainer).callbacks:  # type: ignore[attr-defined]
            if isinstance(cb, ModelCheckpoint):
                dirpath = getattr(cb, "dirpath", None)
                if not dirpath:
                    every_n = int(getattr(cb, "every_n_epochs", 0) or 0)
                    sub = "interval" if every_n > 0 else "best"
                    cast(Any, cb).dirpath = os.path.join(run_dir, "checkpoints", sub)  # type: ignore[assignment]
