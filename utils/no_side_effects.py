from __future__ import annotations

import tempfile
import contextlib
from typing import Iterator, List, Any, Optional, cast
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch import Trainer
from lightning.pytorch.loggers.logger import Logger


@contextlib.contextmanager
def no_side_effects(trainer: Trainer) -> Iterator[str]:
    """
    Temporarily make the Trainer 'stateless' wrt filesystem:
    - disable logger (set to None)
    - disable checkpointing (strip ModelCheckpoint callbacks)
    - strip any custom file-writing callbacks by name/type
    - redirect default_root_dir to a TemporaryDirectory
    Restores everything afterward.

    Yields the temporary directory path (if you still want to store something there).
    """
    # --- Snapshot originals (typed to satisfy Pyright) ---
    old_logger: Optional[Logger] = cast(Optional[Logger], getattr(trainer, "logger", None))
    old_callbacks: List[Callback] = list(getattr(trainer, "callbacks", []))  # fall back to []
    old_root: Optional[str] = cast(Optional[str], getattr(trainer, "default_root_dir", None))

    # Some Lightning versions keep a separate flag for checkpointing at init-time only.
    # We emulate "disable checkpointing" by removing ModelCheckpoint callbacks.
    def _keep(cb: Callback) -> bool:
        if isinstance(cb, ModelCheckpoint):
            return False
        # Drop your own filesystem-writing callbacks by class name if desired:
        if cb.__class__.__name__ in {"SampleImages"}:
            return False
        return True

    with tempfile.TemporaryDirectory(prefix="dryrun_") as tmpdir:
        try:
            # Turn off the logger: use None (not False) post-construction
            cast(Any, trainer).logger = None  # type: ignore[assignment]

            # Strip checkpoint/image writer callbacks
            cast(Any, trainer).callbacks = [cb for cb in old_callbacks if _keep(cb)]  # type: ignore[assignment]

            # Redirect any stray writes
            cast(Any, trainer).default_root_dir = tmpdir  # type: ignore[assignment]

            yield tmpdir
        finally:
            # Restore everything
            cast(Any, trainer).logger = old_logger  # type: ignore[assignment]
            cast(Any, trainer).callbacks = old_callbacks  # type: ignore[assignment]
            # Only restore root if it existed; some trainers compute a default on the fly
            if old_root is not None:
                cast(Any, trainer).default_root_dir = old_root  # type: ignore[assignment]
