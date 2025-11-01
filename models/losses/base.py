from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class LossInputs:
    """
    Container for the tensors produced by a CVAE forward pass.

    Storing them in a dataclass keeps the loss interfaces explicit and
    self-documenting as we add new loss variants.
    """

    x: Tensor
    x_hat: Tensor
    log_sigma: Tensor
    mu: Tensor
    logvar: Tensor
    z: Tensor


@dataclass(frozen=True)
class LossContext:
    """
    Runtime context that may influence the loss weighting.

    beta:
        KL/regularizer weight after scheduling (e.g. KL warm-up).
    epoch:
        Current epoch (0-indexed) provided mainly for scheduling hooks.
    global_step:
        Global optimization step, useful for step-wise annealing.
    """

    beta: float
    epoch: int
    global_step: int


@dataclass
class LossOutput:
    """
    Standard structure returned by all loss modules.
    """

    total: Tensor
    reconstruction: Tensor
    kl: Tensor
    metrics: Dict[str, Tensor] = field(default_factory=dict)

    def detach(self) -> "LossOutput":
        """
        Convenience helper to return a detached copy for logging. Not currently used
        but kept to make the contract explicit for future variants.
        """

        def _safe_detach(value: Tensor) -> Tensor:
            return value.detach() if value.requires_grad else value

        detached_metrics = {k: _safe_detach(v) for k, v in self.metrics.items()}
        return LossOutput(
            total=_safe_detach(self.total),
            reconstruction=_safe_detach(self.reconstruction),
            kl=_safe_detach(self.kl),
            metrics=detached_metrics,
        )


class BaseLossModule:
    """
    Base class for all CVAE losses used in this project.

    Loss implementations can optionally demand the training dataset size to compute
    minibatch-based density estimates (as required by β-TCVAE). The default
    implementation is a no-op so that simple objectives remain straightforward.
    """

    def __init__(self) -> None:
        self._dataset_size: Optional[int] = None

    def requires_dataset_size(self) -> bool:
        return False

    def set_dataset_size(self, dataset_size: Optional[int]) -> None:
        self._dataset_size = dataset_size

    @property
    def dataset_size(self) -> Optional[int]:
        return self._dataset_size

    def __call__(self, inputs: LossInputs, context: LossContext) -> LossOutput:
        raise NotImplementedError


def ensure_tensor(value: float | Tensor) -> Tensor:
    """
    Utility for creating scalar tensors on the correct device without
    scattering manual `torch.tensor(...)` calls across logging code.
    """

    if isinstance(value, Tensor):
        return value
    return torch.tensor(float(value))

