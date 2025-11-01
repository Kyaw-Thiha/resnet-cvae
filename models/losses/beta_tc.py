from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Set, Tuple

import torch
from torch import Tensor

from models.losses.base import BaseLossModule, LossContext, LossInputs, LossOutput, ensure_tensor
from models.utils import gaussian_nll

LOG_TWO_PI = math.log(2.0 * math.pi)


def _log_normal_diag(value: Tensor, mean: Tensor, logvar: Tensor) -> Tensor:
    """
    Log-density of a diagonal-covariance Gaussian for each element in `value`.

    Returns a tensor with the same shape as `value`, containing per-dimension log
    probabilities.
    """

    return -0.5 * (LOG_TWO_PI + logvar + (value - mean) ** 2 / logvar.exp())


def _matrix_log_density_gaussian(z: Tensor, mean: Tensor, logvar: Tensor) -> Tensor:
    """
    Builds a (B, B, D) tensor where entry [i, j, d] is
    log q(z_i_d | x_j) under the encoder's Gaussian posterior.
    """

    z_expanded = z.unsqueeze(1)  # (B, 1, D)
    mean_expanded = mean.unsqueeze(0)  # (1, B, D)
    logvar_expanded = logvar.unsqueeze(0)  # (1, B, D)
    return _log_normal_diag(z_expanded, mean_expanded, logvar_expanded)


def _log_standard_normal(z: Tensor) -> Tensor:
    return -0.5 * (LOG_TWO_PI + z.pow(2))


def _parse_anneal_targets(target: str | Iterable[str]) -> Set[str]:
    if isinstance(target, str):
        tokens = {token.strip().lower() for token in target.split(",")}
    else:
        tokens = {str(t).strip().lower() for t in target}

    valid = {"none", "mi", "mutual_information", "tc", "total_correlation", "dw", "dimension_wise", "all"}
    if not tokens.issubset(valid):
        unknown = tokens - valid
        raise ValueError(f"Unknown anneal targets: {sorted(unknown)}")
    if "none" in tokens and len(tokens) > 1:
        raise ValueError("Anneal target 'none' cannot be combined with other targets.")
    if "all" in tokens:
        return {"mi", "tc", "dw"}

    mapped = set()
    for token in tokens:
        if token in {"mi", "mutual_information"}:
            mapped.add("mi")
        elif token in {"tc", "total_correlation"}:
            mapped.add("tc")
        elif token in {"dw", "dimension_wise"}:
            mapped.add("dw")
    return mapped


@dataclass
class BetaTCLossConfig:
    """
    Configuration for the β-TCVAE objective from Chen et al. (2018).

    The KL divergence is decomposed into:
        • Mutual information         (I)
        • Total correlation          (TC)
        • Dimension-wise KL          (DW)

    Each component receives its own weight (α, β, γ in the paper). The scheduled
    beta passed via `LossContext` can optionally modulate any subset of these
    weights using the `anneal_target` field.
    """

    reconstruction_weight: float = 1.0
    mutual_information_weight: float = 1.0  # α
    total_correlation_weight: float = 6.0  # β
    dimension_wise_kl_weight: float = 1.0  # γ
    log_sigma_bounds: Tuple[float, float] = (-3.0, 0.5)
    log_sigma_l2_weight: float = 1e-4
    anneal_target: str = "total_correlation"


class BetaTCLoss(BaseLossModule):
    """
    Implements the β-TCVAE objective with minibatch-based estimators for the KL
    decomposition. Requires the training dataset size to properly normalise the
    aggregated posterior estimates.
    """

    def __init__(self, config: BetaTCLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or BetaTCLossConfig()
        self._anneal_targets = _parse_anneal_targets(self.config.anneal_target)

    def requires_dataset_size(self) -> bool:
        return True

    def __call__(self, inputs: LossInputs, context: LossContext) -> LossOutput:
        if self.dataset_size is None:
            raise RuntimeError(
                "BetaTCLoss requires the training dataset size. "
                "Ensure `set_dataset_size` is called before training starts."
            )

        cfg = self.config
        log_sigma_min, log_sigma_max = cfg.log_sigma_bounds

        log_sigma_clamped = inputs.log_sigma.clamp(min=log_sigma_min, max=log_sigma_max)
        sigma = log_sigma_clamped.exp()

        recon_vector = gaussian_nll(inputs.x, inputs.x_hat, sigma=sigma)
        recon = recon_vector.mean() * float(cfg.reconstruction_weight)

        if cfg.log_sigma_l2_weight > 0:
            recon = recon + float(cfg.log_sigma_l2_weight) * (log_sigma_clamped**2).mean()

        log_q_z_given_x = _log_normal_diag(inputs.z, inputs.mu, inputs.logvar).sum(dim=1)

        # Estimate aggregated posterior statistics following Chen et al. (2018).
        log_qz_matrix = _matrix_log_density_gaussian(inputs.z, inputs.mu, inputs.logvar)
        batch_size = inputs.z.shape[0]
        dataset_size = max(1, int(self.dataset_size))
        log_qz = torch.logsumexp(log_qz_matrix.sum(dim=2), dim=1) - math.log(batch_size * dataset_size)
        log_qz_prod = (
            torch.logsumexp(log_qz_matrix, dim=1) - math.log(batch_size * dataset_size)
        ).sum(dim=1)

        log_p_z = _log_standard_normal(inputs.z).sum(dim=1)

        mutual_information = (log_q_z_given_x - log_qz).mean()
        total_correlation = (log_qz - log_qz_prod).mean()
        dimension_wise_kl = (log_qz_prod - log_p_z).mean()

        # Combine the components with configurable weights.
        beta_scale = float(context.beta)
        weights = {
            "mi": float(cfg.mutual_information_weight),
            "tc": float(cfg.total_correlation_weight),
            "dw": float(cfg.dimension_wise_kl_weight),
        }

        if "mi" in self._anneal_targets:
            weights["mi"] *= beta_scale
        if "tc" in self._anneal_targets:
            weights["tc"] *= beta_scale
        if "dw" in self._anneal_targets:
            weights["dw"] *= beta_scale

        kl = mutual_information + total_correlation + dimension_wise_kl
        total = recon + weights["mi"] * mutual_information + weights["tc"] * total_correlation + weights["dw"] * dimension_wise_kl

        metrics: Dict[str, Tensor] = {
            "beta": ensure_tensor(context.beta),
            "mi": mutual_information.detach(),
            "tc": total_correlation.detach(),
            "dw_kl": dimension_wise_kl.detach(),
            "mi_weight": ensure_tensor(weights["mi"]),
            "tc_weight": ensure_tensor(weights["tc"]),
            "dw_weight": ensure_tensor(weights["dw"]),
        }

        return LossOutput(total=total, reconstruction=recon, kl=kl, metrics=metrics)

