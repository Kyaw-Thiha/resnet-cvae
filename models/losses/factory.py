from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict, Mapping, MutableMapping

from models.losses.base import BaseLossModule
from models.losses.beta_tc import BetaTCLoss, BetaTCLossConfig
from models.losses.standard import StandardLoss, StandardLossConfig


class LossConfigurationError(ValueError):
    pass


def _normalise_name(name: str) -> str:
    return name.replace("-", "_").strip().lower()


def _build_dataclass_config(cls, data: Mapping[str, Any]) -> Any:
    allowed = {field.name for field in fields(cls)}
    kwargs: Dict[str, Any] = {}
    for key, value in data.items():
        if key not in allowed:
            raise LossConfigurationError(f"Unknown configuration field '{key}' for {cls.__name__}")
        kwargs[key] = value
    return cls(**kwargs)


def build_loss_module(config: Mapping[str, Any] | None) -> BaseLossModule:
    """
    Factory that maps a configuration dict to a concrete loss implementation.

    Passing `None` returns the default `StandardLoss` to preserve backward
    compatibility with previous configs that did not have a dedicated loss block.
    """

    if config is None:
        return StandardLoss()

    if not isinstance(config, Mapping):
        raise LossConfigurationError("Loss configuration must be a mapping/dict.")

    config_map: MutableMapping[str, Any] = dict(config)
    raw_name = config_map.pop("name", "standard")
    name = _normalise_name(str(raw_name))

    if name in {"standard", "beta", "beta_vae"}:
        cfg = _build_dataclass_config(StandardLossConfig, config_map)
        return StandardLoss(cfg)
    if name in {"beta_tc", "beta_tcvae", "betatc", "beta-tc"}:
        cfg = _build_dataclass_config(BetaTCLossConfig, config_map)
        return BetaTCLoss(cfg)

    raise LossConfigurationError(f"Unknown loss name '{raw_name}'.")

