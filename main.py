from __future__ import annotations

from typing import Any
import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import seed_everything

from model import CVAELightning
from data_module import MNISTDataModule


class CVAECLI(LightningCLI):
    def add_arguments_to_parser(self, parser: Any) -> None:  # type: ignore[override]
        # Link model <-> datamodule defaults
        parser.link_arguments("data.num_classes", "model.num_classes")
        parser.link_arguments("data.image_channels", "model.in_channels")
        parser.link_arguments("data.image_channels", "model.out_channels")


def cli_main() -> None:
    CVAECLI(model_class=CVAELightning, datamodule_class=MNISTDataModule)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    cli_main()
