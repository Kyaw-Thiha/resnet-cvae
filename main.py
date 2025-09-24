from __future__ import annotations

from typing import Any

import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.tuner.tuning import Tuner

from model import CVAELightning
from data_module import MNISTDataModule

from utils.no_side_effects import no_side_effects


class CVAECLI(LightningCLI):
    def add_arguments_to_parser(self, parser: Any) -> None:  # type: ignore[override]
        # Link model <-> datamodule defaults
        parser.link_arguments("data.num_classes", "model.num_classes")
        parser.link_arguments("data.image_channels", "model.in_channels")
        parser.link_arguments("data.image_channels", "model.out_channels")

        # Batch Size Finder
        parser.add_argument("--run_batch_size_finder", type=bool, default=False, help="Whether to run the batch size finder")
        parser.add_argument(
            "--batch_size_finder_mode", type=str, default="power", help="Mode for batch size finder (power|binsearch)"
        )
        # Learning Rate Finder
        parser.add_argument("--run_lr_finder", type=bool, default=False, help="Whether to run learning rate finder")

    def before_fit(self):
        tuner = Tuner(self.trainer)

        # ----------------------------------
        # Batch Size Finder
        # ----------------------------------
        # CLI params
        #   run_batch_size_finder (bool): Determines if batch_size finder is ran. Default is True.
        #   batch_size_finder_mode (str): "power" or "binsearch". Determines the mode of batch_size finder
        # ----------------------------------
        if self.config.fit.run_batch_size_finder:
            if self.trainer.fast_dev_run:  # pyright: ignore
                print("🚫 Skipping batch finder due to fast_dev_run")
            else:
                mode = self.config.fit.batch_size_finder_mode
                print(f"\n📦 Running batch size finder (mode: {mode})...")

                with no_side_effects(self.trainer):
                    new_batch_size = tuner.scale_batch_size(self.model, datamodule=self.datamodule, mode=mode)

                print(f"✅ Suggested batch size: {new_batch_size}")

                if new_batch_size is None:
                    print("⚠️ Could not find optimal batch size")
            raise SystemExit(0)

        # ----------------------------------
        # Finding Optimal Learning Rate
        # ----------------------------------
        # CLI params
        #   run_lr_finder (bool): Determines if LR finder is ran. Default is True.
        #   show_lr_plot (bool): Determines if LR finder plot is show. Default is False. [Disabled]
        # ----------------------------------
        if self.config.fit.run_lr_finder:
            if self.trainer.fast_dev_run:  # pyright: ignore
                print("🚫 Skipping LR finder due to fast_dev_run")
            else:
                with no_side_effects(self.trainer):
                    lr_finder = tuner.lr_find(self.model, datamodule=self.datamodule)

                    if lr_finder is not None:
                        suggested_lr = lr_finder.suggestion()
                        print(f"\n🔎 Suggested Learning Rate: {suggested_lr:.2e}")
                    else:
                        print("⚠️ Could not find optimal learning rate")
            raise SystemExit(0)


def cli_main() -> None:
    CVAECLI(model_class=CVAELightning, datamodule_class=MNISTDataModule)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    cli_main()
