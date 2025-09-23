from __future__ import annotations

from typing import Any
import datetime
import pathlib

import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.tuner.tuning import Tuner
from matplotlib.figure import Figure

from model import CVAELightning
from data_module import MNISTDataModule


def make_run_dir(mode: str, base: str = "runs") -> str:
    """
    Create a unique run directory like:
      runs/train_15_08_2025_19_30

    Args:
        mode: "train" | "valid" | "test" | etc.
        base: parent directory under which runs are saved.

    Returns:
        Absolute string path to the run directory.
    """
    ts = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")
    run_name = f"{mode}_{ts}"
    run_dir = pathlib.Path(base) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return str(run_dir)


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
        parser.add_argument("--show_lr_plot", type=bool, default=True, help="Whether to plot learning rate finder")

    def before_instantiate_classes(self) -> None:
        # pick the "mode" based on the subcommand (fit/validate/test/predict)
        subcmd = self.subcommand or "train"
        run_dir = make_run_dir(subcmd, base="runs")
        os.environ["RUN_DIR"] = run_dir  # available in YAML config via ${oc.env:RUN_DIR}

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

                new_batch_size = tuner.scale_batch_size(self.model, datamodule=self.datamodule, mode=mode)

                print(f"✅ Suggested batch size: {new_batch_size}")

                if new_batch_size is None:
                    print("⚠️ Could not find optimal batch size")
            exit(0)

        # ----------------------------------
        # Finding Optimal Learning Rate
        # ----------------------------------
        # CLI params
        #   run_lr_finder (bool): Determines if LR finder is ran. Default is True.
        #   show_lr_plot (bool): Determines if LR finder plot is show. Default is False.
        # ----------------------------------
        if self.config.fit.run_lr_finder:
            if self.trainer.fast_dev_run:  # pyright: ignore
                print("🚫 Skipping LR finder due to fast_dev_run")
            else:
                lr_finder = tuner.lr_find(self.model, datamodule=self.datamodule)

                if lr_finder is not None:
                    if self.config.fit.show_lr_plot:
                        fig = lr_finder.plot(suggest=True)
                        if isinstance(fig, Figure):
                            fig.savefig("logs/lr_finder_plot.png")

                    suggested_lr = lr_finder.suggestion()
                    print(f"\n🔎 Suggested Learning Rate: {suggested_lr:.2e}")
                else:
                    print("⚠️ Could not find optimal learning rate")
            exit(0)


def cli_main() -> None:
    CVAECLI(model_class=CVAELightning, datamodule_class=MNISTDataModule)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    cli_main()
