from __future__ import annotations

import os
from typing import Any

import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.tuner.tuning import Tuner

from model import CVAELightning
from data_module import MNISTDataModule

from utils.batch_size_finder import run_bs_finder_ephemeral
from utils.learning_rate_finder import run_lr_finder_ephemeral
from utils.make_run_dir import make_run_dir


def _active_group(cfg) -> Any:
    """
    Return the dict-like node for the active subcommand (fit/test/validate/predict)
    """
    for k in ("fit", "test", "validate", "predict"):
        if k in cfg:
            return getattr(cfg, k)
    return cfg  # if subcommands=False


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
        """
        Adding the batch size & learning rate finders
        """
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

                # with no_side_effects(self.trainer):
                #     new_batch_size = tuner.scale_batch_size(self.model, datamodule=self.datamodule, mode=mode)
                new_batch_size = run_bs_finder_ephemeral(self.trainer, self.model, self.datamodule)

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
                # lr_finder = tuner.lr_find(self.model, datamodule=self.datamodule)
                lr_finder = run_lr_finder_ephemeral(self.trainer, self.model, self.datamodule)

                if lr_finder is not None:
                    suggested_lr = lr_finder.suggestion()
                    print(f"\n🔎 Suggested Learning Rate: {suggested_lr:.2e}")
                else:
                    print("⚠️ Could not find optimal learning rate")
            raise SystemExit(0)

    def before_instantiate_classes(self) -> None:
        """
        Adding in the logger & callbacks
        """
        cfg = self.config
        run_command = getattr(self, "subcommand", None)
        run_dir = make_run_dir(str(run_command), base=cfg.get("run", {}).get("base_dir", "runs"))
        root = _active_group(cfg)

        trainer_dict = getattr(root, "trainer", {}) or {}
        trainer_dict["default_root_dir"] = run_dir
        trainer_dict["logger"] = {
            "class_path": "lightning.pytorch.loggers.tensorboard.TensorBoardLogger",
            "init_args": {"save_dir": run_dir, "name": "logs", "default_hp_metric": False},
        }
        trainer_dict["callbacks"] = [
            # Saving the best checkpoint
            {
                "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                "init_args": {
                    "dirpath": os.path.join(run_dir, "checkpoints", "best"),
                    "filename": "e_{epoch}-l_{val_loss:.4f}",
                    "monitor": "val_loss",
                    "mode": "min",
                    "save_top_k": 1,
                },
            },
            # Saving checkpoints every 5 epochs
            {
                "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                "init_args": {
                    "dirpath": os.path.join(run_dir, "checkpoints", "interval"),
                    "filename": "e_{epoch}",
                    "every_n_epochs": 5,
                    "save_top_k": -1,
                    "save_last": False,
                },
            },
            # Monitoring the learning rate
            {
                "class_path": "lightning.pytorch.callbacks.LearningRateMonitor",
                "init_args": {"logging_interval": "epoch"},
            },
            # Sampling 8 images per validation
            {
                "class_path": "callbacks.sample_images.SampleImages",
                "init_args": {"num_per_class": 8},
            },
            # Saving the outputs from prediction
            {
                "class_path": "callbacks.predict_save.SavePredictionsCallback",
                "init_args": {
                    "out_dir": os.path.join(run_dir, "predict"),
                    "grid_nrow": 8,
                },
            },
        ]
        root.trainer = trainer_dict


def cli_main() -> None:
    CVAECLI(model_class=CVAELightning, datamodule_class=MNISTDataModule)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    cli_main()
