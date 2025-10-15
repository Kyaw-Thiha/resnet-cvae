from typing import List, Dict, Optional
import os
import torch
from torch import Tensor
from torchvision.utils import save_image, make_grid
from lightning.pytorch import Callback, Trainer, LightningModule


def _to_01(x: Tensor) -> Tensor:
    # Map from [-1,1] to [0,1]; clamp in case
    return (x.clamp(-1, 1) + 1) * 0.5


class SavePredictionsCallback(Callback):
    """
    Collect predictions during predict() and save:
      - per-sample PNGs in out_dir/samples/
      - per-class grids in out_dir/grids/
    Assumes predict batches contain 'y' labels and model outputs are (B,C,H,W).
    """

    def __init__(self, out_dir: str, grid_nrow: Optional[int] = None) -> None:
        super().__init__()
        self.out_dir = out_dir
        self.grid_nrow = grid_nrow
        self._imgs: List[Tensor] = []
        self._ys: List[Tensor] = []

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._imgs.clear()
        self._ys.clear()
        if trainer.is_global_zero:
            os.makedirs(self.out_dir, exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, "samples"), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, "grids"), exist_ok=True)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor,
        batch: Dict[str, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._imgs.append(outputs.detach().cpu())
        self._ys.append(batch["y"].detach().cpu())

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        imgs = _to_01(torch.cat(self._imgs, dim=0))
        ys = torch.cat(self._ys, dim=0)

        # per-sample
        samp_dir = os.path.join(self.out_dir, "samples")
        for i in range(imgs.size(0)):
            save_image(imgs[i], os.path.join(samp_dir, f"{i:05d}_y{int(ys[i])}.png"))

        # per-class grids
        grid_dir = os.path.join(self.out_dir, "grids")
        for c in sorted(set(ys.tolist())):
            idx = (ys == c).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            cls_imgs = imgs[idx]
            nrow = self.grid_nrow or cls_imgs.size(0)
            grid = make_grid(cls_imgs, nrow=nrow, padding=2)
            save_image(grid, os.path.join(grid_dir, f"class_{int(c)}.png"))
