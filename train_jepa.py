"""JEPA pretraining of Temporal BIT on Cross-ROI targets (slide 2 of Temporal_BIT_JEPA_widescreen.pdf).

    python train_jepa.py --config configs/pretraining/hcp.yaml [--max-steps N]

One batch: the target encoder sees the whole window, the context encoder only the context ROIs, the
predictor predicts the target ROIs, and the loss is smooth L1 between predictions and targets.

Follows Brain-JEPA's src/train.py and src/helper.py, with the values of configs/ukb_vitb_ep300.yaml:
    - gradient accumulation: every batch divides its loss by ``accumulation_steps`` and calls
      backward(); every ``accumulation_steps`` batches come the optimizer step and the EMA update
    - AdamW; biases and 1-D parameters have no weight decay
    - LR: linear warmup, then cosine; weight decay: cosine. Both count optimizer steps

Deviations from Brain-JEPA:
    - The EMA momentum counts optimizer steps too, so it reaches its end value. Brain-JEPA's schedule
      is sized in batches but advanced on optimizer steps, so at 8 accumulation steps it stops near 0.9965.
    - No GradScaler: bf16 doesn't underflow the way fp16 does.
    - Non-finite losses and gradient norms stop the run, before they can reach the weights.
    - One GPU; no DistributedDataParallel.

Everything is a function of the optimizer step, and checkpoints are written right after one, when no
partial gradients exist. So a cancelled job resumes from ``latest.pt`` by running the same command.
A resumed epoch reshuffles the windows instead of continuing the old order.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from ema_target_encoder import EMATargetEncoder, ema_momentum
from predictor import Predictor, PredictorConfig
from roi_masks import ROIMaskCollator, ROIMaskConfig
from temporal_bit import TemporalBIT, TemporalBITConfig
from window_dataset import WindowDataset


@dataclass
class TrainConfig:
    data_root: str = ""
    output_dir: str = ""
    subjects_file: str | None = None    # one subject per line; None = all stored runs
    batch_size: int = 16
    accumulation_steps: int = 8
    epochs: int = 300
    warmup_epochs: int = 40
    start_lr: float = 5e-5
    lr: float = 1e-3
    final_lr: float = 1e-6
    weight_decay: float = 0.04
    final_weight_decay: float = 0.4
    ema_start: float = 0.996
    ema_end: float = 1.0
    precision: str = "bf16"             # "bf16" or "fp32"
    seed: int = 0
    num_workers: int = 8
    log_every: int = 10                 # optimizer steps
    checkpoint_every: int = 500         # optimizer steps
    encoder: dict = field(default_factory=dict)      # TemporalBITConfig overrides
    predictor: dict = field(default_factory=dict)    # PredictorConfig overrides
    mask: dict = field(default_factory=dict)         # ROIMaskConfig overrides (placeholder values)


# These can change between a run and its resume; every other field defines the run.
RESUMABLE_FIELDS = {"data_root", "output_dir", "num_workers", "log_every", "checkpoint_every"}


# ----------------------------------------------------------------------------------------------
# Schedules and optimizer
# ----------------------------------------------------------------------------------------------

def lr_at(step: int, warmup_steps: int, total_steps: int, start: float, ref: float, final: float) -> float:
    """Linear from ``start`` to ``ref`` over the warmup, then cosine down to ``final``."""
    if step < warmup_steps:
        return start + step / warmup_steps * (ref - start)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return final + (ref - final) * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def weight_decay_at(step: int, total_steps: int, ref: float, final: float) -> float:
    """Cosine from ``ref`` to ``final``."""
    return final + (ref - final) * 0.5 * (1.0 + math.cos(math.pi * min(step / total_steps, 1.0)))


def build_optimizer(modules: list[nn.Module]) -> torch.optim.AdamW:
    """AdamW; biases and 1-D parameters (LayerNorm weights) are never decayed. LR and weight decay
    are set before every step."""
    decay, no_decay = [], []
    for module in modules:
        for name, param in module.named_parameters():
            (no_decay if "bias" in name or param.ndim == 1 else decay).append(param)
    return torch.optim.AdamW([{"params": decay, "decayed": True},
                              {"params": no_decay, "decayed": False, "weight_decay": 0.0}])


def set_schedules(optimizer: torch.optim.Optimizer, lr: float, weight_decay: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
        if group["decayed"]:
            group["weight_decay"] = weight_decay


# ----------------------------------------------------------------------------------------------
# One batch
# ----------------------------------------------------------------------------------------------

def jepa_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Smooth L1 (beta = 1, mean over all elements) as in Brain-JEPA; computed in fp32."""
    return F.smooth_l1_loss(prediction.float(), target.float())


def batch_step(encoder: TemporalBIT, predictor: Predictor, target: EMATargetEncoder, batch: dict,
               device: torch.device, accumulation_steps: int, precision: str) -> dict:
    """Forward and backward for one batch; gradients accumulate. Returns the unscaled loss and the
    collapse indicators: the std over ROI tokens per feature, 0 if all tokens were the same."""
    x = batch["x"].to(device, non_blocking=True)                                # (B, T, V)
    valid = batch["valid"].to(device, non_blocking=True)                        # (B, V)
    tr = batch["tr"].to(device, non_blocking=True)                              # (B,)
    context_rois = batch["context_rois"].to(device, non_blocking=True)          # (B, Kc)
    target_rois = batch["target_rois"].to(device, non_blocking=True)            # (B, Kt)

    # GPU only: CPU autocast rejects the float16 voxels in torch.cat.
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16" and device.type == "cuda"):
        h = target(x, target_rois, voxel_valid=valid, tr=tr)                    # (B, Kt, D)
        z = encoder(x, roi_keep=context_rois, voxel_valid=valid, tr=tr)         # (B, Kc, D)
        prediction = predictor(z, context_rois, target_rois)                    # (B, Kt, D)
    loss = jepa_loss(prediction, h)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite loss: {loss.item()}")
    (loss / accumulation_steps).backward()
    D = h.shape[-1]
    return {"loss": loss.item(),
            "target_std": h.detach().float().reshape(-1, D).std(0).mean().item(),
            "pred_std": prediction.detach().float().reshape(-1, D).std(0).mean().item()}


# ----------------------------------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------------------------------

def run_identity(cfg: TrainConfig) -> dict:
    return {k: v for k, v in asdict(cfg).items() if k not in RESUMABLE_FIELDS}


def save_checkpoint(path: Path, step: int, total_steps: int, cfg: TrainConfig, encoder: TemporalBIT,
                    predictor: Predictor, target: EMATargetEncoder, optimizer: torch.optim.Optimizer) -> None:
    """Written to a temporary file first, so a job cancelled mid-write leaves the old one intact."""
    tmp = path.with_suffix(".tmp")
    torch.save({"step": step, "total_steps": total_steps, "config": run_identity(cfg),
                "encoder": encoder.state_dict(), "predictor": predictor.state_dict(),
                "target_encoder": target.state_dict(), "optimizer": optimizer.state_dict()}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, total_steps: int, cfg: TrainConfig, encoder: TemporalBIT, predictor: Predictor,
                    target: EMATargetEncoder, optimizer: torch.optim.Optimizer, device: torch.device) -> int:
    """Restore a run and return its optimizer step. A checkpoint from different settings, or from a
    dataset of a different size (which would change the schedules), is refused."""
    ckpt = torch.load(path, map_location=device)
    if ckpt["config"] != run_identity(cfg):
        raise ValueError(f"{path} was written with other settings; use a new output_dir")
    if ckpt["total_steps"] != total_steps:
        raise ValueError(f"{path} expects {ckpt['total_steps']} optimizer steps, this dataset gives "
                         f"{total_steps}; use a new output_dir")
    encoder.load_state_dict(ckpt["encoder"])
    predictor.load_state_dict(ckpt["predictor"])
    target.load_state_dict(ckpt["target_encoder"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"]


# ----------------------------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------------------------

LOG_FIELDS = ["step", "epoch", "loss", "target_std", "pred_std", "grad_norm", "lr", "weight_decay",
              "momentum", "seconds_per_step", "peak_gpu_gib"]


def train(cfg: TrainConfig, dataset: WindowDataset, labels: np.ndarray, group_gradients: torch.Tensor,
          device: torch.device, stop_after: int | None = None) -> int:
    """Run or resume training in ``cfg.output_dir``; returns the last optimizer step. ``stop_after``
    ends the run early at that optimizer step without changing the schedules."""
    assert cfg.precision in ("bf16", "fp32"), cfg.precision
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest, log_path = output_dir / "latest.pt", output_dir / "train_log.csv"

    torch.manual_seed(cfg.seed)
    encoder_cfg = TemporalBITConfig(**cfg.encoder)
    assert encoder_cfg.gradient_source == "group", "training only supports the fixed group gradients"
    assert encoder_cfg.n_timepoints == dataset.n_timepoints
    predictor_cfg = PredictorConfig(**{"embed_dim": encoder_cfg.out_dim, "grad_dim": encoder_cfg.grad_dim,
                                       **cfg.predictor})
    encoder = TemporalBIT(encoder_cfg, labels, group_gradients).to(device).train()
    predictor = Predictor(predictor_cfg, group_gradients).to(device).train()
    target = EMATargetEncoder(encoder)
    optimizer = build_optimizer([encoder, predictor])

    mask_cfg = ROIMaskConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg.mask.items()})
    batches_per_epoch = len(dataset) // cfg.batch_size
    total_steps = cfg.epochs * batches_per_epoch // cfg.accumulation_steps
    warmup_steps = cfg.warmup_epochs * batches_per_epoch // cfg.accumulation_steps
    if total_steps == 0:
        raise ValueError(f"{len(dataset)} windows give no optimizer step at this batch size and accumulation")

    step = 0
    if latest.exists():
        step = load_checkpoint(latest, total_steps, cfg, encoder, predictor, target, optimizer, device)
        print(f"resumed from {latest} at optimizer step {step}", flush=True)
    last_step = total_steps if stop_after is None else min(stop_after, total_steps)
    print(f"{len(dataset)} windows, {batches_per_epoch} batches per epoch, {total_steps} optimizer steps "
          f"({warmup_steps} warmup), {cfg.precision}, device {device}", flush=True)

    # Seeded by the step, so a resumed run doesn't repeat the shuffles and masks it already used.
    torch.manual_seed(cfg.seed + step)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                        num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0,
                        pin_memory=device.type == "cuda", collate_fn=ROIMaskCollator(encoder.n_rois, mask_cfg),
                        generator=torch.Generator().manual_seed(cfg.seed + step))

    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as log_file:
        log = csv.DictWriter(log_file, LOG_FIELDS)
        if write_header:
            log.writeheader()

        accumulated, t0 = [], time.time()
        while step < last_step:
            for batch in loader:                # batches left over at an epoch's end carry into the next
                accumulated.append(batch_step(encoder, predictor, target, batch, device,
                                              cfg.accumulation_steps, cfg.precision))
                if len(accumulated) < cfg.accumulation_steps:
                    continue

                params = [p for group in optimizer.param_groups for p in group["params"]]
                grad_norm = nn.utils.clip_grad_norm_(params, math.inf).item()   # measures, doesn't clip
                if not math.isfinite(grad_norm):
                    raise FloatingPointError(f"non-finite gradient norm at optimizer step {step}")
                lr = lr_at(step, warmup_steps, total_steps, cfg.start_lr, cfg.lr, cfg.final_lr)
                wd = weight_decay_at(step, total_steps, cfg.weight_decay, cfg.final_weight_decay)
                set_schedules(optimizer, lr, wd)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                momentum = ema_momentum(step, total_steps, cfg.ema_start, cfg.ema_end)
                target.update(encoder, momentum)
                step += 1

                if step % cfg.log_every == 0 or step == last_step:
                    seconds = (time.time() - t0) / cfg.log_every if step % cfg.log_every == 0 else float("nan")
                    row = {k: float(np.mean([a[k] for a in accumulated])) for k in accumulated[0]}
                    row.update(step=step, epoch=step * cfg.accumulation_steps / batches_per_epoch,
                               grad_norm=grad_norm, lr=lr, weight_decay=wd, momentum=momentum,
                               seconds_per_step=seconds,
                               peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30
                               if device.type == "cuda" else float("nan"))
                    log.writerow(row)
                    log_file.flush()
                    print(" ".join(f"{k}={v:.4g}" for k, v in row.items()), flush=True)
                    t0 = time.time()
                accumulated = []
                if step % cfg.checkpoint_every == 0 or step == last_step:
                    save_checkpoint(latest, step, total_steps, cfg, encoder, predictor, target, optimizer)
                if step == last_step:
                    return step
    return step


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None, help="stop at this optimizer step (for tests)")
    args = parser.parse_args(argv)
    with open(args.config) as f:
        cfg = TrainConfig(**yaml.safe_load(f))

    import compute_gradients as cg                   # imports nilearn; only the real run needs it
    from positional_embedding import load_gradients

    subjects = None
    if cfg.subjects_file is None:
        print("WARNING: no subjects_file, so this trains on every stored run, with no held-out subjects", flush=True)
    else:
        subjects = Path(cfg.subjects_file).read_text().split()
    dataset = WindowDataset(cfg.data_root, TemporalBITConfig(**cfg.encoder).n_timepoints, subjects=subjects)
    atlas = cg.load_atlas()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(cfg, dataset, atlas.labels, load_gradients(cg.group_gradient_path()), device, stop_after=args.max_steps)


if __name__ == "__main__":
    main()
