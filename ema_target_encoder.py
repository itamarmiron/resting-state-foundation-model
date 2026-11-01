"""EMA target encoder for JEPA pretraining: a frozen copy of Temporal BIT that follows the context
encoder as an exponential moving average and produces the regression targets.

Follows Brain-JEPA's src/train.py: the copy never receives gradients, it sees all ROIs, its output
goes through a parameter-free LayerNorm over the feature dimension, and the target ROIs are read
from that. The momentum grows linearly from 0.996 to 1.0 (configs/ukb_vitb_ep300.yaml).

One deviation: Brain-JEPA leaves its target encoder in train mode, which makes no difference at its
zero dropout. Here it is always in eval mode, so targets stay deterministic if dropout is enabled.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from temporal_bit import TemporalBIT


def ema_momentum(step: int, total_steps: int, start: float = 0.996, end: float = 1.0) -> float:
    """Momentum at an optimizer step: linear from ``start`` to ``end``, then held at ``end``."""
    assert total_steps > 0 and step >= 0
    return start + min(step / total_steps, 1.0) * (end - start)


class EMATargetEncoder(nn.Module):
    """(B, T, V) voxel window -> (B, Kt, D) normalized features of the target ROIs."""

    def __init__(self, encoder: TemporalBIT):
        super().__init__()
        self.encoder = copy.deepcopy(encoder)
        self.encoder.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True) -> "EMATargetEncoder":
        """Stay in eval mode whatever a parent module's ``.train()`` asks for."""
        return super().train(False)

    @torch.no_grad()
    def update(self, encoder: TemporalBIT, momentum: float) -> None:
        """target = momentum * target + (1 - momentum) * encoder, after each optimizer step.

        Only parameters are averaged: the model's buffers are fixed (atlas tables, gradients).
        """
        assert 0.0 <= momentum <= 1.0
        source = dict(encoder.named_parameters())
        target = dict(self.encoder.named_parameters())
        assert source.keys() == target.keys(), "target and context encoders have different parameters"
        for name, param in target.items():
            param.mul_(momentum).add_(source[name].detach(), alpha=1.0 - momentum)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, target_rois: torch.Tensor | None = None,
                voxel_valid: torch.Tensor | None = None, tr: float | torch.Tensor | None = None,
                gradients: torch.Tensor | None = None) -> torch.Tensor:
        """
        x:           (B, T, V)  the full window, with no ROI removed
        target_rois: (B, Kt)    long, ROIs to return; None = all N
        voxel_valid, tr, gradients: as in ``TemporalBIT.forward``
        returns      (B, Kt, D) or (B, N, D), LayerNorm over D

        The encoder always runs on all ROIs: ``target_rois`` only selects from its output.
        """
        h = self.encoder(x, roi_keep=None, voxel_valid=voxel_valid, tr=tr, gradients=gradients)
        h = F.layer_norm(h, (h.shape[-1],))                                     # (B, N, D)
        if target_rois is None:
            return h
        return torch.gather(h, 1, target_rois[..., None].expand(-1, -1, h.shape[-1]))
