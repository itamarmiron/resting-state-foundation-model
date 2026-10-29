"""ROI masks for JEPA pretraining on one window: which ROIs the context encoder sees, which ROIs
the predictor must predict.

This is the Cross-ROI ("R") target of slide 2 of Temporal_BIT_JEPA_widescreen.pdf: masked ROIs of
the same window. The Cross-time ("T") and Double-cross ("D") targets need several windows of
one scan and come later.

Ratios are Brain-JEPA's (src/masks/spatialtemporal_multiblock.py ``MaskCollator_fmri`` with
configs/ukb_vitb_ep300.yaml, which overrides two class defaults):
    target_scale  = pred_mask_R_scale = (0.45, 0.6)   fraction of all ROIs hidden as targets
    context_scale = enc_mask_scale    = (0.84, 1.0)   fraction kept by the observation block
    min_context   = min_keep          = 4
Brain-JEPA picks both sets with ``randperm`` (random ROIs, not contiguous blocks), samples the
two sizes once per batch and the ROIs per sample, and makes the context the observation block
minus the target ROIs. Its time-axis parts (target time span, the beta/gamma blocks) have no
counterpart here because the encoder pools time into one token per ROI.

One deviation: Brain-JEPA's observation block is drawn independently of the targets, so the
context size varies per sample and every sample's index list is truncated to the batch minimum.
Its indices are sorted, so the truncation drops high-index ROIs (Schaefer, here) first. Here the
context is drawn directly from the non-target ROIs, int(context_scale * (N - n_target)) of them:
the same expected size, an exact size per batch, and no truncation. ``context_scale=(1, 1)``
gives slide 2's version, where every ROI that is not a target is visible.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import default_collate


@dataclass
class ROIMaskConfig:
    target_scale: tuple[float, float] = (0.45, 0.6)    # Brain-JEPA pred_mask_R_scale
    context_scale: tuple[float, float] = (0.84, 1.0)   # Brain-JEPA enc_mask_scale
    min_context: int = 4                               # Brain-JEPA min_keep


class ROIMaskSampler:
    """Disjoint context and target ROI sets for a batch.

    Sizes are sampled once per call (so every sample in the batch has the same K), the ROIs
    per sample. Without a ``generator`` the global torch RNG is used, which DataLoader seeds
    per worker.
    """

    def __init__(self, n_rois: int, cfg: ROIMaskConfig | None = None):
        self.n_rois = n_rois
        self.cfg = cfg or ROIMaskConfig()
        lo, hi = self.cfg.target_scale
        assert 0 < lo <= hi < 1, "target_scale must leave some ROIs for the context"
        lo, hi = self.cfg.context_scale
        assert 0 < lo <= hi <= 1
        assert 0 < self.cfg.min_context < n_rois

    def sample_sizes(self, generator: torch.Generator | None = None) -> tuple[int, int]:
        """(n_context, n_target) for one batch."""
        N, cfg = self.n_rois, self.cfg
        u = torch.rand(2, generator=generator).tolist()
        s_t = cfg.target_scale[0] + u[0] * (cfg.target_scale[1] - cfg.target_scale[0])
        s_c = cfg.context_scale[0] + u[1] * (cfg.context_scale[1] - cfg.context_scale[0])
        n_target = max(1, int(N * s_t))
        n_target = min(n_target, N - cfg.min_context)
        n_context = max(cfg.min_context, int((N - n_target) * s_c))
        return n_context, n_target

    def __call__(self, batch_size: int, generator: torch.Generator | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns context (B, Kc) and target (B, Kt) ROI indices, long, sorted per row, disjoint."""
        n_context, n_target = self.sample_sizes(generator)
        perm = torch.rand(batch_size, self.n_rois, generator=generator).argsort(dim=1)  # (B, N)
        target = perm[:, :n_target].sort(dim=1).values
        context = perm[:, n_target:n_target + n_context].sort(dim=1).values
        return context, target


class ROIMaskCollator:
    """DataLoader ``collate_fn``: the default collate of ``WindowDataset`` items plus
    ``context_rois`` (B, Kc) and ``target_rois`` (B, Kt).

    The context encoder takes ``roi_keep=batch["context_rois"]``; the target encoder sees all
    ROIs and its output is read at ``target_rois``.
    """

    def __init__(self, n_rois: int, cfg: ROIMaskConfig | None = None):
        self.sampler = ROIMaskSampler(n_rois, cfg)

    def __call__(self, items: list[dict]) -> dict:
        batch = default_collate(items)
        batch["context_rois"], batch["target_rois"] = self.sampler(len(items))
        return batch
