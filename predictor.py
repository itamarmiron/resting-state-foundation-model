"""JEPA predictor: a small ViT that predicts target-ROI features from context-ROI features.

Follows Brain-JEPA's ``VisionTransformerPredictor`` (src/models/vision_transformer.py), with the
depth, width and heads of configs/ukb_vitb_ep300.yaml. One token per ROI replaces its ROI x time
patches, because the encoder pools time:
    1. project the context features to the predictor width and add their ROI position
    2. build one query per target ROI: a shared mask token + that ROI's position
    3. run self-attention over context and queries together
    4. normalize, keep the query tokens and project back to the encoder width

This replaces the "narrow BIT cross-transformer" written on slide 2. The slide's minute offset
(delta t) is not here yet, because the only targets so far are in the context's own window.

Positions always come from the fixed group gradients, also for an encoder trained with
``gradient_source="subject"``. Shape names: B batch, N ROIs, Kc context ROIs, Kt target ROIs,
D encoder output width, P predictor width, G gradient components.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from positional_embedding import GradientPositionalEmbedding
from temporal_bit import Block


@dataclass
class PredictorConfig:
    embed_dim: int = 512            # D, must equal TemporalBITConfig.out_dim
    predictor_dim: int = 384        # P, Brain-JEPA pred_emb_dim
    depth: int = 6                  # Brain-JEPA pred_depth
    heads: int = 12                 # Brain-JEPA's predictor default
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    drop_path: float = 0.0          # largest rate; grows linearly with depth
    grad_dim: int = 30              # G
    init_std: float = 0.02


class Predictor(nn.Module):
    """(B, Kc, D) context features -> (B, Kt, D) predicted target features."""

    def __init__(self, cfg: PredictorConfig, group_gradients: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        group_gradients = torch.as_tensor(group_gradients, dtype=torch.float32)
        assert group_gradients.ndim == 2 and group_gradients.shape[1] == cfg.grad_dim
        self.register_buffer("group_gradients", group_gradients)                # (N, G)

        self.embed = nn.Linear(cfg.embed_dim, cfg.predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.predictor_dim))
        self.grad_pe = GradientPositionalEmbedding(cfg.grad_dim, cfg.predictor_dim)
        rates = torch.linspace(0, cfg.drop_path, cfg.depth).tolist()
        self.blocks = nn.ModuleList([
            Block(cfg.predictor_dim, cfg.heads, cfg.mlp_ratio, cfg.dropout, rate) for rate in rates])
        self.norm = nn.LayerNorm(cfg.predictor_dim)
        self.proj = nn.Linear(cfg.predictor_dim, cfg.embed_dim)

        nn.init.trunc_normal_(self.mask_token, std=cfg.init_std)
        self.apply(self._init_weights)
        for layer, block in enumerate(self.blocks, start=1):                    # as in Brain-JEPA
            block.attn.proj.weight.data.div_(math.sqrt(2.0 * layer))
            block.mlp.net[3].weight.data.div_(math.sqrt(2.0 * layer))

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=self.cfg.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, context_rois: torch.Tensor, target_rois: torch.Tensor) -> torch.Tensor:
        """
        z:            (B, Kc, D)  context encoder output, in the order of context_rois
        context_rois: (B, Kc)     long, the ROIs of z
        target_rois:  (B, Kt)     long, the ROIs to predict
        returns       (B, Kt, D)  predictions, in the order of target_rois

        Steps:
            pe      = GradPE(group_gradients)            (N, P), min-max over the full atlas
            context = embed(z) + pe[context_rois]        (B, Kc, P)
            queries = mask_token + pe[target_rois]       (B, Kt, P)
            x       = blocks([context, queries])         (B, Kc + Kt, P)
            out     = proj(norm(x)[:, Kc:])              (B, Kt, D)
        """
        B, Kc, D = z.shape
        Kt = target_rois.shape[1]
        assert D == self.cfg.embed_dim, f"expected features of width {self.cfg.embed_dim}, got {D}"
        assert context_rois.shape == (B, Kc) and target_rois.shape == (B, Kt)

        pe = self.grad_pe(self.group_gradients)                                 # (N, P)
        context = self.embed(z) + pe[context_rois]                              # (B, Kc, P)
        queries = self.mask_token + pe[target_rois]                             # (B, Kt, P)
        x = torch.cat([context, queries], dim=1)                                # (B, Kc + Kt, P)
        for block in self.blocks:
            x = block(x)
        out = self.proj(self.norm(x)[:, Kc:])                                   # (B, Kt, D)
        assert out.shape == (B, Kt, D)
        return out
