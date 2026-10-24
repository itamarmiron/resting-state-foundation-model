"""Positional embeddings for Temporal BIT, ported and adapted from Brain-JEPA.

Source: /home/itamarm/ptsd_project/rsPTSD/Brain-JEPA/src/models/vision_transformer.py
    - ``get_1d_sincos_pos_embed_from_grid``  (module-level function; same code as the method
      ``GradTs_2dPE.get_1d_sincos_pos_embed_from_grid``)
    - ``GradTs_2dPE`` with ``add_w='mapping'`` (the setting in configs/ukb_vitb_ep300.yaml)

What changed relative to Brain-JEPA:
    - Time and ROI are separate. Brain-JEPA builds one (N_roi * 10, D) table for its fixed grid of
      10 temporal patches (``repeat_interleave(10)``). Here the temporal part is computed for any
      T, and the gradient part is per ROI.
    - Time is continuous: position = t * TR / TR_ref. With TR == TR_ref (HCP, 0.72 s) the positions
      are the integer indices 0..T-1, exactly as in Brain-JEPA; other TRs give fractional positions.
    - The gradient projection accepts a fixed (N, G) matrix (one population-level gradient set) or a
      per-sample (B, N, G) tensor (subject-level gradients).

Brain gradients themselves are not computed here; see ``compute_gradients.py``.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """Sin-cos embedding of arbitrary (possibly fractional) positions: (...,) -> (..., embed_dim).

    Port of Brain-JEPA ``get_1d_sincos_pos_embed_from_grid`` (numpy -> torch, any leading shape):
    omega_i = 1 / 10000^(2i / D), emb = [sin(pos * omega), cos(pos * omega)].
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float64, device=pos.device)
    omega = 1.0 / 10000 ** (omega / (embed_dim / 2.0))                 # (D/2,)
    out = pos.to(torch.float64)[..., None] * omega                      # (..., D/2)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=-1).float()  # (..., D)


def temporal_sincos(n_timepoints: int, embed_dim: int, tr: torch.Tensor, tr_ref: float) -> torch.Tensor:
    """Temporal positional embedding for a window of T TRs, time measured in reference TRs.

    tr: (B,) repetition time of each sample in seconds.
    Returns (B, T, embed_dim). Fixed, no parameters.
    """
    assert tr.ndim == 1
    steps = torch.arange(n_timepoints, dtype=torch.float64, device=tr.device)   # (T,)
    pos = steps[None, :] * tr.to(torch.float64)[:, None] / tr_ref               # (B, T)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)                    # (B, T, D)


class GradientPositionalEmbedding(nn.Module):
    """Brain Gradient Positioning: learned projection of each ROI's gradient coordinates.

    Port of the ``add_w='mapping'`` branch of Brain-JEPA ``GradTs_2dPE.forward``:
        emb = Linear(G -> D)(gradient)
        emb = (emb - emb.min()) / (emb.max() - emb.min()) * 2 - 1      # global min-max to [-1, 1]
    The min-max runs over all ROIs and channels of one gradient set, i.e. over the full atlas,
    *before* any ROI is dropped, so masking never changes the embedding of a kept ROI.
    """

    def __init__(self, grad_dim: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(grad_dim, embed_dim)

    def forward(self, gradients: torch.Tensor) -> torch.Tensor:
        """gradients: (N, G) fixed, or (B, N, G) per sample -> (N, D) or (B, N, D)."""
        assert gradients.ndim in (2, 3)
        emb = self.proj(gradients)                                      # (..., N, D)
        lo = emb.amin(dim=(-2, -1), keepdim=True)                       # one min/max per gradient set
        hi = emb.amax(dim=(-2, -1), keepdim=True)
        return (emb - lo) / (hi - lo).clamp_min(1e-6) * 2 - 1


def load_gradients(path: str) -> torch.Tensor:
    """Load an (N, G) gradient matrix saved by ``compute_gradients.py`` (.npy) as float32."""
    grad = np.load(path)
    assert grad.ndim == 2 and np.isfinite(grad).all(), f"{path}: expected a finite (N, G) matrix"
    return torch.from_numpy(grad.astype(np.float32))
