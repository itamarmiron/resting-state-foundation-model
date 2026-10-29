"""Temporal BIT: an fMRI encoder from a window of T TRs of voxel-level BOLD to N x D ROI features.

Architecture (slide 1 of Temporal_BIT_JEPA_widescreen.pdf):
    1. BrainTokenizer        per TR, shared weights   (B, T, V)       -> (B, T, N, d)
       Each voxel's activation scales its learned voxel embedding; a single-head graph attention
       with learned ROI embeddings as queries pools each ROI's voxels (V2ROI mask) into one token.
    2. TemporalTransformer   per ROI, shared weights  (B, T, N, d)    -> (B, N, d)
       Self-attention over the T tokens of one ROI, then attention pooling over time.
       ROIs never interact here.
    3. SpatialTransformer    across ROIs              (B, N, d)       -> (B, N, D)
       L blocks of self-attention over the ROI tokens + FFN, then a projection to D.

The tokenizer is inspired by BIT in Brain-IT (models/decoder_models.py ``Decoder``: per-voxel
embeddings modulated by the activation, pooled by torch_geometric ``TransformerConv`` onto cluster
"centers"), re-derived here for one shared MNI voxel space and without materializing the
(B, T, V, d) modulated voxels. See ``BrainTokenizer`` for the algebra.

Voxel embeddings: by default each labelled voxel has its own free d-dim embedding, as in Brain-IT
(whose table is (sum of subjects' voxels, 1024), full rank). ``voxel_rank=r`` instead factorizes the
table as codes (V', r) @ W_up (r, d), our addition: ~8x fewer parameters, but all voxel embeddings
then share one r-dim subspace.

ROI masking: ``roi_keep`` selects the ROIs to encode. Dropped ROIs are removed completely (their
voxels are never read, they get no temporal pass and no token), and nothing mixes information
across ROIs before the spatial transformer.

Shape names: B batch, T TRs, V stored voxels, N ROIs, K kept ROIs, P max voxels per ROI,
d model width, r voxel-embedding rank, D output width, G gradient components.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from positional_embedding import GradientPositionalEmbedding, temporal_sincos


@dataclass
class TemporalBITConfig:
    n_timepoints: int = 20          # T, TRs per input window
    tr: float = 0.72                # default TR (s) when forward() gets no ``tr``
    tr_ref: float = 0.72            # temporal PE position = t * TR / tr_ref
    d_model: int = 512              # d, width of voxel/ROI tokens in all three stages
    voxel_rank: int | None = None   # None: free (V', d) voxel embeddings (Brain-IT);
                                    # r: low rank, codes (V', r) @ W_up (r, d)
    temporal_depth: int = 2
    temporal_heads: int = 8
    temporal_pool: str = "attn"     # "attn" (learned query) or "mean"
    spatial_depth: int = 6          # L
    spatial_heads: int = 8
    mlp_ratio: float = 4.0
    out_dim: int = 512              # D
    dropout: float = 0.0
    drop_path: float = 0.0
    grad_dim: int = 30              # G, brain gradient components
    gradient_source: str = "group"  # "group": fixed gradients; "subject": per-sample input
    subject_gradients_in_temporal: bool = True   # False: subject gradients only before the spatial stage
    init_std: float = 0.02


# ----------------------------------------------------------------------------------------------
# V2ROI: voxel -> ROI mapping as a padded table
# ----------------------------------------------------------------------------------------------

@dataclass
class V2ROI:
    """Padded voxel->ROI table built from one atlas label per stored voxel.

    pad_index: (N, P) index into the stored-voxel axis V; padding slots hold V (a zero column).
    pad_row:   (N, P) row of the voxel-embedding table (labelled voxels only); padding holds 0.
    pad_mask:  (N, P) True for real voxels.
    """
    pad_index: torch.Tensor
    pad_row: torch.Tensor
    pad_mask: torch.Tensor
    n_voxels: int           # V, stored voxels (all of them, labelled or not)
    n_labelled: int         # V', voxels that belong to an ROI

    @property
    def n_rois(self) -> int:
        return self.pad_index.shape[0]


def build_v2roi(labels: np.ndarray, n_rois: int | None = None) -> V2ROI:
    """labels: (V,) int, ROI r is label r + 1 and 0 means "no ROI" (the voxel is never used).

    This is ``Atlas.labels`` from rsfm.preprocessing.atlas (``atlas/<name>/labels.npy``).
    """
    labels = np.asarray(labels)
    assert labels.ndim == 1 and labels.min() >= 0
    n_rois = int(labels.max()) if n_rois is None else n_rois
    voxels = np.flatnonzero(labels > 0)                     # (V',) in stored (C) order
    rows = np.arange(len(voxels))                           # row j of the embedding table = voxels[j]
    roi = labels[voxels] - 1
    order = np.argsort(roi, kind="stable")                  # group voxels by ROI
    voxels, rows, roi = voxels[order], rows[order], roi[order]
    counts = np.bincount(roi, minlength=n_rois)
    slot = np.arange(len(roi)) - np.repeat(np.cumsum(counts) - counts, counts)

    P = int(counts.max())
    pad_index = np.full((n_rois, P), len(labels), dtype=np.int64)
    pad_row = np.zeros((n_rois, P), dtype=np.int64)
    pad_mask = np.zeros((n_rois, P), dtype=bool)
    pad_index[roi, slot] = voxels
    pad_row[roi, slot] = rows
    pad_mask[roi, slot] = True
    return V2ROI(torch.from_numpy(pad_index), torch.from_numpy(pad_row), torch.from_numpy(pad_mask),
                 n_voxels=len(labels), n_labelled=len(voxels))


# ----------------------------------------------------------------------------------------------
# Transformer building blocks (pre-LN)
# ----------------------------------------------------------------------------------------------

def drop_path(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Stochastic depth: drop the whole residual branch per sample."""
    if p == 0.0 or not training:
        return x
    keep = torch.rand(x.shape[0], *([1] * (x.ndim - 1)), device=x.device) >= p
    return x * keep / (1 - p)


class Attention(nn.Module):
    """Multi-head attention; self-attention when ``context`` is None, else cross-attention."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.dropout = heads, dropout
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        """x: (S, Lq, dim), context: (S, Lk, dim) or None -> (S, Lq, dim)."""
        context = x if context is None else context
        q = rearrange(self.q(x), "s l (h c) -> s h l c", h=self.heads)
        k, v = rearrange(self.kv(context), "s l (two h c) -> two s h l c", two=2, h=self.heads)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(rearrange(out, "s h l c -> s l (h c)"))


class MLP(nn.Module):
    def __init__(self, dim: int, ratio: float, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * ratio)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Pre-LN transformer block: x + Attn(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float, drop_path: float):
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(dim), Attention(dim, heads, dropout)
        self.norm2, self.mlp = nn.LayerNorm(dim), MLP(dim, mlp_ratio, dropout)
        self.drop_path = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (S, L, dim) -> (S, L, dim); attention runs over L only."""
        x = x + drop_path(self.attn(self.norm1(x)), self.drop_path, self.training)
        x = x + drop_path(self.mlp(self.norm2(x)), self.drop_path, self.training)
        return x


class AttentionPool(nn.Module):
    """Pool a sequence into one vector: a learned query cross-attends over it, then an MLP."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.norm_kv, self.attn = nn.LayerNorm(dim), Attention(dim, heads, dropout)
        self.norm_mlp, self.mlp = nn.LayerNorm(dim), MLP(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (S, L, dim) -> (S, dim)."""
        y = self.attn(self.query.expand(x.shape[0], -1, -1), context=self.norm_kv(x))  # (S, 1, dim)
        y = y + self.mlp(self.norm_mlp(y))
        return y[:, 0]


# ----------------------------------------------------------------------------------------------
# 1. Brain tokenizer
# ----------------------------------------------------------------------------------------------

class BrainTokenizer(nn.Module):
    """Pool each ROI's voxels into one token per TR with single-head graph attention.

    For ROI r with voxels j in r (V2ROI) and activation a_tj at TR t:
        e_j     = voxel embedding, one per labelled voxel, shared by all subjects and TRs:
                  a free row of voxel_embed (V', d), or W_up z_j if low rank (z_j in R^r)
        q_r     = W_q u_r                               u_r = learned ROI embedding (N x d)
        s_tj    = q_r . W_k (a_tj e_j) / sqrt(d)        key of the modulated voxel a_tj e_j
                = a_tj * c_j,   c_j = q_r . W_k e_j / sqrt(d)
        alpha_t = softmax_{j in r}(s_tj)
        tok_tr  = W_v sum_j alpha_tj a_tj e_j + W_root u_r
               (= W_v W_up (sum_j alpha_tj a_tj z_j) + W_root u_r if low rank)
    Keys and values are linear in the scalar activation, so the per-voxel coefficient c_j is
    computed once per forward and each TR only needs a softmax over scalars and a weighted sum
    of embeddings per ROI (r-dimensional if low rank). This is BIT's TransformerConv (heads=1,
    root_weight=True); the key and value biases are dropped because a key bias cancels in the
    softmax and a value bias is a constant offset.

    Full-rank memory: the padded (N, P, d) embedding block (~0.9 GB fp32 for the HCP atlas) is
    built once per forward and shared by the batch, never gathered per sample.

    No LayerNorm is applied to the modulated voxels: LN(a * e) = sign(a) * LN(e) would erase the
    activation magnitude. The ROI token is normalized after pooling.
    """

    def __init__(self, cfg: TemporalBITConfig, v2roi: V2ROI):
        super().__init__()
        d, r = cfg.d_model, cfg.voxel_rank
        self.d_model, self.n_voxels, self.n_rois = d, v2roi.n_voxels, v2roi.n_rois
        self.low_rank = r is not None
        self.register_buffer("pad_index", v2roi.pad_index, persistent=False)   # (N, P)
        self.register_buffer("pad_row", v2roi.pad_row, persistent=False)       # (N, P)
        self.register_buffer("pad_mask", v2roi.pad_mask, persistent=False)     # (N, P)

        if self.low_rank:
            self.voxel_codes = nn.Parameter(torch.randn(v2roi.n_labelled, r))  # z, (V', r)
            self.voxel_up = nn.Linear(r, d, bias=False)                        # W_up
        else:
            self.voxel_embed = nn.Parameter(cfg.init_std * torch.randn(v2roi.n_labelled, d))  # e, (V', d)
        self.roi_embed = nn.Parameter(cfg.init_std * torch.randn(v2roi.n_rois, d))  # u, (N, d)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.root = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, roi_keep: torch.Tensor,
                voxel_valid: torch.Tensor | None = None) -> torch.Tensor:
        """
        x:           (B, T, V)  BOLD, any float dtype (stored data is float16)
        roi_keep:    (B, K)     long, ROIs to tokenize
        voxel_valid: (B, V)     bool, False for voxels outside this run's brain mask
        returns      (B, T, K, d)

        Steps:
            idx   = pad_index[roi_keep]                    (B, K, P)   voxel ids of kept ROIs only
            a     = x gathered at idx                      (B, T, K, P)
            real  = pad_mask & voxel_valid                 (B, K, P)
            c     = e . (W_k^T W_q u) / sqrt(d)            (B, K, P)
            alpha = softmax_P(a * c), masked by real       (B, T, K, P)
            h     = sum_P alpha * a * e                    (B, T, K, d)
            tok   = LN(W_v h + W_root u)                   (B, T, K, d)
        Full rank: e = voxel_embed[pad_row] is (N, P, d), shared by the batch; c is computed for
        all N ROIs and h by scattering the kept weights into N slots (dropped ROIs stay 0).
        Low rank: z = voxel_codes[pad_row[roi_keep]] is (B, K, P, r), c = z . (W_up^T W_k^T W_q u)
        and h = W_up sum_P alpha * a * z, so nothing d-wide is built per voxel.
        """
        B, T, V = x.shape
        K = roi_keep.shape[1]
        P = self.pad_index.shape[1]
        assert V == self.n_voxels, f"expected {self.n_voxels} voxels, got {V}"
        assert roi_keep.shape == (B, K)

        idx = self.pad_index[roi_keep]                                          # (B, K, P)
        real = self.pad_mask[roi_keep]                                          # (B, K, P)
        x_ext = torch.cat([x, x.new_zeros(B, T, 1)], dim=2)                    # (B, T, V + 1)
        a = torch.gather(x_ext, 2, idx.reshape(B, 1, K * P).expand(B, T, K * P))
        a = a.reshape(B, T, K, P).float()                                      # (B, T, K, P)
        if voxel_valid is not None:
            assert voxel_valid.shape == (B, V)
            valid_ext = torch.cat([voxel_valid, voxel_valid.new_zeros(B, 1)], dim=1)   # (B, V + 1)
            real = real & torch.gather(valid_ext, 1, idx.reshape(B, K * P)).reshape(B, K, P)

        u = self.roi_embed[roi_keep]                                            # (B, K, d)
        if self.low_rank:
            z = self.voxel_codes[self.pad_row[roi_keep]]                        # (B, K, P, r)
            q_code = self.q(u) @ self.k.weight @ self.voxel_up.weight          # (B, K, r)
            c = torch.einsum("bkpr,bkr->bkp", z, q_code) / self.d_model ** 0.5  # (B, K, P)
        else:
            e = self.voxel_embed[self.pad_row]                                  # (N, P, d)
            q_key = self.q(self.roi_embed) @ self.k.weight                      # (N, d)
            c = torch.einsum("npd,nd->np", e, q_key)[roi_keep] / self.d_model ** 0.5  # (B, K, P)

        scores = a * c[:, None]                                                 # (B, T, K, P)
        scores = scores.masked_fill(~real[:, None], float("-inf"))
        # An ROI with no valid voxel would softmax over all -inf (NaN); give it finite scores
        # and zero its weights below, so it contributes only the root term.
        empty = ~real.any(dim=-1)                                               # (B, K)
        scores = scores.masked_fill(empty[:, None, :, None], 0.0)
        alpha = torch.softmax(scores, dim=-1) * real[:, None]                   # (B, T, K, P)

        w = alpha * a                                                           # (B, T, K, P)
        if self.low_rank:
            h = self.voxel_up(torch.einsum("btkp,bkpr->btkr", w, z))           # (B, T, K, d)
        else:
            slots = roi_keep[:, None, :, None]                                  # (B, 1, K, 1)
            w_all = w.new_zeros(B, T, self.n_rois, P).scatter(2, slots.expand(B, T, K, P), w)
            h_all = torch.einsum("btnp,npd->btnd", w_all, e)                    # (B, T, N, d)
            h = torch.gather(h_all, 2, slots.expand(B, T, K, self.d_model))    # (B, T, K, d)
        tok = self.v(h) + self.root(u)[:, None]                                 # (B, T, K, d)
        tok = self.norm(tok)
        assert tok.shape == (B, T, K, self.d_model)
        return tok


# ----------------------------------------------------------------------------------------------
# 2. Temporal transformer
# ----------------------------------------------------------------------------------------------

class TemporalTransformer(nn.Module):
    """Self-attention over the T tokens of each ROI separately, then pooling over time.

    ROIs are folded into the batch axis, so attention, LayerNorm and pooling never see two ROIs
    at once. The positional embedding follows Brain-JEPA's GradTs_2dPE layout:
    [temporal sin-cos (d/2) | brain gradient embedding (d/2)], so the shared weights know both
    when each token is and which ROI they are processing.
    """

    def __init__(self, cfg: TemporalBITConfig):
        super().__init__()
        d = cfg.d_model
        assert d % 4 == 0, "d/2 must be even for the sin-cos half"
        self.d_model = d
        self.blocks = nn.ModuleList([
            Block(d, cfg.temporal_heads, cfg.mlp_ratio, cfg.dropout, cfg.drop_path)
            for _ in range(cfg.temporal_depth)])
        self.norm = nn.LayerNorm(d)
        self.pool = (AttentionPool(d, cfg.temporal_heads, cfg.mlp_ratio, cfg.dropout)
                     if cfg.temporal_pool == "attn" else None)
        assert cfg.temporal_pool in ("attn", "mean")

    def forward(self, tokens: torch.Tensor, time_pe: torch.Tensor, grad_pe: torch.Tensor) -> torch.Tensor:
        """
        tokens:  (B, T, K, d)   ROI tokens per TR
        time_pe: (B, T, d/2)    temporal sin-cos
        grad_pe: (B, K, d/2)    gradient embedding of each kept ROI
        returns  (B, K, d)

        Steps:
            pe     = [time_pe | grad_pe] broadcast       (B, T, K, d)
            x      = tokens + pe -> (B*K, T, d)
            blocks                                       (B*K, T, d)
            pool over T                                  (B*K, d) -> (B, K, d)
        """
        B, T, K, d = tokens.shape
        assert time_pe.shape == (B, T, d // 2) and grad_pe.shape == (B, K, d // 2)
        pe = torch.cat([time_pe[:, :, None].expand(B, T, K, d // 2),
                        grad_pe[:, None].expand(B, T, K, d // 2)], dim=-1)    # (B, T, K, d)
        x = rearrange(tokens + pe, "b t k d -> (b k) t d")                     # (B*K, T, d)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.pool(x) if self.pool is not None else x.mean(dim=1)           # (B*K, d)
        x = rearrange(x, "(b k) d -> b k d", b=B)
        assert x.shape == (B, K, d)
        return x


# ----------------------------------------------------------------------------------------------
# 3. Spatial transformer
# ----------------------------------------------------------------------------------------------

class SpatialTransformer(nn.Module):
    """L blocks of self-attention across the ROI tokens (+ FFN), then a projection to D.

    This is the only stage in which ROIs exchange information. There are no query tokens:
    K ROI tokens in, K ROI features out.
    """

    def __init__(self, cfg: TemporalBITConfig):
        super().__init__()
        d = cfg.d_model
        self.blocks = nn.ModuleList([
            Block(d, cfg.spatial_heads, cfg.mlp_ratio, cfg.dropout, cfg.drop_path)
            for _ in range(cfg.spatial_depth)])
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, cfg.out_dim)

    def forward(self, x: torch.Tensor, grad_pe: torch.Tensor) -> torch.Tensor:
        """
        x:       (B, K, d)   one token per kept ROI
        grad_pe: (B, K, d)   full-width gradient embedding of each kept ROI
        returns  (B, K, D)
        """
        B, K, d = x.shape
        assert grad_pe.shape == (B, K, d)
        x = x + grad_pe
        for block in self.blocks:
            x = block(x)                                                        # (B, K, d)
        x = self.proj(self.norm(x))                                             # (B, K, D)
        return x


# ----------------------------------------------------------------------------------------------
# Full model
# ----------------------------------------------------------------------------------------------

class TemporalBIT(nn.Module):
    """Temporal BIT encoder: (B, T, V) voxel BOLD window -> (B, K, D) ROI features.

    Construction needs the atlas (``labels``, one ROI label per stored voxel) and the fixed
    population-level brain gradients (``group_gradients``, (N, G)). With
    ``cfg.gradient_source == "subject"`` the forward also takes per-sample gradients.
    """

    def __init__(self, cfg: TemporalBITConfig, labels: np.ndarray, group_gradients: torch.Tensor):
        super().__init__()
        assert cfg.gradient_source in ("group", "subject")
        self.cfg = cfg
        v2roi = build_v2roi(labels)
        self.n_rois = v2roi.n_rois
        group_gradients = torch.as_tensor(group_gradients, dtype=torch.float32)
        assert group_gradients.shape == (self.n_rois, cfg.grad_dim), \
            f"gradients {tuple(group_gradients.shape)} do not match N={self.n_rois}, G={cfg.grad_dim}"
        self.register_buffer("group_gradients", group_gradients)               # (N, G)

        self.tokenizer = BrainTokenizer(cfg, v2roi)
        self.temporal_grad_pe = GradientPositionalEmbedding(cfg.grad_dim, cfg.d_model // 2)
        self.temporal = TemporalTransformer(cfg)
        self.spatial_grad_pe = GradientPositionalEmbedding(cfg.grad_dim, cfg.d_model)
        self.spatial = SpatialTransformer(cfg)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=self.cfg.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _roi_keep(self, roi_keep: torch.Tensor | None, B: int, device) -> torch.Tensor:
        """None -> all ROIs; (K,) -> the same ROIs for every sample; (B, K) as is."""
        if roi_keep is None:
            roi_keep = torch.arange(self.n_rois, device=device)
        roi_keep = torch.as_tensor(roi_keep, device=device).long()
        if roi_keep.ndim == 1:
            roi_keep = roi_keep[None].expand(B, -1)
        assert roi_keep.ndim == 2 and roi_keep.shape[0] == B
        assert 0 <= int(roi_keep.min()) and int(roi_keep.max()) < self.n_rois
        return roi_keep

    def _gradient_pe(self, module: GradientPositionalEmbedding, gradients: torch.Tensor,
                     roi_keep: torch.Tensor) -> torch.Tensor:
        """Embed the full-atlas gradients ((N, G) or (B, N, G)), then select kept ROIs -> (B, K, dim)."""
        pe = module(gradients)                                                  # (N, dim) or (B, N, dim)
        if pe.ndim == 2:
            return pe[roi_keep]
        return torch.gather(pe, 1, roi_keep[..., None].expand(-1, -1, pe.shape[-1]))

    def forward(self, x: torch.Tensor, roi_keep: torch.Tensor | None = None,
                voxel_valid: torch.Tensor | None = None, tr: float | torch.Tensor | None = None,
                gradients: torch.Tensor | None = None, return_intermediates: bool = False):
        """
        x:           (B, T, V)   one window of T TRs over the stored voxels (T == cfg.n_timepoints)
        roi_keep:    (B, K) or (K,) long, ROIs to encode; None = all N
        voxel_valid: (B, V) bool, per-run valid voxels (the stored ``_valid.npy``); None = all valid
        tr:          float or (B,) seconds; None = cfg.tr
        gradients:   (B, N, G) subject-level gradients; required iff cfg.gradient_source == "subject"
        returns      (B, K, D), plus a dict of intermediate tensors if return_intermediates

        Steps:
            tokens  = BrainTokenizer(x)                              (B, T, K, d)
            time_pe = sincos(t * TR / tr_ref)                        (B, T, d/2)
            grad_pe = GradPE_temporal(gradients)[roi_keep]           (B, K, d/2)
            roi     = TemporalTransformer(tokens, time_pe, grad_pe)  (B, K, d)
            out     = SpatialTransformer(roi + GradPE_spatial)       (B, K, D)
        """
        cfg = self.cfg
        B, T, V = x.shape
        assert T == cfg.n_timepoints, f"window has {T} TRs, config says {cfg.n_timepoints}"
        roi_keep = self._roi_keep(roi_keep, B, x.device)
        K = roi_keep.shape[1]

        tr = torch.as_tensor(cfg.tr if tr is None else tr, dtype=torch.float32, device=x.device)
        if tr.ndim == 0:
            tr = tr.expand(B)
        assert tr.shape == (B,)

        if cfg.gradient_source == "subject":
            assert gradients is not None and gradients.shape == (B, self.n_rois, cfg.grad_dim), \
                "gradient_source='subject' needs gradients of shape (B, N, G)"
            temporal_grad = gradients if cfg.subject_gradients_in_temporal else self.group_gradients
            spatial_grad = gradients
        else:
            assert gradients is None, "gradient_source='group' uses the fixed gradients"
            temporal_grad = spatial_grad = self.group_gradients

        tokens = self.tokenizer(x, roi_keep, voxel_valid)                       # (B, T, K, d)
        time_pe = temporal_sincos(T, cfg.d_model // 2, tr, cfg.tr_ref)          # (B, T, d/2)
        grad_pe_t = self._gradient_pe(self.temporal_grad_pe, temporal_grad, roi_keep)  # (B, K, d/2)
        roi_tokens = self.temporal(tokens, time_pe, grad_pe_t)                  # (B, K, d)
        grad_pe_s = self._gradient_pe(self.spatial_grad_pe, spatial_grad, roi_keep)    # (B, K, d)
        out = self.spatial(roi_tokens, grad_pe_s)                               # (B, K, D)
        assert out.shape == (B, K, cfg.out_dim)

        if return_intermediates:
            return out, {"roi_keep": roi_keep, "tokens": tokens, "temporal": roi_tokens}
        return out
