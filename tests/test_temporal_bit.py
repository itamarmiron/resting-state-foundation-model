import numpy as np
import pytest
import torch

import compute_gradients as cg
from temporal_bit import TemporalBIT, TemporalBITConfig, build_v2roi

N_ROIS, N_VOX, G = 5, 80, 6


def make_atlas(seed: int = 0) -> np.ndarray:
    """Synthetic atlas over N_VOX stored voxels: ROIs of unequal size, some unlabelled voxels."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, N_ROIS + 1, N_VOX)
    labels[:N_ROIS] = np.arange(1, N_ROIS + 1)          # every ROI non-empty
    return labels


def make_model(T: int = 20, source: str = "group", **kw) -> TemporalBIT:
    cfg = TemporalBITConfig(n_timepoints=T, d_model=32, voxel_rank=8, temporal_depth=1, temporal_heads=4,
                            spatial_depth=2, spatial_heads=4, out_dim=16, grad_dim=G,
                            gradient_source=source, **kw)
    torch.manual_seed(0)
    return TemporalBIT(cfg, make_atlas(), torch.randn(N_ROIS, G)).eval()


def test_v2roi_padding():
    labels = make_atlas()
    v = build_v2roi(labels)
    assert v.pad_index.shape == (N_ROIS, np.bincount(labels)[1:].max())
    for r in range(N_ROIS):
        real = v.pad_index[r][v.pad_mask[r]].numpy()
        assert sorted(real) == list(np.flatnonzero(labels == r + 1))
        assert (v.pad_index[r][~v.pad_mask[r]] == N_VOX).all()
    # embedding rows point at the matching labelled voxel
    labelled = np.flatnonzero(labels > 0)
    assert (labelled[v.pad_row[v.pad_mask]] == v.pad_index[v.pad_mask].numpy()).all()


@pytest.mark.parametrize("T", [7, 20])
@pytest.mark.parametrize("pool", ["attn", "mean"])
def test_shapes(T, pool):
    model = make_model(T, temporal_pool=pool)
    x = torch.randn(3, T, N_VOX).half()
    out, mid = model(x, return_intermediates=True)
    assert out.shape == (3, N_ROIS, 16)
    assert mid["tokens"].shape == (3, T, N_ROIS, 32)
    assert mid["temporal"].shape == (3, N_ROIS, 32)
    with pytest.raises(AssertionError):
        model(torch.randn(3, T + 1, N_VOX))


def test_subject_gradients_shapes_and_required():
    model = make_model(source="subject")
    x = torch.randn(2, 20, N_VOX)
    assert model(x, gradients=torch.randn(2, N_ROIS, G)).shape == (2, N_ROIS, 16)
    with pytest.raises(AssertionError):
        model(x)


def test_keep_all_equals_none():
    model = make_model()
    x = torch.randn(2, 20, N_VOX)
    assert torch.allclose(model(x), model(x, roi_keep=torch.arange(N_ROIS)), atol=1e-6)


@pytest.mark.parametrize("source", ["group", "subject"])
def test_dropped_roi_has_no_influence(source):
    """Changing the voxels of a dropped ROI must not change anything."""
    model = make_model(source=source)
    labels = make_atlas()
    keep = torch.tensor([[0, 2, 4], [1, 2, 3]])
    grads = torch.randn(2, N_ROIS, G)
    x = torch.randn(2, 20, N_VOX)
    x2 = x.clone()
    x2[0][:, torch.from_numpy(np.isin(labels, [2, 4]))] += 5.0     # ROIs 1, 3 dropped in sample 0
    x2[1][:, torch.from_numpy(labels == 1)] += 5.0                 # ROI 0 dropped in sample 1
    kw = {"gradients": grads} if source == "subject" else {}
    out, out2 = model(x, roi_keep=keep, **kw), model(x2, roi_keep=keep, **kw)
    assert out.shape == (2, 3, 16)
    assert torch.equal(out, out2)


def test_no_cross_roi_mixing_before_spatial():
    """Perturbing ROI j changes only ROI j's token up to the spatial transformer."""
    model = make_model()
    labels = make_atlas()
    x = torch.randn(1, 20, N_VOX)
    x2 = x.clone()
    x2[0][:, torch.from_numpy(labels == 3)] += 5.0                 # ROI index 2
    _, mid = model(x, return_intermediates=True)
    _, mid2 = model(x2, return_intermediates=True)
    others = [0, 1, 3, 4]
    for key in ("tokens", "temporal"):
        a, b = mid[key][..., others, :], mid2[key][..., others, :]
        assert torch.allclose(a, b, atol=1e-6), key
        assert not torch.allclose(mid[key][..., 2, :], mid2[key][..., 2, :])


def test_invalid_voxels_masked_and_empty_roi_no_nan():
    model = make_model().train()
    labels = make_atlas()
    x = torch.randn(2, 20, N_VOX, requires_grad=True)
    valid = torch.ones(2, N_VOX, dtype=torch.bool)
    valid[0, torch.from_numpy(labels == 2)] = False                # ROI 1 fully invalid in sample 0
    out = model(x, voxel_valid=valid)
    out.sum().backward()
    assert torch.isfinite(out).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    # invalid voxels' values are ignored
    with torch.no_grad():
        model.eval()
        x2 = x.detach().clone()
        x2[0][:, torch.from_numpy(labels == 2)] = 100.0
        assert torch.equal(model(x.detach(), voxel_valid=valid), model(x2, voxel_valid=valid))


def test_gradient_pe_unchanged_by_masking():
    model = make_model()
    keep = torch.tensor([[1, 3]])
    full = model._gradient_pe(model.spatial_grad_pe, model.group_gradients, torch.arange(N_ROIS)[None])
    part = model._gradient_pe(model.spatial_grad_pe, model.group_gradients, keep)
    assert torch.equal(part, full[:, [1, 3]])


def test_tr_changes_only_time_positions():
    model = make_model()
    x = torch.randn(1, 20, N_VOX)
    assert torch.equal(model(x), model(x, tr=0.72))
    assert not torch.allclose(model(x), model(x, tr=torch.tensor([1.5])))


# --- gradient pipeline -------------------------------------------------------------------------

def test_diffusion_map_and_procrustes():
    rng = np.random.default_rng(0)
    ts = rng.standard_normal((300, 3)) @ rng.standard_normal((3, 40)) + 0.5 * rng.standard_normal((300, 40))
    fc = np.corrcoef(ts.T)
    grad = cg.diffusion_mapping(cg.compute_affinity(fc), n_components=5)
    assert grad.shape == (40, 5) and np.isfinite(grad).all()
    rot, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    assert np.allclose(cg.procrustes(grad @ rot, grad), grad, atol=1e-8)


def test_subject_gradients_fill_missing_roi():
    rng = np.random.default_rng(1)
    fc = np.corrcoef(rng.standard_normal((200, 30)).T)
    fc[4, :] = fc[:, 4] = np.nan
    reference = rng.standard_normal((30, 5))
    out = cg.subject_gradients(fc, reference)
    assert out.shape == (30, 5) and np.isfinite(out).all()
    assert np.array_equal(out[4], reference[4])
