import pytest
import torch

from roi_masks import ROIMaskCollator, ROIMaskConfig, ROIMaskSampler

N = 450


def test_masks_are_disjoint_sorted_and_in_range():
    context, target = ROIMaskSampler(N)(8, generator=torch.Generator().manual_seed(0))
    assert context.dtype == target.dtype == torch.long
    assert context.shape[0] == target.shape[0] == 8
    for c, t in zip(context, target):
        assert (c.diff() > 0).all() and (t.diff() > 0).all()            # sorted, no repeats
        assert not set(c.tolist()) & set(t.tolist())
        assert 0 <= int(min(c.min(), t.min())) and int(max(c.max(), t.max())) < N
    assert not torch.equal(target[0], target[1])                         # ROIs differ per sample


def test_sizes_follow_brainjepa_ratios():
    sampler, g = ROIMaskSampler(N), torch.Generator().manual_seed(1)
    sizes = [sampler.sample_sizes(g) for _ in range(500)]
    n_tgt = torch.tensor([t for _, t in sizes], dtype=torch.float)
    n_ctx = torch.tensor([c for c, _ in sizes], dtype=torch.float)
    assert n_tgt.min() >= int(N * 0.45) and n_tgt.max() <= int(N * 0.6)
    assert (n_ctx >= torch.floor(0.84 * (N - n_tgt))).all() and (n_ctx <= N - n_tgt).all()
    assert n_tgt.std() > 0 and n_ctx.std() > 0                           # sizes vary across batches


def test_full_context_covers_every_roi():
    """context_scale=(1, 1) is slide 2's setting: every non-target ROI is visible."""
    context, target = ROIMaskSampler(N, ROIMaskConfig(context_scale=(1.0, 1.0)))(4)
    for c, t in zip(context, target):
        assert sorted(c.tolist() + t.tolist()) == list(range(N))


def test_generator_reproducible():
    sampler = ROIMaskSampler(N)
    a = sampler(3, generator=torch.Generator().manual_seed(7))
    b = sampler(3, generator=torch.Generator().manual_seed(7))
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_small_atlas_keeps_min_context():
    sampler = ROIMaskSampler(6, ROIMaskConfig(target_scale=(0.5, 0.9), context_scale=(0.1, 0.1),
                                              min_context=2))
    for _ in range(50):
        context, target = sampler(2)
        assert context.shape[1] >= 2 and target.shape[1] >= 1
    with pytest.raises(AssertionError):
        ROIMaskSampler(N, ROIMaskConfig(target_scale=(0.5, 1.0)))


def test_collator_adds_masks_to_batch():
    items = [{"x": torch.zeros(5, 30, dtype=torch.float16), "valid": torch.ones(30, dtype=torch.bool),
              "tr": torch.tensor(0.72)} for _ in range(3)]
    batch = ROIMaskCollator(N)(items)
    assert batch["x"].shape == (3, 5, 30)
    assert batch["context_rois"].shape[0] == batch["target_rois"].shape[0] == 3
