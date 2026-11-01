import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ema_target_encoder import EMATargetEncoder, ema_momentum
from predictor import Predictor, PredictorConfig
from temporal_bit import TemporalBIT, TemporalBITConfig

N_ROIS, N_VOX, G, T, D = 5, 80, 6, 4, 16


def make_encoder(rank=None, seed=0, **kw) -> TemporalBIT:
    labels = np.random.default_rng(0).integers(0, N_ROIS + 1, N_VOX)
    labels[:N_ROIS] = np.arange(1, N_ROIS + 1)
    cfg = TemporalBITConfig(n_timepoints=T, d_model=32, voxel_rank=rank, temporal_depth=1, temporal_heads=4,
                            spatial_depth=1, spatial_heads=4, out_dim=D, grad_dim=G, **kw)
    torch.manual_seed(seed)
    return TemporalBIT(cfg, labels, torch.randn(N_ROIS, G))


def test_starts_as_a_frozen_independent_copy():
    encoder = make_encoder()
    target = EMATargetEncoder(encoder)
    for (name, p), q in zip(encoder.named_parameters(), target.encoder.parameters()):
        assert torch.equal(p, q), name
        assert p is not q and not q.requires_grad and p.requires_grad


@pytest.mark.parametrize("rank", [None, 8])
def test_update_is_the_moving_average(rank):
    encoder = make_encoder(rank)
    target = EMATargetEncoder(encoder)
    old = {k: v.clone() for k, v in target.encoder.named_parameters()}
    with torch.no_grad():
        for p in encoder.parameters():
            p.add_(torch.randn_like(p))

    target.update(encoder, momentum=1.0)
    for k, p in target.encoder.named_parameters():
        assert torch.equal(p, old[k]), k
    target.update(encoder, momentum=0.75)
    for k, p in encoder.named_parameters():
        assert torch.allclose(dict(target.encoder.named_parameters())[k], 0.75 * old[k] + 0.25 * p, atol=1e-6), k
    target.update(encoder, momentum=0.0)
    for k, p in encoder.named_parameters():
        assert torch.equal(dict(target.encoder.named_parameters())[k], p), k


def test_output_is_normalized_and_read_at_target_rois():
    encoder = make_encoder().eval()
    target = EMATargetEncoder(encoder)
    x = torch.randn(2, T, N_VOX).half()
    full = target(x)
    assert full.shape == (2, N_ROIS, D) and not full.requires_grad
    assert torch.allclose(full, F.layer_norm(encoder(x), (D,)), atol=1e-5)
    assert torch.allclose(full.mean(-1), torch.zeros(2, N_ROIS), atol=1e-5)

    rois = torch.tensor([[3, 0], [1, 4]])
    out = target(x, rois)
    assert torch.equal(out[0], full[0, [3, 0]]) and torch.equal(out[1], full[1, [1, 4]])


def test_targets_see_every_roi():
    """Targets come from a full forward: a change in another ROI's voxels reaches them."""
    target = EMATargetEncoder(make_encoder())
    labels = np.random.default_rng(0).integers(0, N_ROIS + 1, N_VOX)
    labels[:N_ROIS] = np.arange(1, N_ROIS + 1)
    x = torch.randn(1, T, N_VOX)
    changed = x.clone()
    changed[:, :, labels == 2] += 5.0                   # ROI index 1
    rois = torch.tensor([[0]])
    assert not torch.allclose(target(x, rois), target(changed, rois), atol=1e-5)


def test_stays_in_eval_mode():
    target = EMATargetEncoder(make_encoder(dropout=0.5))
    parent = torch.nn.ModuleDict({"target": target}).train()
    assert not parent["target"].training and not target.encoder.spatial.training
    x = torch.randn(1, T, N_VOX)
    assert torch.equal(target(x), target(x))


def test_momentum_schedule():
    assert ema_momentum(0, 100) == 0.996
    assert ema_momentum(50, 100) == pytest.approx(0.998)
    assert ema_momentum(100, 100) == 1.0
    assert ema_momentum(150, 100) == 1.0


def test_training_step_moves_only_what_it_should():
    """Context encoder + predictor + target encoder on an explicit mask."""
    encoder = make_encoder().train()
    target = EMATargetEncoder(encoder)
    predictor = Predictor(PredictorConfig(embed_dim=D, predictor_dim=24, depth=1, heads=4, grad_dim=G),
                          encoder.group_gradients)
    context_rois, target_rois = torch.tensor([[0, 1], [2, 3]]), torch.tensor([[2, 3, 4], [0, 1, 4]])
    x = torch.randn(2, T, N_VOX).half()

    loss = F.mse_loss(predictor(encoder(x, roi_keep=context_rois), context_rois, target_rois),
                      target(x, target_rois))
    loss.backward()

    assert all(p.grad is None for p in target.parameters())
    assert encoder.tokenizer.roi_embed.grad.abs().sum() > 0
    assert predictor.mask_token.grad.abs().sum() > 0

    before = target.encoder.tokenizer.roi_embed.clone()
    with torch.no_grad():
        encoder.tokenizer.roi_embed.sub_(0.1 * encoder.tokenizer.roi_embed.grad)
    target.update(encoder, momentum=0.9)
    assert not torch.equal(target.encoder.tokenizer.roi_embed, before)
