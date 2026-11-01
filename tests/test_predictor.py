import pytest
import torch

from predictor import Predictor, PredictorConfig

N_ROIS, D, G = 12, 16, 6


def make_predictor(**kw) -> Predictor:
    cfg = PredictorConfig(embed_dim=D, predictor_dim=24, depth=2, heads=4, grad_dim=G, **kw)
    torch.manual_seed(0)
    return Predictor(cfg, torch.randn(N_ROIS, G)).eval()


CONTEXT = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
TARGET = torch.tensor([[4, 5, 6], [0, 1, 2]])


def test_default_config_is_a_small_vit():
    model = Predictor(PredictorConfig(), torch.randn(450, 30))
    out = model(torch.randn(2, 8, 512), torch.arange(8).expand(2, -1), torch.arange(8, 11).expand(2, -1))
    assert out.shape == (2, 3, 512)
    assert sum(p.numel() for p in model.parameters()) < 15_000_000


def test_shapes_and_bad_inputs():
    model = make_predictor()
    z = torch.randn(2, 4, D)
    assert model(z, CONTEXT, TARGET).shape == (2, 3, D)
    with pytest.raises(AssertionError):
        model(z, CONTEXT[:, :3], TARGET)                # z and context_rois disagree
    with pytest.raises(AssertionError):
        model(torch.randn(2, 4, D + 1), CONTEXT, TARGET)


def test_output_follows_target_order():
    model = make_predictor()
    z = torch.randn(2, 4, D)
    perm = torch.tensor([2, 0, 1])
    assert torch.allclose(model(z, CONTEXT, TARGET[:, perm]), model(z, CONTEXT, TARGET)[:, perm], atol=1e-5)


def test_prediction_depends_on_query_position_and_context():
    model = make_predictor()
    z = torch.randn(2, 4, D)
    out = model(z, CONTEXT, TARGET)
    assert not torch.allclose(out, model(z, CONTEXT, TARGET.flip(0)), atol=1e-4)    # other target ROIs
    assert not torch.allclose(out, model(z, CONTEXT.flip(0), TARGET), atol=1e-4)    # other context positions
    assert not torch.allclose(out, model(torch.randn(2, 4, D), CONTEXT, TARGET), atol=1e-4)  # other features


def test_samples_do_not_interact():
    model = make_predictor()
    z = torch.randn(2, 4, D)
    full = model(z, CONTEXT, TARGET)
    assert torch.allclose(full[:1], model(z[:1], CONTEXT[:1], TARGET[:1]), atol=1e-5)
    changed = z.clone()
    changed[1] += 1.0
    assert torch.equal(full[0], model(changed, CONTEXT, TARGET)[0])


def test_every_parameter_gets_a_gradient():
    model = make_predictor().train()
    z = torch.randn(2, 4, D, requires_grad=True)
    model(z, CONTEXT, TARGET).pow(2).mean().backward()
    assert z.grad.abs().sum() > 0
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_gradients_are_not_a_parameter():
    assert "group_gradients" not in dict(make_predictor().named_parameters())
    assert "group_gradients" in make_predictor().state_dict()
