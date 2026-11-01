import csv
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from ema_target_encoder import EMATargetEncoder
from predictor import Predictor, PredictorConfig
from rsfm.preprocessing.inventory import save_manifest
from rsfm.preprocessing.voxel_store import Grid, save_grid, save_run
from temporal_bit import TemporalBIT, TemporalBITConfig
from train_jepa import (TrainConfig, batch_step, build_optimizer, jepa_loss, lr_at, set_schedules, train,
                        weight_decay_at)
from window_dataset import WindowDataset

SHAPE, T, N_ROIS, G, D = (4, 3, 2), 10, 4, 6, 16
ENCODER = dict(n_timepoints=T, d_model=32, temporal_depth=1, temporal_heads=4, spatial_depth=1, spatial_heads=4,
               out_dim=D, grad_dim=G)
PREDICTOR = dict(predictor_dim=24, depth=1, heads=4)
MASK = dict(target_scale=[0.25, 0.5], context_scale=[1.0, 1.0], min_context=1)
CPU = torch.device("cpu")


def make_data_root(path, value=None):
    """Three stored runs of 30 TRs over 23 voxels: 9 windows."""
    grid_mask = np.ones(SHAPE, bool)
    grid_mask[0, 0, 0] = False
    grid = Grid(mask=grid_mask, affine=np.diag([2.0, 2.0, 2.0, 1.0]))
    save_grid(path, grid, {"n_voxels": grid.n_voxels})
    rng, rows = np.random.default_rng(0), []
    for sub in ("01", "02", "03"):
        row = {"run_id": f"sub-{sub}_ses-1_task-rest_dir-LR", "bold": "", "bold_json": "", "mask": "",
               "confounds": "", "confounds_json": "", "sub": sub, "ses": "1", "task": "rest", "acq": "",
               "dir": "LR", "run": "", "missing": "", "tr": 0.72, "n_tr": 30, "mean_fd": 0.1, "max_fd": 0.2,
               "pct_high_fd": 0.0, "exclude": False}
        data = rng.standard_normal((30, grid.n_voxels)).astype(np.float32)
        if value is not None:
            data[:] = value
        save_run(path, row, data, np.ones(grid.n_voxels, bool), np.zeros(30, np.float32))
        rows.append(row)
    save_manifest(pd.DataFrame(rows), path)
    return path


def labels() -> np.ndarray:
    return (np.arange(23) % N_ROIS) + 1


def gradients() -> torch.Tensor:
    return torch.randn(N_ROIS, G, generator=torch.Generator().manual_seed(0))


def config(tmp_path, **kw) -> TrainConfig:
    values = dict(data_root=str(tmp_path / "data"), output_dir=str(tmp_path / "run"), batch_size=2,
                  accumulation_steps=2, epochs=2, warmup_epochs=1, precision="fp32", num_workers=0, log_every=1,
                  checkpoint_every=100, encoder=ENCODER, predictor=PREDICTOR, mask=MASK)
    return TrainConfig(**{**values, **kw})


@pytest.fixture
def setup(tmp_path):
    (tmp_path / "data").mkdir()
    make_data_root(tmp_path / "data")
    return tmp_path, WindowDataset(tmp_path / "data", T)


def models():
    torch.manual_seed(0)
    encoder = TemporalBIT(TemporalBITConfig(**ENCODER), labels(), gradients())
    predictor = Predictor(PredictorConfig(embed_dim=D, grad_dim=G, **PREDICTOR), gradients())
    return encoder, predictor, EMATargetEncoder(encoder)


def log_rows(tmp_path) -> list[dict]:
    with open(tmp_path / "run" / "train_log.csv") as f:
        return list(csv.DictReader(f))


def test_loss_is_smooth_l1():
    assert jepa_loss(torch.tensor([0.5]), torch.tensor([0.0])).item() == pytest.approx(0.125)   # 0.5 e^2
    assert jepa_loss(torch.tensor([3.0]), torch.tensor([0.0])).item() == pytest.approx(2.5)     # |e| - 0.5


def test_lr_schedule():
    args = dict(warmup_steps=10, total_steps=110, start=0.1, ref=1.0, final=0.01)
    assert lr_at(0, **args) == pytest.approx(0.1)
    assert lr_at(5, **args) == pytest.approx(0.55)
    assert lr_at(10, **args) == pytest.approx(1.0)
    assert lr_at(60, **args) == pytest.approx(0.505)
    assert lr_at(110, **args) == pytest.approx(0.01)


def test_weight_decay_schedule():
    assert weight_decay_at(0, 100, 0.04, 0.4) == pytest.approx(0.04)
    assert weight_decay_at(50, 100, 0.04, 0.4) == pytest.approx(0.22)
    assert weight_decay_at(100, 100, 0.04, 0.4) == pytest.approx(0.4)


def test_optimizer_groups():
    encoder, predictor, _ = models()
    optimizer = build_optimizer([encoder, predictor])
    set_schedules(optimizer, 0.5, 0.3)
    decayed, not_decayed = optimizer.param_groups
    assert decayed["lr"] == not_decayed["lr"] == 0.5
    assert decayed["weight_decay"] == 0.3 and not_decayed["weight_decay"] == 0.0
    all_params = list(encoder.parameters()) + list(predictor.parameters())
    assert len(decayed["params"]) + len(not_decayed["params"]) == len(all_params)
    decayed_ids = {id(p) for p in decayed["params"]}
    assert id(encoder.tokenizer.voxel_embed) in decayed_ids and id(predictor.mask_token) in decayed_ids
    assert id(encoder.spatial.norm.weight) not in decayed_ids and id(predictor.proj.bias) not in decayed_ids


def make_batch(seed: int, B: int = 2) -> dict:
    g = torch.Generator().manual_seed(seed)
    perm = torch.rand(B, N_ROIS, generator=g).argsort(1)
    return {"x": torch.randn(B, T, 23, generator=g).half(), "valid": torch.ones(B, 23, dtype=torch.bool),
            "tr": torch.full((B,), 0.72), "target_rois": perm[:, :2].sort(1).values,
            "context_rois": perm[:, 2:].sort(1).values}


def test_accumulated_gradients_equal_one_big_batch():
    a, b = make_batch(1), make_batch(2)
    encoder, predictor, target = models()
    batch_step(encoder, predictor, target, a, CPU, 2, "fp32")
    batch_step(encoder, predictor, target, b, CPU, 2, "fp32")
    accumulated = [p.grad.clone() for p in encoder.parameters()]

    encoder, predictor, target = models()
    both = {k: torch.cat([a[k], b[k]]) for k in a}
    batch_step(encoder, predictor, target, both, CPU, 1, "fp32")
    for grad, p in zip(accumulated, encoder.parameters()):
        assert torch.allclose(grad, p.grad, atol=1e-5)


def test_target_encoder_gets_no_gradient():
    encoder, predictor, target = models()
    out = batch_step(encoder, predictor, target, make_batch(1), CPU, 1, "bf16")
    assert np.isfinite(out["loss"]) and out["target_std"] > 0 and out["pred_std"] > 0
    assert all(p.grad is None for p in target.parameters())
    assert encoder.tokenizer.voxel_embed.grad is not None and predictor.mask_token.grad is not None


def test_training_runs_its_schedule(setup):
    tmp_path, dataset = setup
    cfg = config(tmp_path)
    assert train(cfg, dataset, labels(), gradients(), CPU) == 4       # 2 epochs * 4 batches / 2
    rows = log_rows(tmp_path)
    assert [int(r["step"]) for r in rows] == [1, 2, 3, 4]
    assert float(rows[0]["lr"]) == pytest.approx(cfg.start_lr)        # warmup is 2 optimizer steps
    assert float(rows[2]["lr"]) == pytest.approx(cfg.lr)
    assert float(rows[0]["momentum"]) == pytest.approx(cfg.ema_start)
    ckpt = torch.load(tmp_path / "run" / "latest.pt")
    assert ckpt["step"] == ckpt["total_steps"] == 4
    assert not torch.equal(ckpt["target_encoder"]["encoder.tokenizer.roi_embed"], ckpt["encoder"]["tokenizer.roi_embed"])


def test_resume_continues_the_same_schedule(setup):
    tmp_path, dataset = setup
    cfg = config(tmp_path)
    assert train(cfg, dataset, labels(), gradients(), CPU, stop_after=1) == 1
    saved = torch.load(tmp_path / "run" / "latest.pt")["encoder"]["tokenizer.roi_embed"]
    assert train(cfg, dataset, labels(), gradients(), CPU) == 4
    rows = log_rows(tmp_path)
    assert [int(r["step"]) for r in rows] == [1, 2, 3, 4]
    assert float(rows[2]["lr"]) == pytest.approx(cfg.lr)
    assert not torch.equal(saved, torch.load(tmp_path / "run" / "latest.pt")["encoder"]["tokenizer.roi_embed"])
    assert train(cfg, dataset, labels(), gradients(), CPU) == 4       # already finished: nothing runs
    assert len(log_rows(tmp_path)) == 4


def test_resume_refuses_a_different_run(setup):
    tmp_path, dataset = setup
    cfg = config(tmp_path)
    train(cfg, dataset, labels(), gradients(), CPU, stop_after=1)
    with pytest.raises(ValueError, match="other settings"):
        train(replace(cfg, lr=0.5), dataset, labels(), gradients(), CPU)
    with pytest.raises(ValueError, match="optimizer steps"):
        smaller = WindowDataset(tmp_path / "data", T, subjects=["01"])
        train(cfg, smaller, labels(), gradients(), CPU)
    train(replace(cfg, log_every=2), dataset, labels(), gradients(), CPU, stop_after=2)    # logging may change


def test_non_finite_loss_stops_before_any_update(tmp_path):
    (tmp_path / "data").mkdir()
    make_data_root(tmp_path / "data", value=float("nan"))
    with pytest.raises(FloatingPointError):
        train(config(tmp_path), WindowDataset(tmp_path / "data", T), labels(), gradients(), CPU)
    assert not (tmp_path / "run" / "latest.pt").exists()


def test_too_little_data_is_an_error(setup):
    tmp_path, dataset = setup
    with pytest.raises(ValueError, match="no optimizer step"):
        train(config(tmp_path, batch_size=20), dataset, labels(), gradients(), CPU)
