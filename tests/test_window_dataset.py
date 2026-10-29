import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from rsfm.preprocessing.inventory import save_manifest
from rsfm.preprocessing.voxel_store import Grid, save_grid, save_run
from window_dataset import WindowDataset

SHAPE = (4, 3, 2)
AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])
T = 10


def _row(sub, ses, n_tr, tr=0.72, exclude=False):
    run_id = f"sub-{sub}_ses-{ses}_task-rest_dir-LR"
    return {"run_id": run_id, "bold": "", "bold_json": "", "mask": "", "confounds": "",
            "confounds_json": "", "sub": sub, "ses": ses, "task": "rest", "acq": "", "dir": "LR",
            "run": "", "missing": "", "tr": tr, "n_tr": n_tr, "mean_fd": 0.1, "max_fd": 0.2,
            "pct_high_fd": 0.0, "exclude": exclude}


@pytest.fixture
def data_root(tmp_path):
    """Stored runs whose voxel 0 holds the TR index and voxel 1 the run's position in ``rows``."""
    mask = np.ones(SHAPE, bool)
    mask[0, 0, 0] = False
    grid = Grid(mask=mask, affine=AFFINE)
    save_grid(tmp_path, grid, {"n_voxels": grid.n_voxels})
    rows = [_row("01", "1", 35), _row("01", "2", 20, tr=0.8), _row("02", "1", 30),
            _row("03", "1", 30, exclude=True),      # flagged high motion
            _row("04", "1", 8),                     # shorter than one window
            _row("05", "1", 30)]                    # in the manifest, never stored
    for k, row in enumerate(rows[:5]):
        data = np.zeros((row["n_tr"], grid.n_voxels), np.float32)
        data[:, 0] = np.arange(row["n_tr"])
        data[:, 1] = k
        valid = np.ones(grid.n_voxels, bool)
        valid[-1] = k % 2 == 0
        save_run(tmp_path, row, data, valid, np.zeros(row["n_tr"], np.float32))
    save_manifest(pd.DataFrame(rows), tmp_path)
    return tmp_path


def test_run_selection(data_root):
    ds = WindowDataset(data_root, T, mode="grid")
    assert ds.runs["sub"].tolist() == ["01", "01", "02"]
    assert WindowDataset(data_root, T, mode="grid", include_excluded=True).runs["sub"].tolist() == \
        ["01", "01", "02", "03"]
    assert WindowDataset(data_root, T, subjects=["02"]).runs["sub"].tolist() == ["02"]
    with pytest.raises(ValueError):
        WindowDataset(data_root, T, subjects=["05"])


def test_grid_windows(data_root):
    ds = WindowDataset(data_root, T, mode="grid")
    assert len(ds) == 3 + 2 + 3
    starts = [int(ds[i]["start"]) for i in range(len(ds))]
    assert starts == [0, 10, 20, 0, 10, 0, 10, 20]
    item = ds[4]                                            # run 1 (sub-01 ses-2), second window
    assert item["x"].shape == (T, ds.grid.n_voxels) and item["x"].dtype == torch.float16
    assert item["x"][:, 0].tolist() == list(range(10, 20)) and (item["x"][:, 1] == 1).all()
    assert item["valid"].dtype == torch.bool and not item["valid"][-1]
    assert item["tr"].dtype == torch.float32 and abs(float(item["tr"]) - 0.8) < 1e-6
    assert int(item["run"]) == 1
    assert len(WindowDataset(data_root, T, mode="grid", stride=5)) == 6 + 3 + 5


def test_random_windows(data_root):
    ds = WindowDataset(data_root, T)
    assert len(ds) == 3 + 2 + 3                             # n_tr // T windows per run
    torch.manual_seed(0)
    starts = set()
    for _ in range(20):
        item = ds[0]                                        # run 0 has 35 TRs
        s = int(item["start"])
        assert 0 <= s <= 25 and item["x"][:, 0].tolist() == list(range(s, s + T))
        starts.add(s)
    assert len(starts) > 5


def test_dataloader_batches_match_model_inputs(data_root):
    ds = WindowDataset(data_root, T)
    batch = next(iter(DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)))
    assert batch["x"].shape == (4, T, ds.grid.n_voxels) and batch["x"].dtype == torch.float16
    assert batch["valid"].shape == (4, ds.grid.n_voxels) and batch["valid"].dtype == torch.bool
    assert batch["tr"].shape == (4,) and batch["tr"].dtype == torch.float32
    # each window's content agrees with its run and start
    for b in range(4):
        s, r = int(batch["start"][b]), int(batch["run"][b])
        assert batch["x"][b, :, 0].tolist() == list(range(s, s + T)) and (batch["x"][b, :, 1] == r).all()
