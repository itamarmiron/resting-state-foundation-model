import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from rsfm.preprocessing import atlas as atlas_mod
from rsfm.preprocessing.denoise import denoise_run, select_confounds
from rsfm.preprocessing.inventory import describe_run, load_manifest, run_files, save_manifest
from rsfm.preprocessing.voxel_store import Grid, VoxelRun, build_group_mask, save_run

SHAPE = (6, 5, 4)
AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])
CONF_CFG = {"motion": ["trans_x", "rot_x"], "n_acompcor": 2, "cosine": True, "global_signal": False}


def make_run(func: Path, run_id: str, n_tr: int = 120, tr: float = 0.72, seed: int = 0) -> Path:
    """Synthetic fMRIPrep run: BOLD = smooth signal + a known confound, plus mask/confounds."""
    rng = np.random.default_rng(seed)
    func.mkdir(parents=True, exist_ok=True)
    t = np.arange(n_tr) * tr
    confound = rng.standard_normal(n_tr)
    neural = np.sin(2 * np.pi * 0.05 * t)                       # in the 0.008-0.1 Hz band
    mask = np.zeros(SHAPE, bool)
    mask[1:5, 1:4, :3] = True
    bold = np.zeros(SHAPE + (n_tr,), np.float32)
    bold[mask] = 100 + neural + 3 * confound + 0.1 * rng.standard_normal((mask.sum(), n_tr))
    stem = f"{run_id}_space-MNI152NLin2009cAsym_res-2"
    bold_path = func / f"{stem}_desc-preproc_bold.nii.gz"
    nib.save(nib.Nifti1Image(bold, AFFINE), bold_path)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), AFFINE), func / f"{stem}_desc-brain_mask.nii.gz")
    (func / f"{stem}_desc-preproc_bold.json").write_text(json.dumps({"RepetitionTime": tr}))

    # aCompCor numbering where CSF components come first: selection must use the sidecar.
    conf = pd.DataFrame({
        "trans_x": rng.standard_normal(n_tr), "rot_x": rng.standard_normal(n_tr),
        "a_comp_cor_00": rng.standard_normal(n_tr), "a_comp_cor_01": confound,
        "a_comp_cor_02": rng.standard_normal(n_tr), "cosine00": np.cos(np.pi * np.arange(n_tr) / n_tr),
        "framewise_displacement": np.r_[np.nan, np.full(n_tr - 1, 0.1)],
        "global_signal": rng.standard_normal(n_tr),
    })
    conf.to_csv(func / f"{run_id}_desc-confounds_timeseries.tsv", sep="\t", index=False, na_rep="n/a")
    sidecar = {"a_comp_cor_00": {"Mask": "CSF", "Retained": True},
               "a_comp_cor_01": {"Mask": "combined", "Retained": True},
               "a_comp_cor_02": {"Mask": "combined", "Retained": True}}
    (func / f"{run_id}_desc-confounds_timeseries.json").write_text(json.dumps(sidecar))
    return bold_path


@pytest.fixture
def run_row(tmp_path):
    bold = make_run(tmp_path / "sub-01" / "ses-1" / "func", "sub-01_ses-1_task-rest_dir-LR")
    row = pd.Series(describe_run(bold, fd_threshold=0.5))
    return row


def test_run_files_keeps_dir_entity():
    files = run_files(Path("/x/sub-1_ses-2_task-rest_dir-RL_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"))
    assert files["run_id"] == "sub-1_ses-2_task-rest_dir-RL"
    assert files["confounds"].name == "sub-1_ses-2_task-rest_dir-RL_desc-confounds_timeseries.tsv"
    assert files["mask"].name.endswith("_res-2_desc-brain_mask.nii.gz")


def test_describe_run_reads_tr_from_json(run_row):
    assert run_row["missing"] == ""
    assert run_row["tr"] == 0.72 and run_row["n_tr"] == 120
    assert run_row["dir"] == "LR" and run_row["sub"] == "01" and run_row["ses"] == "1"


def test_manifest_roundtrip(run_row, tmp_path):
    df = pd.DataFrame([run_row.to_dict()]).assign(exclude=False)
    save_manifest(df, tmp_path / "out")
    back = load_manifest(tmp_path / "out").iloc[0]
    assert back["sub"] == "01" and back["tr"] == 0.72 and not back["exclude"]
    assert isinstance(back["bold"], Path)


def test_select_confounds_uses_combined_acompcor(run_row):
    conf = select_confounds(run_row["confounds"], run_row["confounds_json"], CONF_CFG)
    assert list(conf.columns) == ["trans_x", "rot_x", "a_comp_cor_01", "a_comp_cor_02", "cosine00"]
    assert np.allclose(conf.mean(), 0)


def test_select_confounds_needs_enough_components(run_row):
    with pytest.raises(ValueError, match="combined aCompCor"):
        select_confounds(run_row["confounds"], run_row["confounds_json"], {**CONF_CFG, "n_acompcor": 3})


def _cfg(low_pass=0.1):
    return {"confounds": CONF_CFG, "io": {"chunk_trs": 50},
            "denoise": {"low_pass": low_pass, "detrend": True, "standardize": "zscore_sample"}}


def test_denoise_removes_confound_and_keeps_signal(tmp_path):
    n_tr = 600
    bold = make_run(tmp_path / "sub-02" / "func", "sub-02_task-rest_dir-RL", n_tr=n_tr)
    row = pd.Series(describe_run(bold, fd_threshold=0.5))
    grid = Grid(mask=np.ones(SHAPE, bool), affine=AFFINE)      # bigger than the run mask
    data, valid, fd = denoise_run(row, grid, _cfg())
    assert data.shape == (n_tr, np.prod(SHAPE)) and valid.sum() == 4 * 3 * 3
    assert np.all(data[:, ~valid] == 0)
    v = data[:, valid]
    assert np.allclose(v.mean(0), 0, atol=1e-5) and np.allclose(v.std(0, ddof=1), 1, atol=1e-4)

    neural = np.sin(2 * np.pi * 0.05 * np.arange(n_tr) * 0.72)
    confound = pd.read_csv(row["confounds"], sep="\t")["a_comp_cor_01"].to_numpy()
    assert np.corrcoef(v.mean(1), neural)[0, 1] > 0.95
    assert abs(np.corrcoef(v.mean(1), confound)[0, 1]) < 0.05
    assert fd[0] == 0 and fd.dtype == np.float32


def test_grid_mismatch_is_rejected(run_row):
    grid = Grid(mask=np.ones(SHAPE, bool), affine=np.diag([3.0, 3.0, 3.0, 1.0]))
    with pytest.raises(ValueError, match="not on the group grid"):
        denoise_run(run_row, grid, _cfg())


def test_group_mask_fraction(tmp_path, run_row):
    template = tmp_path / "template.nii.gz"
    tmask = np.ones(SHAPE, np.uint8)
    tmask[0] = 0
    nib.save(nib.Nifti1Image(tmask, AFFINE), template)
    grid, info = build_group_mask(pd.DataFrame([run_row]), template, 0.5)
    assert info["n_voxels"] == 36 and grid.mask[1:5, 1:4, :3].all()


def test_save_and_load_run(tmp_path, run_row):
    grid = Grid(mask=np.ones(SHAPE, bool), affine=AFFINE)
    data, valid, fd = denoise_run(run_row, grid, _cfg(low_pass=None))
    save_run(tmp_path, run_row, data, valid, fd)
    run = VoxelRun.load(tmp_path, run_row, grid)
    assert run.data.dtype == np.float16 and run.data.shape == data.shape
    vol = run.to_volume(5)
    assert vol.shape == SHAPE and np.allclose(vol[grid.mask], np.where(valid, data[5], 0), atol=1e-2)
    assert run.to_volume().shape == SHAPE + (120,)


def test_parcellate_means_valid_voxels_only():
    labels = np.array([1, 1, 2, 2, 0, 3])
    atlas = atlas_mod.Atlas("toy", labels, pd.DataFrame({"index": [1, 2, 3], "n_voxels": [2, 4, 1]}))
    data = np.array([[1, 3, 10, 99, 7, 5], [3, 5, 20, 99, 7, 6]], dtype=np.float16)
    valid = np.array([True, True, True, False, True, False])
    ts, cov = atlas_mod.parcellate(data, valid, atlas, zscore=False)
    assert np.allclose(ts[:, 0], [2, 4]) and np.allclose(ts[:, 1], [10, 20])
    assert np.isnan(ts[:, 2]).all()
    assert np.allclose(cov, [1.0, 0.25, 0.0])


def test_parse_atlas_name_rejects_unknown():
    with pytest.raises(ValueError, match="unknown atlas component"):
        atlas_mod.parse_atlas_name("schaefer400_aal")
