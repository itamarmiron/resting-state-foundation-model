"""Denoise one run in memory, on the voxels of the group mask.

Recipe (same as the rsPTSD SwiFT pipeline, minus the bugs):
  - confounds: 6 motion + first N "combined" aCompCor + fMRIPrep cosines (high-pass ~0.008 Hz)
    (+ global signal if enabled), demeaned; read explicitly from the TSV/JSON;
  - nilearn ``signal.clean``: detrend, Butterworth low-pass (applied to the confounds too,
    keeping filters orthogonal to the regression), confound regression, voxelwise z-score;
  - TR comes from the manifest (JSON sidecar), never from the NIfTI header.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import signal

from .voxel_store import Grid

ACOMPCOR_RE = re.compile(r"^a_comp_cor_(\d+)$")


def select_confounds(tsv: Path, sidecar: Path, conf_cfg: dict) -> pd.DataFrame:
    """Return the T x K confound matrix (demeaned) with named columns."""
    header = pd.read_csv(tsv, sep="\t", nrows=0).columns.tolist()
    with open(sidecar) as fh:
        meta = json.load(fh)

    # aCompCor numbering spans the combined/WM/CSF masks, so pick by the sidecar's Mask field.
    combined = sorted(
        (k for k, v in meta.items()
         if ACOMPCOR_RE.match(k) and v.get("Mask") == "combined" and v.get("Retained", True)),
        key=lambda k: int(ACOMPCOR_RE.match(k).group(1)),
    )
    n = conf_cfg["n_acompcor"]
    if len(combined) < n:
        raise ValueError(f"{tsv.name}: only {len(combined)} combined aCompCor components, need {n}")

    columns = list(conf_cfg["motion"]) + combined[:n]
    if conf_cfg.get("cosine", True):
        columns += [c for c in header if c.startswith("cosine")]
    if conf_cfg.get("global_signal", False):
        columns.append("global_signal")
    missing = [c for c in columns if c not in header]
    if missing:
        raise ValueError(f"{tsv.name}: missing confound columns {missing}")

    df = pd.read_csv(tsv, sep="\t", usecols=columns, na_values="n/a")[columns]
    return (df - df.mean()).fillna(0.0)


def load_masked_bold(path: Path, grid: Grid, chunk_trs: int = 100) -> np.ndarray:
    """Read a 4D BOLD file as T x V float32 over the group mask, streaming chunks of TRs.

    A full HCP run is ~5 GB as float32; streaming keeps peak memory near the T x V result.
    Time is the slowest axis on disk, so consecutive chunks are a forward read of the gzip.
    """
    img = nib.load(path, keep_file_open=True)
    grid.check_image(img, path.name)
    n_tr = img.shape[3]
    out = np.empty((n_tr, grid.n_voxels), dtype=np.float32)
    for t0 in range(0, n_tr, chunk_trs):
        t1 = min(t0 + chunk_trs, n_tr)
        chunk = np.asarray(img.dataobj[..., t0:t1], dtype=np.float32)
        out[t0:t1] = chunk[grid.mask].T
    return out


def denoise_run(row: pd.Series, grid: Grid, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (data T x V float32, valid V bool, fd T float32) for one manifest row.

    A stored voxel is valid for this run if it is inside the run's own fMRIPrep brain mask
    and has non-zero raw variance; invalid voxels are stored as 0.
    """
    run_mask_img = nib.load(row["mask"])
    grid.check_image(run_mask_img, row["mask"].name)
    in_run = (np.asarray(run_mask_img.dataobj) > 0)[grid.mask]

    raw = load_masked_bold(row["bold"], grid, cfg["io"]["chunk_trs"])
    valid = in_run & (raw.std(axis=0) > 0)

    confounds = select_confounds(row["confounds"], row["confounds_json"], cfg["confounds"])
    if len(confounds) != raw.shape[0]:
        raise ValueError(f"{row['run_id']}: {len(confounds)} confound rows vs {raw.shape[0]} TRs")

    den = cfg["denoise"]
    cleaned = signal.clean(
        raw[:, valid],
        detrend=den["detrend"],
        standardize=den["standardize"],
        confounds=confounds.to_numpy(),
        standardize_confounds="zscore_sample",
        low_pass=den["low_pass"],
        high_pass=None,                  # the cosine regressors do the high-pass
        t_r=float(row["tr"]),
        filter="butterworth" if den["low_pass"] else False,
    )
    del raw

    data = np.zeros((cleaned.shape[0], grid.n_voxels), dtype=np.float32)
    data[:, valid] = cleaned
    if not np.isfinite(data).all():
        raise ValueError(f"{row['run_id']}: non-finite values after cleaning")

    fd = pd.read_csv(row["confounds"], sep="\t", usecols=["framewise_displacement"],
                     na_values="n/a")["framewise_displacement"].fillna(0).to_numpy(np.float32)
    return data, valid, fd
