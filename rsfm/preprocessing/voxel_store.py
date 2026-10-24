"""Shared grid + group brain mask, and per-run voxel storage.

Each run is stored as a T x V float16 matrix over the voxels of one group mask, in C order
of the 3D grid (``volume[mask]``). That makes every run share the same voxel axis, so an atlas
becomes one label per stored voxel and full volumes can be rebuilt from the mask.

The group mask is frozen once built: runs added later (fMRIPrep still finishing) are stored
on the same voxel axis.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

GRID_DIR = "grid"
VOXEL_DIR = "voxel"
QC_DIR = "qc"


@dataclass
class Grid:
    mask: np.ndarray        # 3D bool, the group mask
    affine: np.ndarray      # 4x4 voxel -> MNI152NLin2009cAsym mm

    @property
    def shape(self) -> tuple:
        return self.mask.shape

    @property
    def n_voxels(self) -> int:
        return int(self.mask.sum())

    def check_image(self, img: nib.spatialimages.SpatialImage, what: str) -> None:
        if img.shape[:3] != self.shape or not np.allclose(img.affine, self.affine, atol=1e-3):
            raise ValueError(f"{what} is not on the group grid: shape {img.shape[:3]} vs {self.shape}, "
                             f"affine\n{img.affine}\nvs\n{self.affine}")


def build_group_mask(manifest: pd.DataFrame, template_mask: Path,
                     min_run_fraction: float) -> tuple[Grid, dict]:
    """Voxels inside the template brain mask and in-brain in >= min_run_fraction of runs.

    Voxels in the group mask but outside one run's own mask are stored as invalid for that
    run (see ``denoise_run``), so a lenient fraction keeps dropout-prone regions available.
    """
    template = nib.load(template_mask)
    grid = Grid(mask=np.asarray(template.dataobj) > 0, affine=template.affine)

    runs = manifest[manifest["missing"] == ""]
    counts = np.zeros(grid.shape, dtype=np.int32)
    for path in runs["mask"]:
        img = nib.load(path)
        grid.check_image(img, path.name)
        counts += np.asarray(img.dataobj) > 0
    fraction = counts / len(runs)

    mask = grid.mask & (fraction >= min_run_fraction)
    info = {
        "space": "MNI152NLin2009cAsym",
        "shape": list(grid.shape),
        "affine": grid.affine.tolist(),
        "n_voxels": int(mask.sum()),
        "voxel_order": "C order of the 3D grid, i.e. volume[mask]",
        "template_mask": str(template_mask),
        "n_template_voxels": int(grid.mask.sum()),
        "min_run_fraction": min_run_fraction,
        "n_runs_used": int(len(runs)),
        "n_runs_voxels_outside_template": int(((fraction > 0) & ~grid.mask).sum()),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    return Grid(mask=mask, affine=grid.affine), info


def save_grid(out_dir: Path, grid: Grid, info: dict) -> None:
    grid_dir = out_dir / GRID_DIR
    if (grid_dir / "grid.json").exists():
        raise FileExistsError(f"{grid_dir} already exists; the group mask is frozen once built "
                              f"(delete it deliberately to rebuild, then re-run all runs)")
    grid_dir.mkdir(parents=True)
    nib.save(nib.Nifti1Image(grid.mask.astype(np.uint8), grid.affine), grid_dir / "mask.nii.gz")
    with open(grid_dir / "grid.json", "w") as fh:
        json.dump(info, fh, indent=2)


def load_grid(out_dir: Path) -> Grid:
    img = nib.load(out_dir / GRID_DIR / "mask.nii.gz")
    return Grid(mask=np.asarray(img.dataobj) > 0, affine=img.affine)


def run_dir(out_dir: Path, row: pd.Series | dict, kind: str = VOXEL_DIR) -> Path:
    parts = [f"sub-{row['sub']}"] + ([f"ses-{row['ses']}"] if row.get("ses") else [])
    return out_dir.joinpath(kind, *parts)


def run_outputs(out_dir: Path, row: pd.Series | dict) -> dict:
    vdir, qdir = run_dir(out_dir, row), run_dir(out_dir, row, QC_DIR)
    return {
        "bold": vdir / f"{row['run_id']}_bold.npy",
        "valid": vdir / f"{row['run_id']}_valid.npy",
        "fd": qdir / f"{row['run_id']}_fd.npy",
    }


def is_done(out_dir: Path, row: pd.Series | dict) -> bool:
    # bold is written last, so its presence marks a complete run.
    return run_outputs(out_dir, row)["bold"].exists()


def _atomic_save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


def save_run(out_dir: Path, row: pd.Series | dict, data: np.ndarray, valid: np.ndarray,
             fd: np.ndarray) -> None:
    paths = run_outputs(out_dir, row)
    _atomic_save(paths["valid"], valid.astype(bool))
    _atomic_save(paths["fd"], fd.astype(np.float32))
    _atomic_save(paths["bold"], data.astype(np.float16))


@dataclass
class VoxelRun:
    """One stored run. ``data`` is a read-only memmap, T x V float16."""
    run_id: str
    data: np.ndarray
    valid: np.ndarray
    fd: np.ndarray
    grid: Grid

    @classmethod
    def load(cls, out_dir: Path, row: pd.Series | dict, grid: Grid | None = None) -> "VoxelRun":
        paths = run_outputs(out_dir, row)
        grid = grid if grid is not None else load_grid(out_dir)
        data = np.load(paths["bold"], mmap_mode="r")
        if data.shape[1] != grid.n_voxels:
            raise ValueError(f"{row['run_id']}: {data.shape[1]} voxels stored, grid has {grid.n_voxels}")
        return cls(row["run_id"], data, np.load(paths["valid"]), np.load(paths["fd"]), grid)

    def to_volume(self, t: int | slice | None = None, fill: float = 0.0) -> np.ndarray:
        """Rebuild X x Y x Z (single t) or X x Y x Z x T volumes; invalid voxels get ``fill``."""
        frames = self.data[t if t is not None else slice(None)]
        frames = np.where(self.valid, frames, fill).astype(np.float32)
        if frames.ndim == 1:
            vol = np.full(self.grid.shape, fill, dtype=np.float32)
            vol[self.grid.mask] = frames
            return vol
        vol = np.full(self.grid.shape + (frames.shape[0],), fill, dtype=np.float32)
        vol[self.grid.mask] = frames.T
        return vol
