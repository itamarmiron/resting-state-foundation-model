"""Windows of T TRs cut from the stored voxel runs, as input for Temporal BIT pretraining.

Each item is one window of one run:
    x      (T, V) float16   stored (denoised, voxelwise z-scored) BOLD, read from the run's memmap
    valid  (V,)   bool      the run's valid voxels (``_valid.npy``), for ``voxel_valid``
    tr     ()     float32   the run's TR in seconds (manifest, from the JSON sidecar)
    run    ()     int64     index into ``dataset.runs``, to find the run (and its subject) again
    start  ()     int64     first TR of the window
The default collate stacks these into the model's inputs: x (B, T, V), valid (B, V), tr (B,).

Two ways to cut windows:
    "random"  each run gives n_tr // T items per epoch, each at a uniformly random start, so an
              epoch sees every run in proportion to its length and windows straddle the
              non-overlapping grid. Starts come from torch's RNG, which DataLoader seeds per
              worker and per epoch.
    "grid"    fixed starts 0, stride, 2 * stride, ... (stride defaults to T): deterministic, for
              validation and feature extraction.

Nothing is normalized here: the stored data is already z-scored per voxel over the whole run,
and any normalization across voxels would mix ROIs before the spatial transformer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from rsfm.preprocessing.inventory import load_manifest
from rsfm.preprocessing.voxel_store import VoxelRun, is_done, load_grid


def select_runs(manifest: pd.DataFrame, data_root: Path, n_timepoints: int,
                subjects: Iterable[str] | None = None, include_excluded: bool = False) -> pd.DataFrame:
    """Manifest rows that are stored, long enough for one window and (by default) not flagged
    ``exclude``, optionally restricted to ``subjects`` (manifest ``sub``, without "sub-")."""
    runs = manifest[manifest["missing"] == ""]
    if not include_excluded:
        runs = runs[~runs["exclude"]]
    if subjects is not None:
        runs = runs[runs["sub"].isin([str(s) for s in subjects])]
    runs = runs[runs["n_tr"] >= n_timepoints]
    runs = runs[[is_done(data_root, row) for _, row in runs.iterrows()]]
    return runs.reset_index(drop=True)


class WindowDataset(Dataset):
    """T-TR windows of stored voxel runs; see the module docstring for the item layout."""

    def __init__(self, data_root: str | Path, n_timepoints: int, subjects: Iterable[str] | None = None,
                 mode: str = "random", stride: int | None = None, include_excluded: bool = False,
                 runs: pd.DataFrame | None = None):
        """
        data_root:  output_dir of the preprocessing (manifest.tsv, grid/, voxel/)
        subjects:   restrict to these subjects (e.g. the train split); None = all stored runs
        mode:       "random" or "grid"
        stride:     grid mode only, TRs between window starts (default n_timepoints)
        runs:       use these manifest rows instead of selecting them (skips ``select_runs``)
        """
        assert mode in ("random", "grid"), mode
        assert stride is None or mode == "grid", "stride only applies to grid mode"
        self.data_root = Path(data_root)
        self.n_timepoints, self.mode = n_timepoints, mode
        self.stride = stride or n_timepoints
        self.grid = load_grid(self.data_root)
        if runs is None:
            runs = select_runs(load_manifest(self.data_root), self.data_root, n_timepoints,
                               subjects, include_excluded)
        self.runs = runs.reset_index(drop=True)
        if len(self.runs) == 0:
            raise ValueError(f"no stored runs with >= {n_timepoints} TRs in {self.data_root}")

        n_tr = self.runs["n_tr"].to_numpy(int)
        if mode == "random":
            per_run = n_tr // n_timepoints
        else:
            per_run = (n_tr - n_timepoints) // self.stride + 1
        self.item_run = np.repeat(np.arange(len(self.runs)), per_run)          # run of each item
        self.item_slot = np.concatenate([np.arange(k) for k in per_run])        # k-th window of that run
        self._open: dict[int, VoxelRun] = {}                                    # per-process memmaps

    def __len__(self) -> int:
        return len(self.item_run)

    def __getstate__(self):
        # Memmaps are reopened in each DataLoader worker rather than pickled.
        state = self.__dict__.copy()
        state["_open"] = {}
        return state

    def _run(self, i: int) -> VoxelRun:
        if i not in self._open:
            self._open[i] = VoxelRun.load(self.data_root, self.runs.iloc[i], self.grid)
        return self._open[i]

    def __getitem__(self, idx: int) -> dict:
        i = int(self.item_run[idx])
        run = self._run(i)
        T = self.n_timepoints
        if self.mode == "random":
            start = int(torch.randint(run.data.shape[0] - T + 1, ()))
        else:
            start = int(self.item_slot[idx]) * self.stride
        return {
            "x": torch.from_numpy(np.array(run.data[start:start + T])),       # (T, V) float16
            "valid": torch.from_numpy(run.valid),                               # (V,) bool
            "tr": torch.tensor(float(self.runs.at[i, "tr"]), dtype=torch.float32),
            "run": torch.tensor(i),
            "start": torch.tensor(start),
        }
