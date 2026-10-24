"""Optional atlas step on top of the stored voxel data.

An atlas name is a ``_``-joined list of components, cortex and/or subcortex:
    schaefer<N>[n17]   Schaefer-2018, N in {100,...,1000}, 7 networks (or 17 with ``n17``)
    tian<S1..S4>       Tian subcortex, 3T
e.g. ``schaefer400_tianS3`` (450 regions), ``schaefer100_tianS1``, ``schaefer200n17``.

All sources are MNI152NLin2009cAsym 2 mm files, the same space and grid as the fMRIPrep
output, so preparing an atlas is a lookup, not a resampling. ``prepare_atlas`` reduces it to
one region label per stored voxel; ``parcellate`` then averages valid voxels per region.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import sparse

from .voxel_store import Grid

TEMPLATEFLOW_DIR = Path(os.environ.get("TEMPLATEFLOW_HOME", "~/.cache/templateflow")).expanduser()
TIAN_DIR = Path(os.environ.get("RSFM_TIAN_DIR", "~/.cache/rsfm/atlases/tian")).expanduser()
ATLAS_DIR = "atlas"

SCHAEFER_RE = re.compile(r"^schaefer(\d+)(n17)?$")
TIAN_RE = re.compile(r"^tian(S[1-4])$")


@dataclass
class Component:
    name: str
    structure: str          # "cortex" or "subcortex"
    image: Path
    names: list[str]        # region names for label values 1..len(names)


def _schaefer(n: int, networks: int) -> Component:
    tpl = TEMPLATEFLOW_DIR / "tpl-MNI152NLin2009cAsym"
    desc = f"{n}Parcels{networks}Networks"
    image = tpl / f"tpl-MNI152NLin2009cAsym_res-02_atlas-Schaefer2018_desc-{desc}_dseg.nii.gz"
    table = pd.read_csv(tpl / f"tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-{desc}_dseg.tsv", sep="\t")
    if table["index"].tolist() != list(range(1, n + 1)):
        raise ValueError(f"unexpected Schaefer label table for {desc}")
    return Component(f"schaefer{n}" + ("n17" if networks == 17 else ""), "cortex", image,
                     table["name"].tolist())


def _tian(scale: str) -> Component:
    image = TIAN_DIR / f"Tian_Subcortex_{scale}_3T_2009cAsym.nii.gz"
    names = (TIAN_DIR / f"Tian_Subcortex_{scale}_3T_label.txt").read_text().split()
    return Component(f"tian{scale}", "subcortex", image, names)


def parse_atlas_name(name: str) -> list[Component]:
    components = []
    for part in name.split("_"):
        if m := SCHAEFER_RE.match(part):
            components.append(_schaefer(int(m.group(1)), 17 if m.group(2) else 7))
        elif m := TIAN_RE.match(part):
            components.append(_tian(m.group(1)))
        else:
            raise ValueError(f"unknown atlas component {part!r} in {name!r}")
    return components


def prepare_atlas(name: str, grid: Grid, out_dir: Path) -> pd.DataFrame:
    """Combine the components into one label volume and one label per stored voxel.

    Writes ``atlas/<name>/{atlas.nii.gz, labels.npy, regions.tsv, atlas.json}`` and returns
    the regions table. Voxels claimed by two components are left unlabelled and counted.
    """
    combined = np.zeros(grid.shape, dtype=np.int32)
    rows, offset, overlaps = [], 0, {}
    for comp in parse_atlas_name(name):
        img = nib.load(comp.image)
        grid.check_image(img, comp.image.name)
        labels = np.rint(np.asarray(img.dataobj)).astype(np.int32)
        if labels.max() > len(comp.names):
            raise ValueError(f"{comp.image.name}: label {labels.max()} but only {len(comp.names)} names")

        overlap = (combined > 0) & (labels > 0)
        own = (labels > 0) & ~overlap
        combined[own] = labels[own] + offset
        combined[overlap] = 0
        overlaps[comp.name] = int(overlap.sum())
        for i, region in enumerate(comp.names, start=1):
            rows.append({"index": offset + i, "name": region, "component": comp.name,
                         "structure": comp.structure})
        offset += len(comp.names)

    regions = pd.DataFrame(rows)
    idx = regions["index"].to_numpy()
    regions["n_voxels"] = np.bincount(combined.ravel(), minlength=offset + 1)[idx]
    regions["n_voxels_in_mask"] = np.bincount(combined[grid.mask], minlength=offset + 1)[idx]
    regions["mask_coverage"] = regions["n_voxels_in_mask"] / regions["n_voxels"].clip(lower=1)

    adir = out_dir / ATLAS_DIR / name
    adir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(combined, grid.affine), adir / "atlas.nii.gz")
    np.save(adir / "labels.npy", combined[grid.mask])
    regions.to_csv(adir / "regions.tsv", sep="\t", index=False)
    with open(adir / "atlas.json", "w") as fh:
        json.dump({"name": name, "n_regions": offset,
                   "components": [c.name for c in parse_atlas_name(name)],
                   "overlap_voxels_unlabelled": overlaps}, fh, indent=2)
    return regions


@dataclass
class Atlas:
    name: str
    labels: np.ndarray      # V int32, 0 = no region; one per stored voxel
    regions: pd.DataFrame   # row r describes label r + 1

    @classmethod
    def load(cls, out_dir: Path, name: str) -> "Atlas":
        adir = out_dir / ATLAS_DIR / name
        return cls(name, np.load(adir / "labels.npy"), pd.read_csv(adir / "regions.tsv", sep="\t"))

    @property
    def n_regions(self) -> int:
        return len(self.regions)


def parcellate(data: np.ndarray, valid: np.ndarray, atlas: Atlas, zscore: bool = True,
               chunk_trs: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Average valid voxels per region: T x V -> T x R float32.

    Returns (timeseries, coverage) where coverage[r] is the fraction of region r's atlas
    voxels that are stored and valid in this run. Regions with no valid voxel are NaN.
    """
    use = valid & (atlas.labels > 0)
    cols = np.flatnonzero(use)
    weights = sparse.csr_matrix(
        (np.ones(len(cols), dtype=np.float32), (atlas.labels[cols] - 1, cols)),
        shape=(atlas.n_regions, data.shape[1]),
    )
    counts = np.asarray(weights.sum(axis=1)).ravel()

    ts = np.empty((data.shape[0], atlas.n_regions), dtype=np.float32)
    for t0 in range(0, data.shape[0], chunk_trs):        # data may be a float16 memmap
        chunk = np.asarray(data[t0:t0 + chunk_trs], dtype=np.float32)
        ts[t0:t0 + chunk_trs] = (weights @ chunk.T).T
    with np.errstate(invalid="ignore", divide="ignore"):
        ts /= counts
    ts[:, counts == 0] = np.nan

    if zscore:
        std = ts.std(axis=0, ddof=1)
        ok = std > 0
        ts[:, ok] = (ts[:, ok] - ts[:, ok].mean(axis=0)) / std[ok]

    coverage = counts / atlas.regions["n_voxels"].to_numpy().clip(min=1)
    return ts, coverage
