"""Brain gradients for Temporal BIT's Brain Gradient Positioning, computed from HCP.

Two versions (see the model plan):
    group    one fixed (N, G) matrix: diffusion-map embedding of the Fisher-z average FC over
             training runs. Every subject and window gets the same gradients.
    subject  one (N, G) matrix per run, from that run's whole-scan FC (all TRs), Procrustes-aligned
             to the group gradients so that components and signs match across runs.

The pipeline: ROI time series (``rsfm.preprocessing.atlas.parcellate``) -> Pearson FC ->
affinity -> diffusion maps. The affinity and embedding are numpy ports of BrainSpace
(https://github.com/MICA-MNI/BrainSpace), which Brain-JEPA used for its UKB gradients:
    brainspace/gradient/kernels.py     compute_affinity, _build_kernel ('normalized_angle')
    brainspace/gradient/utils.py       dominant_set
    brainspace/gradient/embedding.py   diffusion_mapping
    brainspace/gradient/alignment.py   procrustes
Settings: keep the top 10 % of each FC row (sparsity 0.9), normalized-angle kernel, alpha 0.5,
automatic diffusion time, 30 components. (BrainSpace's ``GradientMaps`` default kernel is None,
i.e. the thresholded FC itself; normalized angle is the common choice in its tutorials.)

Reads the data root only. Outputs go to ``gradients/<atlas>/`` in this repo:
    group_<tag>.npy               (N, G)
    runs/<run_id>.npy             (N, G), one per run (subject version)

Usage:
    python compute_gradients.py --placeholder                    # group + subject, sub-100206 only
    python compute_gradients.py --group --train-subjects subs.txt --tag train
    python compute_gradients.py --subject --reference gradients/<atlas>/group_train.npy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rsfm.preprocessing.atlas import Atlas, prepare_atlas, parcellate
from rsfm.preprocessing.inventory import load_manifest
from rsfm.preprocessing.voxel_store import VoxelRun, is_done, load_grid

DATA_ROOT = Path("/net/mraid16/mnt/pool-16/iranig_workspace/HCP_1200/derivatives/rsfm_voxel")
REPO = Path(__file__).resolve().parent
CACHE_DIR = REPO / "cache"              # local atlas copy until the atlas step runs on the data root
GRADIENT_DIR = REPO / "gradients"
ATLAS_NAME = "tianS3_schaefer400n17"     # Brain-JEPA's layout: Tian S3 (50) then Schaefer-400 17N
PLACEHOLDER_SUBJECT = "100206"


def load_atlas(name: str = ATLAS_NAME, data_root: Path = DATA_ROOT, cache_dir: Path = CACHE_DIR) -> Atlas:
    """The atlas from the data root if it was prepared there, else a local copy (built once).

    Building the local copy only reads the data root's frozen group mask.
    """
    if (data_root / "atlas" / name / "labels.npy").exists():
        return Atlas.load(data_root, name)
    grid = load_grid(data_root)
    if not (cache_dir / "atlas" / name / "labels.npy").exists():
        prepare_atlas(name, grid, cache_dir)
    atlas = Atlas.load(cache_dir, name)
    assert len(atlas.labels) == grid.n_voxels, "local atlas was built on a different group mask"
    return atlas


def processed_runs(data_root: Path = DATA_ROOT, subjects: list[str] | None = None) -> pd.DataFrame:
    """Manifest rows whose voxel data is stored, optionally restricted to some subjects."""
    manifest = load_manifest(data_root)
    if subjects is not None:
        manifest = manifest[manifest["sub"].isin(subjects)]
    done = manifest.apply(lambda row: is_done(data_root, row), axis=1)
    return manifest[done]


def run_fc(data_root: Path, row: pd.Series, atlas: Atlas) -> np.ndarray:
    """Whole-run Pearson FC between ROI time series: (N, N); NaN rows for ROIs with no valid voxel."""
    run = VoxelRun.load(data_root, row)
    ts, _ = parcellate(run.data, run.valid, atlas)               # (T, N)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.corrcoef(ts.T)


# --- BrainSpace ports --------------------------------------------------------------------------

def compute_affinity(fc: np.ndarray, sparsity: float = 0.9) -> np.ndarray:
    """Row-wise top-(1 - sparsity) thresholding, then the normalized-angle kernel: (N, N) -> (N, N).

    Port of brainspace.gradient.kernels.compute_affinity(kernel='normalized_angle',
    pre_sparsify=True, non_negative=True) with utils.dominant_set(k=1 - sparsity).
    """
    n = fc.shape[1]
    k = int((1 - sparsity) * n)
    keep = np.argpartition(fc, n - k, axis=1)[:, -k:]           # top-k entries of every row
    x = np.zeros_like(fc)
    rows = np.arange(fc.shape[0])[:, None]
    x[rows, keep] = fc[rows, keep]

    unit = x / np.linalg.norm(x, axis=1, keepdims=True)
    cos = np.clip(unit @ unit.T, -1.0, 1.0)                      # cosine similarity of rows
    affinity = 1 - np.arccos(cos) / np.pi
    affinity[affinity < 0] = 0
    return affinity


def diffusion_mapping(affinity: np.ndarray, n_components: int = 30, alpha: float = 0.5,
                      diffusion_time: float = 0) -> np.ndarray:
    """Diffusion-map embedding of a symmetric affinity matrix: (N, N) -> (N, n_components).

    Port of brainspace.gradient.embedding.diffusion_mapping. BrainSpace takes eigenvectors of the
    row-normalized (non-symmetric) transition matrix P = D^-1 A; here we use its symmetric
    conjugate D^-1/2 A D^-1/2 (same eigenvalues, eigenvectors mapped back by D^-1/2), which
    gives the same embedding with a symmetric eigensolver.
    """
    a = affinity.astype(np.float64)
    if alpha > 0:                                                # anisotropic normalization
        d = a.sum(axis=1) ** -alpha
        a = d[:, None] * a * d[None, :]
    d = a.sum(axis=1)
    sym = a / np.sqrt(np.outer(d, d))
    lambdas, vectors = np.linalg.eigh(sym)
    lambdas, vectors = lambdas[::-1][:n_components + 1], vectors[:, ::-1][:, :n_components + 1]
    psi = vectors / np.sqrt(d)[:, None]                          # right eigenvectors of P
    psi = psi / psi[:, [0]]                                      # first one is constant
    lambdas, psi = lambdas[1:], psi[:, 1:]
    lambdas = lambdas / (1 - lambdas) if diffusion_time <= 0 else lambdas ** diffusion_time
    return psi * lambdas[None, :]


def procrustes(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotate ``source`` (N, G) onto ``target`` (N, G): port of brainspace.gradient.alignment.procrustes."""
    u, _, vt = np.linalg.svd(source.T @ target)
    return source @ (u @ vt)


# --- the two versions --------------------------------------------------------------------------

def fix_signs(grad: np.ndarray) -> np.ndarray:
    """Deterministic sign per component: the largest-magnitude loading is positive."""
    peak = grad[np.abs(grad).argmax(axis=0), np.arange(grad.shape[1])]
    return grad * np.sign(peak)[None, :]


def group_gradients(fcs: list[np.ndarray], n_components: int = 30) -> np.ndarray:
    """Fisher-z average FC over runs (NaN-aware), then diffusion maps: -> (N, G)."""
    z = np.stack([np.arctanh(np.clip(fc, -0.999999, 0.999999)) for fc in fcs])
    with np.errstate(invalid="ignore"):
        mean_fc = np.tanh(np.nanmean(z, axis=0))
    if np.isnan(mean_fc).any():
        raise ValueError("some ROI pairs have no valid data in any run")
    np.fill_diagonal(mean_fc, 1.0)
    return fix_signs(diffusion_mapping(compute_affinity(mean_fc), n_components))


def subject_gradients(fc: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """One run's gradients, Procrustes-aligned to ``reference`` (N, G): -> (N, G).

    ROIs without valid data in this run are embedded without them and get the reference row.
    """
    ok = ~np.isnan(fc).all(axis=1)
    sub_fc = fc[np.ix_(ok, ok)]
    grad = diffusion_mapping(compute_affinity(sub_fc), reference.shape[1])
    out = reference.copy()
    out[ok] = procrustes(grad, reference[ok])
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--atlas", default=ATLAS_NAME)
    parser.add_argument("--group", action="store_true", help="compute the group gradients")
    parser.add_argument("--subject", action="store_true", help="compute per-run gradients")
    parser.add_argument("--placeholder", action="store_true",
                        help=f"group + subject from sub-{PLACEHOLDER_SUBJECT}'s processed runs only")
    parser.add_argument("--train-subjects", type=Path, help="text file, one subject id per line (group)")
    parser.add_argument("--tag", default="train", help="group output name: group_<tag>.npy")
    parser.add_argument("--reference", type=Path, help="group gradients to align to (subject)")
    parser.add_argument("--n-components", type=int, default=30)
    args = parser.parse_args(argv)

    out_dir = GRADIENT_DIR / args.atlas
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    atlas = load_atlas(args.atlas)

    if args.placeholder:
        args.group = args.subject = True
        args.tag = "placeholder"
        group_runs = subject_runs = processed_runs(subjects=[PLACEHOLDER_SUBJECT])
    else:
        train = args.train_subjects.read_text().split() if args.train_subjects else None
        if args.group and train is None:
            parser.error("--group needs --train-subjects (the group FC must exclude test subjects)")
        group_runs = processed_runs(subjects=train) if args.group else None
        subject_runs = processed_runs() if args.subject else None

    fcs = {}
    def fc_of(row):
        if row["run_id"] not in fcs:
            print(f"FC {row['run_id']}", flush=True)
            fcs[row["run_id"]] = run_fc(DATA_ROOT, row, atlas)
        return fcs[row["run_id"]]

    group_path = out_dir / f"group_{args.tag}.npy"
    if args.group:
        grad = group_gradients([fc_of(row) for _, row in group_runs.iterrows()], args.n_components)
        np.save(group_path, grad.astype(np.float32))
        print(f"group gradients {grad.shape} from {len(group_runs)} runs -> {group_path}")

    if args.subject:
        ref_path = args.reference or group_path
        reference = np.load(ref_path).astype(np.float64)
        for _, row in subject_runs.iterrows():
            grad = subject_gradients(fc_of(row), reference)
            np.save(out_dir / "runs" / f"{row['run_id']}.npy", grad.astype(np.float32))
        print(f"subject gradients for {len(subject_runs)} runs (aligned to {ref_path}) -> {out_dir / 'runs'}")


if __name__ == "__main__":
    main()
