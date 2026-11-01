"""Find fMRIPrep runs and build the manifest (one row per run).

Every later step reads the manifest instead of globbing the filesystem. Exclusion is a flag,
not a deletion, so the motion threshold can change without reprocessing.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd

BOLD_SUFFIX_RE = re.compile(r"_space-[^_]+(?:_res-[^_]+)?_desc-preproc_bold\.nii\.gz$")
ENTITY_RE = re.compile(r"(?:^|_)(sub|ses|task|acq|dir|run)-([a-zA-Z0-9]+)")
ENTITIES = ("sub", "ses", "task", "acq", "dir", "run")
PATH_COLUMNS = ("bold", "bold_json", "mask", "confounds", "confounds_json")
NUMERIC_COLUMNS = ("tr", "n_tr", "mean_fd", "max_fd", "pct_high_fd")

MANIFEST_NAME = "manifest.tsv"


def run_files(bold: Path) -> dict:
    """Derive a run's sidecar paths from its preprocessed BOLD path.

    The confounds name is built from the full run prefix (including ``dir-``), which is
    exactly what nilearn 0.10.1's ``load_confounds`` gets wrong for HCP.
    """
    run_id = BOLD_SUFFIX_RE.sub("", bold.name)
    if run_id == bold.name:
        raise ValueError(f"not a preprocessed BOLD file name: {bold.name}")
    return {
        "run_id": run_id,
        "bold": bold,
        "bold_json": bold.with_name(bold.name.replace(".nii.gz", ".json")),
        "mask": bold.with_name(bold.name.replace("_desc-preproc_bold.nii.gz", "_desc-brain_mask.nii.gz")),
        "confounds": bold.with_name(f"{run_id}_desc-confounds_timeseries.tsv"),
        "confounds_json": bold.with_name(f"{run_id}_desc-confounds_timeseries.json"),
    }


def parse_entities(run_id: str) -> dict:
    found = dict(ENTITY_RE.findall(run_id))
    return {e: found.get(e) for e in ENTITIES}


def describe_run(bold: Path, fd_threshold: float) -> dict:
    row = run_files(bold)
    row.update(parse_entities(row["run_id"]))
    missing = [k for k in PATH_COLUMNS if not row[k].exists()]
    row["missing"] = ",".join(missing)
    if missing:
        return row

    # TR from the JSON sidecar: NIfTI headers written by some tools carry a wrong TR.
    with open(row["bold_json"]) as fh:
        row["tr"] = float(json.load(fh)["RepetitionTime"])
    row["n_tr"] = int(nib.load(bold).shape[3])

    fd = pd.read_csv(row["confounds"], sep="\t", usecols=["framewise_displacement"],
                     na_values="n/a")["framewise_displacement"].fillna(0).to_numpy()
    row["mean_fd"] = float(fd.mean())
    row["max_fd"] = float(fd.max())
    row["pct_high_fd"] = float(100 * (fd > fd_threshold).mean())
    return row


def build_manifest(cfg: dict, n_workers: int = 16, subjects: Iterable[str] | None = None) -> pd.DataFrame:
    """Manifest rows for all runs in fmriprep_dir, or only those of ``subjects``."""
    patterns = [cfg["bold_glob"]] if subjects is None else [
        cfg["bold_glob"].replace("sub-*", f"sub-{s}", 1) for s in subjects]
    bolds = sorted(b for pattern in patterns for b in cfg["fmriprep_dir"].glob(pattern))
    if not bolds:
        raise FileNotFoundError(f"no BOLD files match {cfg['fmriprep_dir'] / patterns[0]}")
    fd_threshold = cfg["qc"]["fd_threshold"]
    with ThreadPoolExecutor(n_workers) as pool:            # I/O bound: NFS reads
        rows = list(pool.map(lambda b: describe_run(b, fd_threshold), bolds))

    df = pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)
    if df["run_id"].duplicated().any():
        raise ValueError(f"duplicate run ids: {df.loc[df['run_id'].duplicated(), 'run_id'].tolist()}")
    df["exclude"] = (df["missing"] != "") | (df["mean_fd"] > cfg["qc"]["max_mean_fd"])
    return df


def save_manifest(df: pd.DataFrame, out_dir: Path) -> Path:
    """Replaces the file atomically: jobs and dataloaders may be reading it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / MANIFEST_NAME
    tmp = path.with_name(f"{MANIFEST_NAME}.tmp.{os.getpid()}")
    df.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, path)
    return path


def update_manifest(new_rows: pd.DataFrame, out_dir: Path) -> Path:
    """Add or replace these runs in the manifest, under a lock: workers on other nodes do the same."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{MANIFEST_NAME}.lock", "a+") as lock:
        fcntl.lockf(lock, fcntl.LOCK_EX)
        try:
            if (out_dir / MANIFEST_NAME).exists():
                old = load_manifest(out_dir)
                new_rows = pd.concat([old[~old["run_id"].isin(new_rows["run_id"])], new_rows], ignore_index=True)
            return save_manifest(new_rows.sort_values("run_id").reset_index(drop=True), out_dir)
        finally:
            fcntl.lockf(lock, fcntl.LOCK_UN)


def load_manifest(out_dir: Path) -> pd.DataFrame:
    # keep_default_na=False keeps entities like "ses-NA" and the empty "missing" field as strings.
    df = pd.read_csv(out_dir / MANIFEST_NAME, sep="\t", dtype=str, keep_default_na=False)
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in PATH_COLUMNS:
        df[col] = df[col].map(Path)
    df["exclude"] = df["exclude"].str.lower() == "true"
    return df


def summarize(df: pd.DataFrame) -> str:
    ok = df[df["missing"] == ""]
    lines = [
        f"runs: {len(df)}  subjects: {df['sub'].nunique()}  missing files: {(df['missing'] != '').sum()}",
        f"excluded (missing or mean FD > threshold): {df['exclude'].sum()}",
    ]
    if len(ok):
        lines.append(f"TR values: {sorted(ok['tr'].unique().tolist())}  "
                     f"n_TR range: {int(ok['n_tr'].min())}-{int(ok['n_tr'].max())}")
        lines.append(f"mean FD: median {np.median(ok['mean_fd']):.3f} mm, max {ok['mean_fd'].max():.3f} mm")
    return "\n".join(lines)
