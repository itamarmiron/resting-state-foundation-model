"""Denoise one subject: the job that scripts/preprocess_manager.sbatch submits.

    python -m rsfm.preprocessing.worker <subject> --config configs/preprocessing/hcp.yaml

Adds the subject's runs to the manifest, then denoises those not stored yet. Runs are saved
atomically and skipped when present, so a cancelled job loses at most the run in progress.
"""
from __future__ import annotations

import argparse
import os
import resource
import signal
import socket
import sys
import time
import traceback

from .config import load_config
from .denoise import denoise_run
from .inventory import build_manifest, update_manifest
from .status import DEFAULT_DIR, Paths, job_done, job_failed, job_started, read_table
from .voxel_store import is_done, load_grid, save_run


def process_subject(cfg: dict, subject: str, expected_runs: int) -> tuple[int, int]:
    """Returns (stored runs, excluded runs). Raises if a run is missing or can't be denoised."""
    runs = build_manifest(cfg, subjects=[subject])
    if len(runs) != expected_runs:
        raise ValueError(f"{len(runs)} runs in fmriprep_dir, expected {expected_runs}")
    missing = runs[runs["missing"] != ""]
    if len(missing):
        raise ValueError(f"missing files: {dict(zip(missing['run_id'], missing['missing']))}")
    update_manifest(runs, cfg["output_dir"])

    out, grid = cfg["output_dir"], load_grid(cfg["output_dir"])
    for _, row in runs.iterrows():
        if is_done(out, row):
            print(f"already stored {row['run_id']}", flush=True)
            continue
        t0 = time.time()
        data, valid, fd = denoise_run(row, grid, cfg)
        save_run(out, row, data, valid, fd)
        print(f"done {row['run_id']}: T={data.shape[0]} valid={valid.mean():.1%} ({time.time() - t0:.0f}s)", flush=True)
    return len(runs), int(runs["exclude"].sum())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("subject")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dir", default=str(DEFAULT_DIR), help="status directory")
    a = ap.parse_args(argv)
    p = Paths(a.dir)
    subject = a.subject
    t0 = time.time()

    def on_signal(signum, frame):
        job_failed(p, subject, signal.Signals(signum).name)
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    job_started(p, subject, os.environ.get("SLURM_JOB_ID", "manual"), socket.gethostname().split(".")[0])
    try:
        expected = int(read_table(p)[subject]["expected_runs"])
        stored, excluded = process_subject(load_config(a.config), subject, expected)
    except Exception as error:
        print(traceback.format_exc(), flush=True)
        job_failed(p, subject, f"{type(error).__name__}: {error}"[:300])
        sys.exit(1)
    minutes = (time.time() - t0) / 60
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    job_done(p, subject, stored, excluded, minutes, note=f"peak {peak:.1f} GiB")
    print(f"subject {subject} done: {stored} runs, {minutes:.1f} min, peak {peak:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
