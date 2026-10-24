"""Command line: python -m rsfm.preprocessing.cli <step> --config configs/preprocessing/hcp.yaml

Steps, in order:
    inventory    find runs, write manifest.tsv
    mask         build the frozen group mask (grid/)
    run          denoise + store voxel data (shardable over a Slurm array, resumable)
    atlas        prepare an atlas on the stored voxels (optional)
    parcellate   cache atlas time series for stored runs (optional; shardable, resumable)
"""
from __future__ import annotations

import argparse
import os
import time
import traceback

import numpy as np

from . import atlas as atlas_mod
from .config import load_config
from .denoise import denoise_run
from .inventory import build_manifest, load_manifest, save_manifest, summarize
from .voxel_store import (VoxelRun, build_group_mask, is_done, load_grid, save_grid,
                          save_run)


def _select_runs(manifest, args):
    # High-motion runs are stored too: exclusion is a manifest flag applied by consumers.
    runs = manifest[manifest["missing"] == ""]
    if args.skip_excluded:
        runs = runs[~runs["exclude"]]
    if args.run_id:
        runs = runs[runs["run_id"].isin(args.run_id)]
    if args.num_shards > 1:
        shard = args.shard if args.shard is not None else os.environ.get("SLURM_ARRAY_TASK_ID")
        if shard is None or not 0 <= int(shard) < args.num_shards:
            raise SystemExit(f"--num-shards {args.num_shards} needs a shard index in [0, {args.num_shards})")
        runs = runs.iloc[int(shard)::args.num_shards]
    if args.limit:
        runs = runs.head(args.limit)
    return runs


def cmd_inventory(cfg, args):
    df = build_manifest(cfg, n_workers=args.workers)
    path = save_manifest(df, cfg["output_dir"])
    print(summarize(df))
    print(f"wrote {path}")


def cmd_mask(cfg, args):
    manifest = load_manifest(cfg["output_dir"])
    grid, info = build_group_mask(manifest, cfg["template_mask"], cfg["group_mask"]["min_run_fraction"])
    save_grid(cfg["output_dir"], grid, info)
    print(f"group mask: {info['n_voxels']} voxels from {info['n_runs_used']} runs "
          f"(template mask {info['n_template_voxels']})")


def cmd_run(cfg, args):
    out = cfg["output_dir"]
    grid = load_grid(out)
    runs = _select_runs(load_manifest(out), args)
    print(f"{len(runs)} runs selected")
    failed = 0
    for _, row in runs.iterrows():
        if is_done(out, row) and not args.overwrite:
            continue
        t0 = time.time()
        try:
            data, valid, fd = denoise_run(row, grid, cfg)
            save_run(out, row, data, valid, fd)
            print(f"done {row['run_id']}: T={data.shape[0]} valid={valid.mean():.1%} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        except Exception:
            failed += 1
            print(f"FAILED {row['run_id']}\n{traceback.format_exc()}", flush=True)
    if failed:
        raise SystemExit(f"{failed} runs failed")


def cmd_atlas(cfg, args):
    regions = atlas_mod.prepare_atlas(args.name, load_grid(cfg["output_dir"]), cfg["output_dir"])
    low = regions[regions["mask_coverage"] < 0.5]
    print(f"{args.name}: {len(regions)} regions; median mask coverage "
          f"{regions['mask_coverage'].median():.1%}; regions under 50%: {low['name'].tolist() or 'none'}")


def cmd_parcellate(cfg, args):
    out = cfg["output_dir"]
    grid = load_grid(out)
    atlas = atlas_mod.Atlas.load(out, args.name)
    ts_dir = out / atlas_mod.ATLAS_DIR / args.name / "timeseries"
    ts_dir.mkdir(parents=True, exist_ok=True)
    for _, row in _select_runs(load_manifest(out), args).iterrows():
        target = ts_dir / f"{row['run_id']}_timeseries.npy"
        if not is_done(out, row) or (target.exists() and not args.overwrite):
            continue
        run = VoxelRun.load(out, row, grid)
        ts, coverage = atlas_mod.parcellate(run.data, run.valid, atlas)
        np.save(target, ts)
        np.save(ts_dir / f"{row['run_id']}_coverage.npy", coverage.astype(np.float32))
        print(f"done {row['run_id']}: {ts.shape}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["inventory", "mask", "run", "atlas", "parcellate"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", help="override output_dir (e.g. a scratch dir for tests)")
    parser.add_argument("--name", default="schaefer400_tianS3", help="atlas name (atlas, parcellate)")
    parser.add_argument("--run-id", nargs="*", help="only these runs")
    parser.add_argument("--shard", type=int, help="shard index (default: $SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, help="process at most this many runs")
    parser.add_argument("--skip-excluded", action="store_true", help="skip runs flagged exclude in the manifest")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=16, help="I/O threads for inventory")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.output_dir)
    {"inventory": cmd_inventory, "mask": cmd_mask, "run": cmd_run,
     "atlas": cmd_atlas, "parcellate": cmd_parcellate}[args.step](cfg, args)


if __name__ == "__main__":
    main()
