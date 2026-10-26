"""Smoke test for Temporal BIT: random input and a real T-TR window from the reference scan.

Prints tensor shapes at every stage, the parameter count and one B=8 forward/backward's peak
memory. Reads the data root only; the atlas and gradients live in this repo (cache/, gradients/).

    python smoke_test_temporal_bit.py
"""
from __future__ import annotations

import resource
import time

import numpy as np
import torch

import compute_gradients as cg
from positional_embedding import load_gradients
from rsfm.preprocessing.inventory import load_manifest
from rsfm.preprocessing.voxel_store import VoxelRun
from temporal_bit import TemporalBIT, TemporalBITConfig

REFERENCE_RUN = "sub-100206_ses-1_task-rest_dir-LR"


def show(name: str, model: TemporalBIT, x: torch.Tensor, **kw) -> torch.Tensor:
    with torch.no_grad():
        out, mid = model(x, return_intermediates=True, **kw)
    print(f"  {name}")
    print(f"    input x                  {tuple(x.shape)}  (B, T, V) {x.dtype}")
    print(f"    roi_keep                 {tuple(mid['roi_keep'].shape)}  (B, K)")
    print(f"    1. tokenizer   -> tokens {tuple(mid['tokens'].shape)}  (B, T, K, d)")
    print(f"    2. temporal    -> ROI    {tuple(mid['temporal'].shape)}  (B, K, d)")
    print(f"    3. spatial     -> out    {tuple(out.shape)}  (B, K, D)")
    assert torch.isfinite(out).all()
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    atlas = cg.load_atlas()
    grad_dir = cg.GRADIENT_DIR / cg.ATLAS_NAME
    if not (grad_dir / "runs" / f"{REFERENCE_RUN}.npy").exists():
        cg.main(["--placeholder"])
    group = load_gradients(cg.group_gradient_path())                    # (N, G) Brain-JEPA UKB
    subject = load_gradients(grad_dir / "runs" / f"{REFERENCE_RUN}.npy")  # HCP, single-subject placeholder

    cfg = TemporalBITConfig()
    T = cfg.n_timepoints
    model = TemporalBIT(cfg, atlas.labels, group).to(device).eval()
    N, V = model.n_rois, len(atlas.labels)
    print(f"atlas {atlas.name}: N={N} ROIs, V={V} stored voxels, "
          f"{int((atlas.labels > 0).sum())} labelled; device {device}")
    print("parameters:")
    for name, part in [("tokenizer", model.tokenizer), ("temporal", model.temporal),
                       ("spatial", model.spatial), ("total", model)]:
        print(f"  {name:10s} {sum(p.numel() for p in part.parameters()) / 1e6:7.2f} M")

    print("\n[random input]")
    x = torch.randn(2, T, V).half().to(device)
    show("all ROIs", model, x)
    keep = torch.stack([torch.randperm(N)[:300].sort().values for _ in range(2)]).to(device)
    show("K=300 kept ROIs (per sample)", model, x, roi_keep=keep)

    print("\n[real window from the reference scan]")
    manifest = load_manifest(cg.DATA_ROOT).set_index("run_id", drop=False)
    row = manifest.loc[REFERENCE_RUN]
    run = VoxelRun.load(cg.DATA_ROOT, row)
    print(f"  {REFERENCE_RUN}: data {run.data.shape} {run.data.dtype}, TR {row['tr']} s, "
          f"{int(run.valid.sum())} valid voxels, {run.data.shape[0] // T} non-overlapping T={T} windows")
    x = torch.from_numpy(np.array(run.data[0:T]))[None].to(device)                # (1, T, V) float16
    valid = torch.from_numpy(run.valid)[None].to(device)                          # (1, V)
    tr = torch.tensor([row["tr"]], device=device)
    show("group gradients", model, x, voxel_valid=valid, tr=tr)

    cfg_subject = TemporalBITConfig(gradient_source="subject")
    model_subject = TemporalBIT(cfg_subject, atlas.labels, group).to(device).eval()
    show("subject gradients", model_subject, x, voxel_valid=valid, tr=tr,
         gradients=subject[None].to(device))

    print("\n[one forward/backward, B=8, all ROIs]")
    model.train()
    x = torch.randn(8, T, V).half().to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model(x).square().mean().backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
        print(f"  {time.time() - t0:.2f} s, peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    else:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        print(f"  {time.time() - t0:.2f} s on CPU, peak process RSS {rss:.2f} GiB")
    print("\nOK (group gradients: Brain-JEPA UKB; subject gradients: single-subject placeholder)")


if __name__ == "__main__":
    main()
