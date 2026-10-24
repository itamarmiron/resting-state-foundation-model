# Project status

Last updated: 2026-10-24

What exists in this repo and what is left to do. Update it at the end of every session that changes anything (see CLAUDE.md).

## Done

### Preprocessing (`rsfm/preprocessing/`)
- CLI steps `inventory` → `mask` → `run` → optional `atlas` / `parcellate`, with unit tests (`tests/test_preprocessing.py`). Design: `docs/PREPROCESSING_PLAN.md`.
- HCP so far:
  - `manifest.tsv` (807 runs) and the frozen group mask (`grid/`, V = 234,961 voxels) exist.
  - Only **sub-100206 ses-1 (LR, RL)** is processed. The full processing run has **not** started, and the HCP download is still in progress.

### Temporal BIT encoder (slide 1 of `Temporal_BIT_JEPA_widescreen.pdf`)
- `temporal_bit.py`: `TemporalBITConfig` (T = 20 by default), `BrainTokenizer`, `TemporalTransformer`, `SpatialTransformer`, `TemporalBIT`.
  - `(B, T, V)` → `(B, T, N, 512)` → `(B, N, 512)` → `(B, N, D)`; 39 M parameters.
  - Voxel embeddings are low rank (64), one per labelled MNI voxel.
  - The tokenizer never builds a `(B, T, V, d)` tensor.
  - Invalid voxels are masked.
  - `roi_keep` removes ROIs completely; tested to leave no influence and to mix no ROIs before the spatial stage.
- `positional_embedding.py`: temporal sin-cos (time in units of the reference TR, so any T works) and the brain-gradient embedding, both ported from Brain-JEPA.
- `compute_gradients.py`: BrainSpace-style gradients (numpy port), in two versions:
  - `group`: one fixed `(N, 30)` matrix;
  - `subject`: per run, from whole-scan FC, aligned to the group gradients.
- Atlas `tianS3_schaefer400n17`: N = 450, Tian S3 first, then Schaefer-400 17-network. **The cerebellum is excluded for now.** It is built locally in `cache/`; the data root has no `atlas/` folder yet.
- **Placeholder** gradients from sub-100206 only, in `gradients/tianS3_schaefer400n17/`. Sanity checks look right: G1 runs somatomotor ↔ default mode, G2 is visual, and the LR and RL runs agree at r ≈ 0.6–0.7.
- `tests/test_temporal_bit.py` (16 tests; 26 in total with preprocessing, all passing) and `smoke_test_temporal_bit.py` (random input and a real window).
  - CPU only so far: a B = 8 forward/backward took 18 s with a peak process memory of 8.9 GiB.

## To do

### Data (blocked on the download or on decisions)
- [ ] The HCP download finishes, then the full processing run (`sbatch scripts/preprocess_hcp.sbatch`). This needs the user's explicit go-ahead.
- [ ] Decide whether the frozen group mask is final. It must not change after training starts, because V, the stored runs and the voxel embeddings all depend on it.
- [ ] Optional: run the `atlas` step on the data root, which adds one ~1 MB folder (needs approval).
- [ ] Define the train/test subject split.
- [ ] Compute the real group gradients: `compute_gradients.py --group --train-subjects <file>`. If version 2 is used, also compute the per-run gradients (`--subject`).

### Encoder leftovers
- [ ] Measure GPU memory and speed with the smoke test on a GPU that is visible. This session had `CUDA_VISIBLE_DEVICES` set to empty.
- [ ] Cross-check the numpy gradient port against BrainSpace (needs `pip install brainspace`).
- [ ] Only if version 2 is used: choose a leak option and, if needed, add "gradients from another run of the same subject".
- [ ] Only if profiling shows the need: speed up the tokenizer by grouping ROIs by size.

### JEPA pretraining (slide 2), not started
- [ ] A dataloader that cuts T = 20 windows from the stored runs and returns `(x, valid, tr)`.
- [ ] A mask sampler for hidden ROIs: first read Brain-JEPA's `src/masks/spatialtemporal_multiblock.py` and copy its ratios.
- [ ] A predictor (mask token + gradient positional embedding), an EMA target encoder (`roi_keep=None`), the loss and the training loop. Use the group gradients.
- [ ] Later: cross-time targets from several windows of the same scan (the minute grid in slide 2).

## Open decisions
- Is the group mask final?
- The train/test split.
- Per-subject gradients: whether to use them, and with which leak option.
- When to move from T = 20 (14.4 s) to the slide-2 "minute" (≈ 83 TRs).
- Permission to install BrainSpace.
- Adding the cerebellum later: this changes N, the voxel embeddings and the gradients.
