# Project status

Last updated: 2026-10-29

What exists in this repo and what is left to do. Update it at the end of every session that changes anything (see CLAUDE.md).

## Done

### Preprocessing (`rsfm/preprocessing/`)
- CLI steps `inventory` → `mask` → `run` → optional `atlas` / `parcellate`, with unit tests (`tests/test_preprocessing.py`). Design: `docs/PREPROCESSING_PLAN.md`.
- HCP so far:
  - `manifest.tsv` (807 runs) and the frozen group mask (`grid/`, V = 234,961 voxels) exist.
  - Only **sub-100206 ses-1 (LR, RL)** is processed. The full processing run has **not** started, and the HCP download is still in progress.

### Temporal BIT encoder (slide 1 of `Temporal_BIT_JEPA_widescreen.pdf`)
- `temporal_bit.py`: `TemporalBITConfig` (T = 20 by default), `BrainTokenizer`, `TemporalTransformer`, `SpatialTransformer`, `TemporalBIT`.
  - `(B, T, V)` → `(B, T, N, 512)` → `(B, N, 512)` → `(B, N, D)`; 102 M parameters, of which 72 M are the voxel embeddings.
  - Voxel embeddings: one per labelled MNI voxel (V′ = 140,526), shared across subjects and TRs.
    - Default `voxel_rank=None`: a free 512-dim row per voxel, as in Brain-IT. This was checked against `brainit-fmri/models/decoder_models.py`: Brain-IT uses a full-rank `(voxels, 1024)` table, but with separate rows per subject.
    - Optional `voxel_rank=r`: our own low-rank version, codes `(V′, r)` @ shared `W_up (r, 512)`. With r = 64 it has 39 M parameters in total. Kept to test later.
  - The tokenizer never builds a `(B, T, V, d)` tensor.
  - Invalid voxels are masked.
  - `roi_keep` removes ROIs completely; tested to leave no influence and to mix no ROIs before the spatial stage.
- `positional_embedding.py`: temporal sin-cos (time in units of the reference TR, so any T works) and the brain-gradient embedding, both ported from Brain-JEPA.
- `compute_gradients.py`: BrainSpace-style gradients (numpy port), in two versions:
  - `group`: one fixed `(N, 30)` matrix;
  - `subject`: per run, from whole-scan FC, aligned to the group gradients.
- Atlas `tianS3_schaefer400n17`: N = 450, Tian S3 first, then Schaefer-400 17-network. **The cerebellum is excluded for now.** It is built locally in `cache/`; the data root has no `atlas/` folder yet.
- **Group gradients in use: Brain-JEPA's UKB gradients** (`gradients/tianS3_schaefer400n17/brainjepa_ukb.csv`, 450 × 30, copied unchanged from Brain-JEPA's `data/gradient_mapping_450.csv`). `compute_gradients.group_gradient_path()` returns it by default; the HCP gradients below are optional, kept for later comparison.
  - ROI order checked: with the Schaefer-17N reading, UKB G1 is visual vs. the rest and G2 runs default mode ↔ somatomotor, and it matches our HCP placeholder (|r| = 0.84 / 0.63 for UKB G2 / G1 vs. our G1 / G2 on cortex). The 7N reading fits worse (0.56 / 0.47). Subcortical hemisphere order can't be told apart this way (UKB is left–right symmetric, r = 0.97); it follows Tian's own order in both.
- Optional HCP **placeholder** gradients from sub-100206 only, in `gradients/tianS3_schaefer400n17/`. Sanity checks look right: G1 runs somatomotor ↔ default mode, G2 is visual, and the LR and RL runs agree at r ≈ 0.6–0.7.
- Gradient port cross-checked against BrainSpace 0.3.0 (now installed in the env) on sub-100206's FC:
  - affinity and Procrustes match exactly;
  - the diffusion map matches at |corr| ≥ 0.9994 per component. Ours is the exact eigen-solution; BrainSpace runs a symmetric solver on a non-symmetric matrix.
  - `test_matches_brainspace` keeps this checked.
- `tests/test_temporal_bit.py` (29 tests; 49 in total with preprocessing, the dataloader and the masks, all passing) and `smoke_test_temporal_bit.py` (random input and a real window).
  - The tokenizer tests run for both voxel-embedding variants. One test checks that a full-rank table set to `codes @ W_up` gives exactly the low-rank output.
  - CPU only so far, B = 8 forward/backward: full rank 16.5 s, peak process memory 9.4 GiB; low rank (r = 64) 18 s, 8.9 GiB.

### JEPA pretraining (slide 2)
- `window_dataset.py`: `WindowDataset` cuts T-TR windows from the stored runs. Each item is a dict `x (T, V)` float16, `valid (V,)`, `tr`, `run`, `start`, and the default collate gives the model's inputs directly.
  - Runs: stored, at least T TRs long, `exclude` runs skipped by default, optional `subjects` list for a split.
  - `mode="random"` (training): n_tr // T windows per run per epoch, each at a random start. `mode="grid"` (validation): fixed starts every `stride` TRs.
  - No normalization: the stored data is already z-scored per voxel.
  - Tested on synthetic runs (`tests/test_window_dataset.py`, 4 tests). On the real data root it finds sub-100206's 2 runs (120 windows); 4 workers read them in 1.2 s, and a batch runs through the model.
- `roi_masks.py`: the Cross-ROI ("R") masks of slide 2 for one window. `ROIMaskSampler` returns disjoint, sorted `context (B, Kc)` and `target (B, Kt)` ROI indices; `ROIMaskCollator` is a `collate_fn` that adds them to a `WindowDataset` batch as `context_rois` / `target_rois`.
  - Ratios copied from Brain-JEPA (`MaskCollator_fmri` with `configs/ukb_vitb_ep300.yaml`): targets are 45–60 % of the ROIs (`pred_mask_R_scale`), the context keeps 84–100 % of the rest (`enc_mask_scale`), at least 4 context ROIs (`min_keep`). As in Brain-JEPA, ROIs are random (not contiguous), sizes are drawn once per batch and ROIs per sample. That gives about 170–250 context and 200–270 target ROIs out of 450.
  - One deviation: Brain-JEPA draws the context independently of the targets and truncates every sample to the batch's smallest context, which drops high-index ROIs first. Ours draws the context from the non-target ROIs at an exact size, so nothing is truncated (same expected size).
  - Brain-JEPA's time-axis masking (the beta/gamma blocks) has no counterpart, because the encoder pools time.
  - Tested (`tests/test_roi_masks.py`, 6 tests) and run on real data with the encoder.

## To do

### Data (blocked on the download or on decisions)
- [ ] The HCP download finishes, then the full processing run (`sbatch scripts/preprocess_hcp.sbatch`). This needs the user's explicit go-ahead.
- [ ] Decide whether the frozen group mask is final. It must not change after training starts, because V, the stored runs and the voxel embeddings all depend on it.
- [ ] Optional: run the `atlas` step on the data root, which adds one ~1 MB folder (needs approval).
- [ ] Define the train/test subject split.
- [ ] Later, to compare with UKB: compute the HCP group gradients: `compute_gradients.py --group --train-subjects <file>`. If version 2 is used, also compute the per-run gradients (`--subject`).

### Encoder leftovers
- [ ] Measure GPU memory and speed with the smoke test on a GPU that is visible. The 2026-10-29 session also had `CUDA_VISIBLE_DEVICES` set to empty, even though `nvidia-smi` lists an L40S.
- [ ] Later: compare full-rank voxel embeddings (default) with low rank (`voxel_rank=64` or other r) once pretraining runs.
- [ ] Only if version 2 is used: choose a leak option and, if needed, add "gradients from another run of the same subject".
- [ ] Only if profiling shows the need: speed up the tokenizer by grouping ROIs by size.

### JEPA pretraining (slide 2)
- [ ] A predictor (mask token + gradient positional embedding), an EMA target encoder (`roi_keep=None`), the loss and the training loop. Use the group gradients.
- [ ] Optional: skip or down-weight high-motion windows using the stored per-TR FD (`qc/*_fd.npy`); the dataloader does not use it yet.
- [ ] Later: cross-time targets from several windows of the same scan (the minute grid in slide 2): the "T" and "D" masks, in `roi_masks.py` next to the "R" masks.

## Open decisions
- Is the group mask final?
- The train/test split.
- Gradients: UKB (Brain-JEPA) for now. Later, test HCP group gradients and per-subject gradients (and which leak option) against it.
- When to move from T = 20 (14.4 s) to the slide-2 "minute" (≈ 83 TRs).
- Context size: Brain-JEPA's (`context_scale=(0.84, 1.0)`, the default) or slide 2's, where every non-target ROI is visible (`context_scale=(1, 1)`).
- Voxel embeddings: full rank (default, as in Brain-IT) or low rank (`voxel_rank`). Decide after comparing them in pretraining.
- Adding the cerebellum later: this changes N, the voxel embeddings and the gradients.
