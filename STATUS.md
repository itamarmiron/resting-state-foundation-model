# Project status

Last updated: 2026-11-01

What exists in this repo and what is left to do. Update it at the end of every session that changes anything (see CLAUDE.md).

## Done

### Preprocessing (`rsfm/preprocessing/`)
- CLI steps `inventory` → `mask` → `run` → optional `atlas` / `parcellate`, with unit tests (`tests/test_preprocessing.py`). Design: `docs/PREPROCESSING_PLAN.md`.
- HCP so far:
  - The upstream run (download + fMRIPrep, `HCP_1200/jobs/full/STATUS.md`) is still in progress: 681 / 1,096 subjects on 2026-11-01. It runs by itself; this repo only reads its `status.tsv`.
  - **Interim split** `splits/hcp_interim_v1*` (seed 0, stratified by complete/partial): train 876, val 110, test 110, over all 1,096 subjects with rest data, processed or not. **It ignores families, so relatives can be in different sets and results are provisional** (see To do).
  - **Group mask rebuilt 2026-11-01** from the 546 train subjects with finished fMRIPrep (2,103 runs): V = 235,026, was 234,961 (83 voxels added, 18 removed). Frozen. The old mask, manifest and 2 stored runs are in `rsfm_voxel/archive_pre_split_20261101/`.
  - **Denoising manager started 2026-11-01** on the user's go-ahead; live progress is in `PREPROCESSING_STATUS.md`, so counts in this file go stale. `scripts/preprocess_manager.sbatch` (`irani_run.q`) submits `scripts/preprocess_subject.sbatch` workers (`irani_risk.q`, up to 8) as fMRIPrep finishes subjects. Live table: `PREPROCESSING_STATUS.md` (generated, not in git).
  - Tested on real data, 2 subjects stored (100206 train, 100307 val; 8 runs): about 2–3 min per subject (40–50 s per run). `sacct` MaxRSS 18–28 GiB, though the worker's own peak is 8.5 GiB; the job limit is 40 GB. Voxels z-scored (std 1.000), invalid voxels 0. Cancelling a worker mid-subject and resubmitting it skipped the finished run.

### Temporal BIT encoder (slide 1 of `Temporal_BIT_JEPA_widescreen.pdf`)
- `temporal_bit.py`: `TemporalBITConfig` (T = 20 by default), `BrainTokenizer`, `TemporalTransformer`, `SpatialTransformer`, `TemporalBIT`.
  - `(B, T, V)` → `(B, T, N, 512)` → `(B, N, 512)` → `(B, N, D)`; 102 M parameters, of which 72 M are the voxel embeddings.
  - Voxel embeddings: one per labelled MNI voxel (V′ = 140,535), shared across subjects and TRs.
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
- `tests/test_temporal_bit.py` (29 tests; 64 in total across the repo, all passing) and `smoke_test_temporal_bit.py` (random input and a real window).
  - The tokenizer tests run for both voxel-embedding variants. One test checks that a full-rank table set to `codes @ W_up` gives exactly the low-rank output.
  - CPU only so far, B = 8 forward/backward: full rank 16.5 s, peak process memory 9.4 GiB; low rank (r = 64) 18 s, 8.9 GiB.

### JEPA pretraining (slide 2)
- `window_dataset.py`: `WindowDataset` cuts T-TR windows from the stored runs. Each item is a dict `x (T, V)` float16, `valid (V,)`, `tr`, `run`, `start`, and the default collate gives the model's inputs directly.
  - Runs: stored, at least T TRs long, `exclude` runs skipped by default, optional `subjects` list for a split.
  - `mode="random"` (training): n_tr // T windows per run per epoch, each at a random start. `mode="grid"` (validation): fixed starts every `stride` TRs.
  - No normalization: the stored data is already z-scored per voxel.
  - Tested on synthetic runs (`tests/test_window_dataset.py`, 4 tests). On the real data root it finds sub-100206's 2 runs (120 windows); 4 workers read them in 1.2 s, and a batch runs through the model.
- `roi_masks.py`: the Cross-ROI ("R") masks of slide 2 for one window. `ROIMaskSampler` returns disjoint, sorted `context (B, Kc)` and `target (B, Kt)` ROI indices; `ROIMaskCollator` is a `collate_fn` that adds them to a `WindowDataset` batch as `context_rois` / `target_rois`.
  - These ratios and this sampling method are provisional (see Open decisions).
  - Ratios copied from Brain-JEPA (`MaskCollator_fmri` with `configs/ukb_vitb_ep300.yaml`): targets are 45–60 % of the ROIs (`pred_mask_R_scale`), the context keeps 84–100 % of the rest (`enc_mask_scale`), at least 4 context ROIs (`min_keep`). As in Brain-JEPA, ROIs are random (not contiguous), sizes are drawn once per batch and ROIs per sample. That gives about 170–250 context and 200–270 target ROIs out of 450.
  - One deviation: Brain-JEPA draws the context independently of the targets and truncates every sample to the batch's smallest context, which drops high-index ROIs first. Ours draws the context from the non-target ROIs at an exact size, so nothing is truncated (same expected size).
  - Brain-JEPA's time-axis masking (the beta/gamma blocks) has no counterpart, because the encoder pools time.
  - Tested (`tests/test_roi_masks.py`, 6 tests) and run on real data with the encoder.
- `predictor.py`: `Predictor`, a small ViT after Brain-JEPA's `VisionTransformerPredictor`, with its config values (depth 6, width 384, 12 heads; 11 M parameters). `(B, Kc, D)` context features → `(B, Kt, D)` predictions.
  - One query per target ROI: a shared mask token + the ROI's gradient position, always from the group gradients. Self-attention runs over context and queries together.
  - This replaces slide 2's "narrow BIT cross-transformer". There is no minute offset (Δt) yet, because all targets are in the context's window.
  - It takes any ROI index lists, so it does not depend on the masking choices. Tested in `tests/test_predictor.py` (7 tests).
- `ema_target_encoder.py`: `EMATargetEncoder`, a frozen copy of the encoder, as in Brain-JEPA's `train.py`. It always encodes all ROIs, applies a parameter-free LayerNorm over D, then reads `target_rois`.
  - `update(encoder, momentum)` averages the parameters; `ema_momentum(step, total_steps)` rises linearly from 0.996 to 1.0.
  - One deviation: it is always in eval mode, so targets stay deterministic if dropout is enabled.
  - Tested in `tests/test_ema_target_encoder.py` (8 tests).
- The encoder, predictor and target encoder ran together for one step on real sub-100206 windows, with a throwaway MSE loss and SGD (full-size model, B = 4, 163 context and 265 target ROIs, CPU): finite outputs, 7.8 s, peak process memory 4.7 GiB. This is a wiring check, not the real loss or a training run.
- `train_jepa.py` (config `configs/pretraining/hcp.yaml`, tests `tests/test_train_jepa.py`): smooth L1 loss, AdamW, gradient accumulation (8 × B = 16), LR warmup + cosine and cosine weight decay, all as in Brain-JEPA and counted in optimizer steps; the schedules match Brain-JEPA's code exactly. Saves `latest.pt` every 500 optimizer steps and resumes from it.
  - Deviations: EMA momentum counts optimizer steps so it reaches 1.0 (Brain-JEPA stops near 0.9965 at 8 accumulation steps); no GradScaler with bf16; non-finite losses stop the run; one GPU only. bf16 applies on GPU only, since CPU autocast rejects float16 input.
- GPU memory measured on an H200 (2026-11-01, Slurm job 324659): full JEPA steps with AdamW on random data, 190 context and 240 target ROIs, excluding data loading. Peak allocated GiB and s/step:
  - fp32, T = 20: B = 16 10.1 GiB, 0.39 s; B = 64 32.9 GiB, 0.91 s.
  - fp32, T = 83: B = 16 28.7 GiB, 0.82 s; B = 32 54.8 GiB, 1.43 s.
  - bf16, T = 83: B = 16 19.6 GiB, 0.41 s; B = 32 36.7 GiB, 0.60 s.
  - Attention backend at T = 83, B = 16: flash and PyTorch's memory-efficient kernel have the same peak; plain matmul + softmax costs 2.4 GiB more in fp32 and 3.1 GiB more in bf16. Our current `scaled_dot_product_attention` is already at the low end, so we keep it.
  - Low rank (r = 64) saves only 1.6–3.3 GiB and ran about 2–4x slower per step.

## To do

### Data
- [ ] Denoising finishes (running; follows fMRIPrep until all 1,096 subjects are done). FAILED subjects need a human: see `PREPROCESSING_STATUS.md`. The manager is `squeue -u $USER -n rsfm_manager`; restart it with `sbatch scripts/preprocess_manager.sbatch` if it stops with work left.
- [ ] **Future: get the HCP restricted CSV (`Family_ID`) and replace the interim split with a family-aware one**, as a new split version. Keep the restricted file out of git. All subjects are stored regardless of set, so no denoised data is lost. The frozen mask keeps a small trace of train subjects who move out; that was accepted.
- [ ] Optional: run the `atlas` step on the data root, which adds one ~1 MB folder (needs approval).
- [ ] Later, to compare with UKB: compute the HCP group gradients: `compute_gradients.py --group --train-subjects <file>`. If version 2 is used, also compute the per-run gradients (`--subject`).

### Encoder leftovers
- [ ] Later: compare full-rank voxel embeddings (default) with low rank (`voxel_rank=64` or other r) once pretraining runs.
- [ ] Only if version 2 is used: choose a leak option and, if needed, add "gradients from another run of the same subject".
- [ ] Only if profiling shows the need: speed up the tokenizer by grouping ROIs by size.
- [ ] Potential: pivot to Brain-IT's attention implementation (xFormers `memory_efficient_attention`, `models/layers_transformer.py` in `brainit-fmri`; Brain-IT pins `xformers==0.0.22.post7`). Not needed for memory so far: the measured PyTorch kernels already match the most efficient case, but xFormers itself is untested here. Check first that it works with torch 2.1.2+cu121 and how the CPU tests would run.

### JEPA pretraining (slide 2)
- [ ] **Investigate possible representation collapse before any long run.** In a 100-step check on one subject (240 windows, job 324762), `target_std` fell 0.56 → 0.058, `pred_std` 0.28 → 0.036 and the loss 0.13 → 0.001. So few data may exaggerate this, but it is what collapse looks like. Check on more subjects, and compare with the settings (LR, EMA momentum, predictor).
- [ ] A long training run needs a fixed `subjects_file`: `train_jepa.py` refuses to resume if the dataset size changed, and it grows while denoising runs.
- [ ] Measure data-loading throughput and host RAM on real data with many workers. A T = 83, B = 16 batch is 624 MB, so at 0.4 s/step this is the likelier bottleneck.
- [ ] Decide the masking ratios and sampling method (see Open decisions), then finalize `roi_masks.py`.
- [ ] Optional: skip or down-weight high-motion windows using the stored per-TR FD (`qc/*_fd.npy`); the dataloader does not use it yet.
- [ ] Later: cross-time targets from several windows of the same scan (the minute grid in slide 2): the "T" and "D" masks, in `roi_masks.py` next to the "R" masks.

## Open decisions
- Gradients: UKB (Brain-JEPA) for now. Later, test HCP group gradients and per-subject gradients (and which leak option) against it.
- When to move from T = 20 (14.4 s) to the slide-2 "minute" (≈ 83 TRs).
- **Future decision: masking ratios and sampling method.** `roi_masks.py` currently uses Brain-JEPA's values as placeholders. Still open:
  - the target fraction (now 45–60 %);
  - the context size: Brain-JEPA's (`context_scale=(0.84, 1.0)`, the default) or slide 2's, where every non-target ROI is visible (`context_scale=(1, 1)`);
  - how ROIs are sampled (now uniformly random, not contiguous or network-based).
- Voxel embeddings: full rank (default, as in Brain-IT) or low rank (`voxel_rank`). Decide after comparing them in pretraining.
- Adding the cerebellum later: this changes N, the voxel embeddings and the gradients.
