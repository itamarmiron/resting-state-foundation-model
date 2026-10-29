# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

resting-state-foundation-model — the goal of this project is to develop a new foundation model for resting-state fMRI.

There is a preprocessing package (`rsfm/preprocessing/`), the Temporal BIT encoder (top-level `temporal_bit.py`, `positional_embedding.py`, `compute_gradients.py`) its window dataloader (`window_dataset.py`) and the ROI masks for pretraining (`roi_masks.py`). The rest of the JEPA pretraining (slide 2 of `Temporal_BIT_JEPA_widescreen.pdf`: predictor, EMA target encoder, loss, training loop) is not written yet.

## Environment and commands

Use the existing conda env (nilearn 0.10.1, nibabel, scipy, pandas, pyyaml, torch, einops; brainspace 0.3.0 is used only to cross-check `compute_gradients.py` in tests); there is no install step. Run everything from the repo root:

```bash
PY=/home/itamarm/miniconda3/envs/py39/bin/python
$PY -m pytest tests -q                                                      # unit tests (synthetic data)
$PY -m rsfm.preprocessing.cli <step> --config configs/preprocessing/hcp.yaml   # steps below
sbatch scripts/preprocess_hcp.sbatch                                        # full HCP denoising (job array)
$PY compute_gradients.py --placeholder                                      # brain gradients from sub-100206 only
$PY smoke_test_temporal_bit.py                                              # model shapes on random + real data
```

Don't launch the full-dataset run unless the user asks for it explicitly.

## Preprocessing

Design and rationale: `docs/PREPROCESSING_PLAN.md`. The final output is **voxel data**; an atlas is an optional step applied to it afterwards.

CLI steps, in order: `inventory` (writes `manifest.tsv`, one row per run) → `mask` (frozen group mask in `grid/`) → `run` (denoise and store voxels; resumable; shard with `--num-shards`) → optional `atlas --name schaefer400_tianS3` → optional `parcellate --name ...`.

Things that aren't obvious from the code:
- All data, the TemplateFlow Schaefer atlases and the Tian atlases are in MNI152NLin2009cAsym on the same 97×115×97, 2 mm grid. Nothing is resampled, and the code asserts the grids match. Don't use nilearn's FSL-space atlases.
- Stored runs are `T × V` float16 over the group-mask voxels in C order (`volume[mask]`). The group mask is frozen once built: rebuilding it invalidates every stored run.
- TR comes from the BOLD JSON sidecar; NIfTI headers can be wrong. Confounds are read from the TSV, and aCompCor columns are chosen by `Mask == "combined"` in its JSON. nilearn's `load_confounds` fails on `dir-` filenames.
- High-motion runs are stored anyway: `exclude` in the manifest is a flag for consumers.
- Tian atlas files are downloaded to `~/.cache/rsfm/atlases/tian` (override with `RSFM_TIAN_DIR`); Schaefer comes from `~/.cache/templateflow`.

HCP output: `/net/mraid16/mnt/pool-16/iranig_workspace/HCP_1200/derivatives/rsfm_voxel/`.

## Temporal BIT model

Input: one window `(B, T, V)` of stored voxels (T = `TemporalBITConfig.n_timepoints`), plus the run's `valid` mask `(B, V)`. Output: `(B, K, D)`, one feature per kept ROI.

Things that aren't obvious from the code:
- The atlas is `tianS3_schaefer400n17` (N = 450, subcortex first, 17-network order, same layout as Brain-JEPA). It hasn't been built on the data root yet, so `compute_gradients.load_atlas` builds a local copy in `cache/` and switches to the data root's copy once `atlas` has run there. Unlabelled voxels (white matter, cerebellum, brainstem…) never enter the model. **The cerebellum is excluded on purpose for now** (about 19 k grey-matter voxels): there is no cerebellar atlas component. Adding one later changes N, the voxel-embedding table and the gradients.
- Start with T = 20 (14.4 s at HCP's 0.72 s TR). The slide-2 "minute" (≈ 83 TRs) is a later decision; T is only a config value.
- Voxel embeddings are one row per labelled MNI voxel, shared across subjects and TRs. Changing the atlas or the group mask means a new model. They are full rank by default (`voxel_rank=None`, a free 512-dim row per voxel, as in Brain-IT, whose code is at `/home/itamarm/brainit-fmri`). `voxel_rank=r` is our own low-rank option, codes `(V′, r)` @ a shared `W_up`, kept for a later comparison. The two variants have different state-dict keys (`voxel_embed` vs `voxel_codes` + `voxel_up`).
- Group gradients default to Brain-JEPA's precomputed UKB file, `gradients/tianS3_schaefer400n17/brainjepa_ukb.csv` (an unchanged copy of Brain-JEPA's `data/gradient_mapping_450.csv`, tracked in git; the rest of `gradients/` is generated and ignored). Its ROI order is Tian S3 then Schaefer-400 **17-network**, matching our atlas (checked against our HCP gradients; 7N order fits worse). `compute_gradients.group_gradient_path(source)` switches to the HCP versions, which stay optional for later testing: `"placeholder"` (sub-100206 only) or `"train"` (`--group --train-subjects`). `gradient_source="group"` uses one fixed `(N, 30)` matrix. `"subject"` takes per-run HCP gradients `(B, N, 30)` from the whole-scan FC; these leak cross-ROI and cross-time information into the JEPA context (see `compute_gradients.py`).
- `roi_keep` drops ROIs completely. Nothing may mix ROIs before the spatial transformer: no BatchNorm, and no spatial normalization in the dataloader. `tests/test_temporal_bit.py` checks this.

## Related data

`/home/itamarm/ptsd_project/rsPTSD` is a related resting-state PTSD project. It can be read freely for reference (read access is pre-approved in `.claude/settings.local.json`).

## Project status

`STATUS.md` lists what exists, what is left to do and the open decisions. Read it at the start of a session. Update it whenever something is completed, started, added to the to-do list or decided, without being asked: tick or remove finished items, add new ones, refresh "Last updated", and keep it describing the repo and data as they are now.

## Keeping this file current

After any change that affects setup, dependencies, build/test/run commands, architecture, data layout or conventions, update this file in the same change. Record only what a new contributor couldn't quickly work out from the code itself, and remove anything that has become outdated.
