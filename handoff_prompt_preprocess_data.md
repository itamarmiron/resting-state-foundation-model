# Handoff: preprocessing session (2026-10-20)

Paste this into a new Claude Code session started in `/home/itamarm/resting_state_model` to
pick up where the previous session stopped. Read `CLAUDE.md` and `docs/PREPROCESSING_PLAN.md`
first; they hold the design and the commands. This file covers the session's state and what to
do next.

## Goal

This repo will hold a new **foundation model for resting-state fMRI**. The current task is
preprocessing: turn fMRIPrep outputs into **denoised voxel data** (the final output), with an
**optional atlas step** (cortex + subcortex) applied to the stored voxels afterwards. It is
**not for SwiFT**: no 96³ crop, no min-max scaling, no per-TR `.pt` files.

The design was adapted from the rsPTSD SwiFT pipeline in
`/home/itamarm/ptsd_project/rsPTSD/Preprocessing/` (`swift_fmriprep.py` for denoising,
`swift_preprocessing.py` for tensors). Read access to that repo is pre-approved.

## Decisions made with the user

- Denoising: 6 motion + 5 "combined" aCompCor + fMRIPrep cosine regressors (high-pass ≈ 0.008 Hz),
  Butterworth low-pass 0.1 Hz (configurable), detrend, voxelwise z-score. No scrubbing (FD per TR
  is saved instead), no GSR by default. TR read from the BOLD JSON sidecar.
- Keep the native fMRIPrep grid: MNI152NLin2009cAsym, 97×115×97, 2 mm. No crop, no smoothing.
- Atlases must be the **2009c** versions (TemplateFlow Schaefer, Tian 3T 2009cAsym). They sit on
  exactly the data grid. The FSL-space (nilearn) atlases were measured to misplace parcels:
  median 1.6 mm, max 3.7 mm shift, per-parcel Dice as low as 0.43 for Schaefer-400.
- Default atlas `schaefer400_tianS3` (450 regions). Any Schaefer/Tian combination works, e.g.
  `schaefer100_tianS1`. The atlas never changes the voxel data.
- Output dir: `/net/mraid16/mnt/pool-16/iranig_workspace/HCP_1200/derivatives/rsfm_voxel/`.
- High-motion runs are stored anyway; `exclude` in the manifest is only a flag.

## User constraints (important)

- **Do not run on the full dataset** until the user explicitly says so. `scripts/preprocess_hcp.sbatch`
  is written but must not be submitted without that go-ahead.
- **The HCP download / fMRIPrep run is still in progress.** Never hardcode subject or run counts.
  Re-run `inventory` to pick up new runs; finished runs are skipped by `run`.

## State at handoff

Code (all **uncommitted**; nothing has been committed yet):
- `rsfm/preprocessing/`: `inventory.py`, `denoise.py`, `voxel_store.py`, `atlas.py`, `cli.py`, `config.py`
- `configs/preprocessing/hcp.yaml`, `scripts/preprocess_hcp.sbatch` (40-shard array on `normal.q`, not submitted)
- `tests/test_preprocessing.py`: 11 tests on synthetic data, all passing
- `CLAUDE.md`, `docs/PREPROCESSING_PLAN.md` updated to match the code

Data already written to `rsfm_voxel/`:
- `manifest.tsv`: 807 runs / 211 subjects at the time (count will grow), no missing files,
  all TR 0.72 s, n_TR 941–1200, 2 runs flagged for mean FD > 0.5 mm.
- `grid/mask.nii.gz` + `grid.json`: frozen group mask, 234,961 voxels (99.6% of the template brain
  mask), built from those 807 runs. Rule: template mask ∩ in-brain in ≥ 50% of runs.

Downloaded: Tian S1–S4 atlases in `~/.cache/rsfm/atlases/tian`. Schaefer is already in
`~/.cache/templateflow`.

**Smoke test in flight:** Slurm job **306196** (`normal.q`, 4 CPU, 32 GB) denoises 2 runs:
`sub-100206_ses-1_task-rest_dir-LR` and `..._dir-RL`. Log: `logs/smoke_306196.out` (includes
`/usr/bin/time -v`, so look for "Elapsed" and "Maximum resident set size"). A first attempt via a
backgrounded `srun` hung without Slurm creating a job; use `sbatch` (or `srun` from an
interactive terminal).

## Next steps

1. Check job 306196: `sacct -j 306196 -o JobID,State,Elapsed,MaxRSS,ExitCode -P` and the log.
   Report to the user: run time per run, peak memory, % valid voxels, any failures. Check that
   `voxel/sub-100206/ses-1/*_bold.npy` is (T, 234961) float16 and `qc/.../*_fd.npy` exists.
   If streaming the gzip turns out slow (no `indexed_gzip` installed), note that.
2. Prepare the atlas and parcellate the 2 runs:
   `cli atlas --name schaefer400_tianS3`, then
   `cli parcellate --name schaefer400_tianS3 --run-id <the 2 runs>`.
   Report regions with low mask coverage and the overlap count from `atlas.json`.
3. Sanity checks to show the user (from the plan):
   - `parcellate()` vs `NiftiLabelsMasker` on the cleaned image.
   - FC: strong left–right homotopic correlations, DMN present, thalamo-cortical coupling.
   - Visual overlay of mask and atlas on the template.
4. From the measured memory/time, set `--mem` / `--time` / array size in
   `scripts/preprocess_hcp.sbatch`. **Still don't submit** until the user says so.
5. Open question to raise: the group mask is frozen from the runs available today. It's
   99.6% of the template mask, so later subjects shouldn't matter much. An alternative is a
   run-independent mask (the template mask alone); changing it now is cheap because only the 2
   smoke-test runs would need redoing.
6. Ask the user before committing (they haven't asked for commits yet).

## Open decisions still with the user
Low-pass 0.1 Hz vs none (HCP's TR of 0.72 s allows up to ~0.7 Hz) · GSR variant ·
motion threshold (0.5 mm is lenient) · atlas resolution (default 400 + S3).
