# Preprocessing plan: fMRIPrep → denoised voxel data (+ optional atlas)

Goal: turn fMRIPrep resting-state outputs into denoised **voxel data** for the new
foundation model. Parcellating with an atlas is an **optional step applied to the stored voxel
data**, never part of preprocessing, so any atlas can be used later without re-denoising.

Adapted from the rsPTSD SwiFT pipeline (`/home/itamarm/ptsd_project/rsPTSD/Preprocessing/`),
minus everything that existed only for SwiFT (96³ crop, per-scan min-max scaling, one `.pt`
file per TR).

```
fMRIPrep derivatives
  → preprocess.py (inventory → denoise → voxel store)   ══► VOXEL DATA (final output)
                                                              │
                                    optional, any time later  ▼
                         parcellate(voxel_run, atlas)  →  T × R parcel time series
```

## Input / output

| | Path |
|---|---|
| Input (HCP) | `/net/mraid16/mnt/pool-16/iranig_workspace/HCP_1200/derivatives/fmriprep_rest` |
| Output (HCP) | `/net/mraid16/mnt/pool-16/iranig_workspace/HCP_1200/derivatives/rsfm_voxel/` (new dir) |

Input facts (checked 2026-10-20): fMRIPrep 25.2.5, `space-MNI152NLin2009cAsym_res-2`,
grid 97×115×97 at 2 mm, not skull-stripped, TR 0.72 s, 1200 TRs per run, 2 runs (`dir-LR`,
`dir-RL`) per session. 209 subjects / 799 runs present so far (the full fMRIPrep run is
still in progress).

## Part A — `preprocess.py` (output: voxel data)

**A0. Inventory.** One manifest row per run (`manifest.tsv`): subject, session, dir, file paths, **TR from the
`.json` sidecar** (NIfTI headers are unreliable), n_TR, mean FD, % high-motion volumes,
exclusion flag. Every later step reads the manifest. Exclusion is a flag, not a deletion:
  high-motion runs are still denoised and stored.

**A1. Denoise (in memory, cleaned 4D is never saved).**
- Confounds read explicitly from `*_desc-confounds_timeseries.tsv` (nilearn 0.10.1's
  `load_confounds` cannot find the file when the name has a `dir-` entity):
  6 motion (`trans_*`, `rot_*`) + 5 `a_comp_cor_*` (combined WM+CSF) + all `cosine*`, demeaned.
- **High-pass ≈ 0.008 Hz** via fMRIPrep's cosine regressors (128 s cutoff), inside the
  regression. No extra `high_pass` in `clean_img`.
- **Low-pass 0.1 Hz** (configurable) via `clean_img(low_pass=0.1, t_r=<TR from JSON>)`, plus
  linear detrend, the run's fMRIPrep brain mask, and voxelwise z-scoring over time.
- No scrubbing (it breaks continuous windows); FD per TR is saved instead.
  No global signal regression by default.

**A2. Store voxels.**
- Native fMRIPrep grid (97×115×97, 2 mm, MNI152NLin2009cAsym). **No crop, no warp, no
  smoothing.**
- One **group brain mask**: template brain mask ∩ voxels in-brain in ≥ 50% of runs (per-run
  masks extend well outside the brain). Built once and **frozen**; runs added later reuse it.
  Only in-mask voxels are stored (234,961 voxels for HCP).
- Per run: `T × V` float16 matrix + a per-run validity vector for voxels outside that run's
  own mask. ≈ 0.56 GB per 1200-TR HCP run → ≈ 0.45 TB for the current 807 runs,
  ≈ 2.5 TB for full HCP.

Output layout:
```
rsfm_voxel/
  manifest.tsv
  grid/  mask.nii.gz, grid.json                     # group mask, shape, affine, V
  voxel/ sub-X/ses-Y/sub-X_ses-Y_dir-LR_bold.npy    # T × V float16
         sub-X/ses-Y/sub-X_ses-Y_dir-LR_valid.npy   # V bool
  qc/    FD per TR per run
  atlas/ <atlas_name>/...                           # only if the optional cache is built
```

## Part B — atlas step (optional, on top of the voxel data)

**B1. Prepare an atlas once** → one label per stored voxel + a labels table (name, network,
cortex/subcortex). Use **MNI152NLin2009cAsym** versions, not nilearn's FSL-space ones
(FSL → 2009c misplaces parcels: median 1.6 mm, max 3.7 mm, per-parcel Dice down to 0.43 for
Schaefer-400). TemplateFlow's 2009c `res-02` Schaefer atlases sit on exactly the data grid,
so no resampling is needed; assert this.
- Cortex: Schaefer-2018 (100 / 200 / 400 parcels, 7 or 17 networks), from TemplateFlow.
- Subcortex: Tian S1–S4, 3T, 2009c release (downloaded to `~/.cache/rsfm/atlases/tian`;
  same grid as the data).
- Default first atlas: Schaefer-400 + Tian S3 (450 ROIs, as in Brain-JEPA). Schaefer-100 +
  Tian S1 is equally supported. Check that cortex and subcortex labels don't overlap.

**B2. `parcellate(voxel_run, atlas)`** = mean of *valid* voxels per parcel (one sparse
matmul), then z-score each parcel over time. Also returns per-parcel coverage.
Usable offline (`parcellate.py --atlas <name>` → `atlas/<name>/…_timeseries.npy`) or on the
fly in the dataset (`--input voxel | atlas:<name>`).

## Validation (before the full run)
1. 2–3 HCP runs + 1 PTSD run: visual check of mask, filtering, atlas overlay; subcortex in mask.
2. `parcellate()` from stored voxels ≈ `NiftiLabelsMasker` on the cleaned image.
3. FC sanity: strong homotopic correlations, DMN present, thalamo-cortical coupling.
4. Measure memory per worker on a 1200-TR run, then set the Slurm worker count.

## Code layout
Package with a per-dataset config (paths, filename pattern, confound names, TR source) so
HCP / PTSD / UCLA share one code path: `inventory.py`, `denoise.py`, `voxel_store.py`
(→ `preprocess.py`), `atlas.py` (prep + `parcellate()` + cache script), a dataset class
(voxel or atlas input), and a resumable Slurm job-array launcher.

## Steps
1. Write the code.
2. Test on 2–3 runs; review outputs together.
3. Full run on Slurm into `rsfm_voxel/`; re-run as more fMRIPrep subjects finish.

## Defaults for open decisions
Low-pass 0.1 Hz (configurable; dropping it keeps HCP signal up to ~0.7 Hz) · no GSR ·
flag runs with mean FD > 0.5 mm · first atlas Schaefer-400 + Tian S3.
