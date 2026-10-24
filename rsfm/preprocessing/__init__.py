"""fMRIPrep derivatives -> denoised voxel data, with optional atlas parcellation.

Pipeline (see docs/PREPROCESSING_PLAN.md):
    inventory   -> manifest.tsv, one row per run
    mask        -> grid/mask.nii.gz, the shared group brain mask (frozen once built)
    run         -> voxel/<sub>/<ses>/<run_id>_bold.npy, T x V float16, V = voxels in group mask
    atlas       -> atlas/<name>/labels.npy, one region label per stored voxel (optional)
    parcellate  -> atlas/<name>/timeseries/<run_id>_timeseries.npy (optional cache)
"""
