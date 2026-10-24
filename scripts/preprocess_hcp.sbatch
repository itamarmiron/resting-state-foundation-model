#!/bin/bash
# Denoise all HCP runs into rsfm_voxel/ as a Slurm job array (resumable: finished runs are skipped).
# Prerequisites, run once from the repo root:
#   python -m rsfm.preprocessing.cli inventory --config configs/preprocessing/hcp.yaml
#   python -m rsfm.preprocessing.cli mask      --config configs/preprocessing/hcp.yaml
# Submit from the repo root:
#   sbatch scripts/preprocess_hcp.sbatch
# Keep NUM_SHARDS equal to the --array size.
#SBATCH --job-name=rsfm_hcp
#SBATCH --partition=normal.q
#SBATCH --array=0-39
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/preprocess_hcp_%A_%a.out

NUM_SHARDS=40

set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

/home/itamarm/miniconda3/envs/py39/bin/python -m rsfm.preprocessing.cli run \
    --config configs/preprocessing/hcp.yaml --num-shards "$NUM_SHARDS"
