#!/bin/bash
# Denoise one subject (resumable). Submitted by preprocess_manager.sbatch as:
#   sbatch -J rsfmd_<S> scripts/preprocess_subject.sbatch <S>
#SBATCH --partition=irani_risk.q
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/net/mraid16/mnt/pool-16/iranig_workspace/HCP_1200/derivatives/rsfm_voxel/jobs/logs/%x_%j.log

set -uo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /home/itamarm/resting_state_model || exit 1
exec /home/itamarm/miniconda3/envs/py39/bin/python -u -m rsfm.preprocessing.worker "$1" \
    --config configs/preprocessing/hcp.yaml
