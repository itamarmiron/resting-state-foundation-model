#!/bin/bash
#SBATCH --job-name=rsfm_manager
#SBATCH --partition=irani_run.q
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
#SBATCH --output=/net/mraid16/mnt/pool-16/iranig_workspace/HCP_1200/derivatives/rsfm_voxel/jobs/logs/rsfm_manager_%j.log
#
# Keeps denoising up with fMRIPrep. Every interval_sec (jobs/manager.conf) it
#   1. runs `status reconcile` (refreshes fMRIPrep states; jobs that died -> PREEMPTED or FAILED),
#   2. counts this user's rsfmd_* jobs on irani_risk.q (pending + running),
#   3. submits `status next` subjects until max_active are active.
# PREEMPTED subjects are resubmitted automatically, FAILED ones never. `touch jobs/STOP` pauses it.
# It keeps waiting while fMRIPrep produces more subjects, and exits only when every subject is
# finished. Before its 7-day limit it submits its successor and exits.
#   sbatch scripts/preprocess_manager.sbatch

set -uo pipefail   # no -e: one failed command must not stop the manager

REPO=/home/itamarm/resting_state_model
JOBS=/net/mraid16/mnt/pool-16/iranig_workspace/HCP_1200/derivatives/rsfm_voxel/jobs
PY=/home/itamarm/miniconda3/envs/py39/bin/python
ME=${SLURM_JOB_ID:-manual$$}
T0=$(date +%s)
RENEW_AFTER=$((6 * 86400 + 12 * 3600))   # 6.5 days

cd "$REPO" || exit 1
STATUS=("$PY" -m rsfm.preprocessing.status)
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
conf() { awk -F= -v k="$1" '{sub(/#.*/, "")} $1 ~ "^[ \t]*" k "[ \t]*$" {gsub(/[ \t]/, "", $2); print $2}' "$JOBS/manager.conf"; }

others=$(squeue -h -u "$USER" -n rsfm_manager -t RUNNING -o %A | grep -vx "$ME" || true)
if [[ -n $others ]]; then
    log "Another manager is already running (job(s) $others); exiting."
    exit 0
fi
log "Manager $ME on $(hostname -s)"

while true; do
    max_active=$(conf max_active); max_active=${max_active:-8}
    interval=$(conf interval_sec); interval=${interval:-900}

    "${STATUS[@]}" reconcile || log "WARNING: reconcile failed"

    if ! q=$(squeue -h -u "$USER" -p irani_risk.q -o %j); then
        log "WARNING: squeue failed; skipping this loop"
        sleep "$interval"; continue
    fi
    active=$(grep -c '^rsfmd_' <<< "$q" || true)
    free=$((max_active - active))
    submitted=0
    if [[ -e $JOBS/STOP ]]; then
        log "STOP file present: not submitting (active $active)"
    elif (( free > 0 )); then
        mapfile -t todo < <("${STATUS[@]}" next --n "$free")
        for S in "${todo[@]}"; do
            [[ -n $S ]] || continue
            if squeue -h -u "$USER" -n "rsfmd_$S" -o %A | grep -q .; then
                log "rsfmd_$S already in the queue; skipping"
                continue
            fi
            if jid=$(sbatch --parsable -J "rsfmd_$S" "$REPO/scripts/preprocess_subject.sbatch" "$S"); then
                "${STATUS[@]}" queued "$S" --jobid "$jid" || log "WARNING: could not mark $S queued"
                log "submitted $S as job $jid"
                submitted=$((submitted + 1))
            else
                log "WARNING: sbatch failed for $S; will retry next loop"
                break
            fi
        done
    fi
    log "loop: active $active, max $max_active, submitted $submitted"

    remaining=$("${STATUS[@]}" remaining) || remaining=unknown
    if [[ $remaining == 0 ]] && (( active == 0 && submitted == 0 )); then
        log "Every subject is finished; manager exits."
        exit 0
    fi
    if (( $(date +%s) - T0 > RENEW_AFTER )); then
        if nj=$(sbatch --parsable --dependency=afterany:"$ME" "$REPO/scripts/preprocess_manager.sbatch"); then
            log "Time limit approaching: submitted successor manager $nj; exiting."
            exit 0
        fi
        log "WARNING: could not submit the successor manager; continuing"
    fi
    sleep "$interval"
done
