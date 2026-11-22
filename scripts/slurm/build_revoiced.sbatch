#!/usr/bin/env bash
# Slurm job: render the re-voiced clip x1 of every segment (CPU). Array task t renders
# the sessions that revoice.assign_tasks gives task t (TASKS = 20, all at once), into
# $LBW_WORKSPACE/views/x1/run-<array job id>/, and prints aggregate numbers only. When
# every task is done, scripts/merge_revoiced.py checks and merges them on a login node.
# If any task fails, nothing is merged, and the whole array runs again.
#
# Submit from an OMNI login node, with the check run that passed. Set the time to 1.5
# times the check job's estimate for the heaviest task (at most 6 h, the limit of short)
# and the memory from the check job's peak (sacct MaxRSS):
#   scripts/slurm/submit.sh [--time=HH:MM:SS] [--mem=<size>] build_revoiced derived/revoice_check/run-<job id>
#
#SBATCH --job-name=lbw-build-revoiced
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-19%20
set -euo pipefail
source scripts/slurm/common.sh

: "${SLURM_ARRAY_TASK_ID:?submit this job as an array}"
python scripts/build_revoiced.py "$@"
