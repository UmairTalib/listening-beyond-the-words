#!/usr/bin/env bash
# Slurm job: unpack the audio and transcripts of the 189 pilot sessions (CPU only).
#
# Submit from the repository root:
#   WS=$(ws_find cpg-thesis)
#   sbatch --output="$WS/edaic/manifest/unpack-%j.log" \
#     scripts/slurm/unpack_edaic_pilot.sbatch "$WS/edaic"
#
#SBATCH --job-name=unpack-edaic
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -uo pipefail
umask 077

EDAIC="${1:?usage: sbatch [--output=...] unpack_edaic_pilot.sbatch <edaic_dir>}"
cd "${SLURM_SUBMIT_DIR:?submit this file with sbatch}" || exit 1
echo "job ${SLURM_JOB_ID:-?} on $(hostname) with ${SLURM_CPUS_PER_TASK:-8} CPUs, started $(date '+%F %T')"
bash scripts/unpack_edaic_pilot.sh "$EDAIC" "${SLURM_CPUS_PER_TASK:-8}"
