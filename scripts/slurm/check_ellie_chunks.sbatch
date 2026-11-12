#!/usr/bin/env bash
# Slurm job: score short speech chunks inside the clips of a listening check (GPU).
# Writes $LBW_WORKSPACE/derived/ellie_chunks/run-<job id>/ and prints numbers by clip number only.
#
# Submit from an OMNI login node:
#   scripts/slurm/submit.sh check_ellie_chunks derived/listening_check/run-<UTC time>
#
#SBATCH --job-name=lbw-check-ellie-chunks
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
source scripts/slurm/common.sh

python scripts/check_ellie_chunks.py "${1:?usage: check_ellie_chunks.sbatch <listening run>}"
