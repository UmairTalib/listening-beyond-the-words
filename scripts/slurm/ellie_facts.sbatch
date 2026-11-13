#!/usr/bin/env bash
# Slurm job: facts and template libraries of the interviewer detector (GPU).
# Writes $LBW_WORKSPACE/derived/ellie_detector/facts-<job id>/ and prints aggregate numbers only.
#
# Submit from an OMNI login node:
#   scripts/slurm/submit.sh ellie_facts
#
#SBATCH --job-name=lbw-ellie-facts
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
source scripts/slurm/common.sh

python scripts/ellie_facts.py
