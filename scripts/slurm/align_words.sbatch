#!/usr/bin/env bash
# Slurm job: align the words of every kept participant line to the audio (GPU).
# Writes $LBW_WORKSPACE/derived/word_times/run-<job id>/ and prints aggregate numbers only.
#
# Fetch the models first, on a login node:
#   "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/fetch_models.py
# Submit from an OMNI login node:
#   scripts/slurm/submit.sh align_words
#
#SBATCH --job-name=lbw-align-words
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
set -euo pipefail
source scripts/slurm/common.sh

python scripts/align_words.py
