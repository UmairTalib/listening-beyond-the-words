#!/usr/bin/env bash
# Slurm job: the word check (GPU). Whisper large-v3 and wav2vec2 read the clip versions of
# the word-check sample, opensmile measures the voice of x0 and x1, and the job prints the
# verdicts of check F4, the x0 control and the checks of x1. Writes
# $LBW_WORKSPACE/derived/wordcheck/run-<job id>/ and prints aggregate numbers only. The
# job is expected to take about 2.5 h; 12 h leaves room for windows that Whisper decodes
# again, because a job killed at its limit loses the whole run.
#
# Submit from an OMNI login node:
#   scripts/slurm/submit.sh check_views
#   scripts/slurm/submit.sh check_views --views <view> ...
#   scripts/slurm/submit.sh --time=<printed limit> check_views --rerun derived/wordcheck/run-<job id>
#
#SBATCH --job-name=lbw-check-views
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
set -euo pipefail
source scripts/slurm/common.sh

nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
python scripts/check_views.py "$@"
