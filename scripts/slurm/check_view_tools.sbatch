#!/usr/bin/env bash
# Slurm job: check the clip-version tools on a fixed public text (GPU). Kokoro loads
# offline with every weight and its espeak fallback on, faster-whisper and wav2vec2 run
# in float16 and read the text back, and opensmile gives all 88 eGeMAPS values.
# Writes $LBW_WORKSPACE/derived/check_view_tools/run-<job id>/ with the text in each
# candidate voice. Never reads participant data.
#
# Fetch the models first, on a login node:
#   "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/fetch_models.py
# Submit from an OMNI login node:
#   scripts/slurm/submit.sh check_view_tools
#
#SBATCH --job-name=lbw-check-view-tools
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
source scripts/slurm/common.sh

nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
python scripts/check_view_tools.py
