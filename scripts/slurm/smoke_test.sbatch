#!/usr/bin/env bash
# Slurm job: GPU smoke test. Shows the GPU, then runs the GPU tests: Voxtral loads
# in fp16, "Yes" and "No" are single tokens, and synthetic clips get finite scores.
#
# Fetch the models first, on a login node:
#   "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/fetch_models.py
# Submit from an OMNI login node:
#   scripts/slurm/submit.sh smoke_test
#
#SBATCH --job-name=lbw-smoke-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
source scripts/slurm/common.sh

nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
python -c "import torch; print('torch', torch.__version__, '| GPU', torch.cuda.get_device_name(0))"
python -m pytest -m gpu -p no:cacheprovider
