#!/usr/bin/env bash
# Slurm job: GPU smoke test. Shows the GPU, then runs the GPU tests, on synthetic clips
# only: Voxtral loads in fp16 with fp32 position weights, "Yes" and "No" are single
# tokens, and clips get finite scores; and for the encoder caches, z from the cache equals
# z from the audio bit for bit for clips of 1 to 37.4 s, the encoder alone gives the
# output it gives inside the model and repeats its bits, the token IDs of a silent clip
# are the processor's own, the processor's stand-in builds the real processor's features,
# and the log-mel features repeat at the jobs' 4 threads. The tests set the numeric
# settings with encoder_cache.cpu_threads, so the 4 CPUs below stay equal to it.
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
