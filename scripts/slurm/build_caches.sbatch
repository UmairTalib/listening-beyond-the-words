#!/usr/bin/env bash
# Slurm job: the encoder caches (GPU). With --view, Voxtral's frozen audio encoder reads
# every clip of one pinned clip version, one clip at a time, and its output before the
# adapter is stored in fp16, one file per segment, in $LBW_WORKSPACE/cache/<view>/run-<job id>/.
# With --verify, a sample of each named cache run is computed again from the audio and
# compared byte for byte with the stored files; then the language model runs without its
# output layer on the longest clips, with the overflow watch on. That mode writes only
# $LBW_WORKSPACE/derived/cache_check/run-<job id>/run.json. The job prints aggregate
# numbers only. Estimates, not yet measured: 12-20 min per view and about 8 min for the
# check; the 2 h limit leaves room, because a job killed at its limit loses its run. The
# 4 CPUs are the 4 PyTorch threads of encoder_cache.cpu_threads, which build the log-mel
# features.
#
# Submit from an OMNI login node, the four views side by side:
#   scripts/slurm/submit.sh build_caches --view <view>
#   scripts/slurm/submit.sh --time=01:00:00 build_caches --verify cache/<view>/run-<job id> ...
#
#SBATCH --job-name=lbw-build-caches
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
source scripts/slurm/common.sh

nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
python scripts/build_caches.py "$@"
