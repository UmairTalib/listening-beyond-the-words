#!/usr/bin/env bash
# Slurm job: the zero-shot scores (GPU). The untrained Voxtral scores every segment of x0,
# x1 and rev from the pinned encoder caches, one clip at a time, after a check that scores
# a sample from the audio and from the cache, which must agree bit for bit. It checks in
# code that it may run: only after the rules of the training code are dated in the pilot
# protocol (configs/pilot/train.yaml exists) and the commit that records the GPU tests and
# the training timing run has set their repeatability mode (precision.deterministic is no
# longer pending), once the checks of x1 are finished (the pinned word-check run), and
# while no cache check flagged the language model. Writes
# $LBW_WORKSPACE/runs/pilot/pilot-zeroshot-<YYYYmmdd-HHMMSS>/ and prints label-free
# aggregate numbers only. Estimate from step 7's measured rates: 45-50 min; the 2 h limit
# leaves room, because a job killed at its limit loses its run. The 4 CPUs are those of the
# cache jobs (encoder_cache.cpu_threads), so that the check builds the log-mel features as
# they did.
#
# Submit from an OMNI login node:
#   scripts/slurm/submit.sh score_zeroshot
#
#SBATCH --job-name=lbw-score-zeroshot
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
python scripts/score_zeroshot.py "$@"
