#!/usr/bin/env bash
# Slurm job: train one arm of the pilot in one round, one fold-run per array task (GPU).
# Each task trains from the pinned encoder caches, saves LoRA and the adapter copy, and then
# scores its own test fold on x0 and rev; it writes
# $LBW_WORKSPACE/runs/pilot/pilot-<arm>-r<round>-f<fold>-<YYYYmmdd-HHMMSS>/ and logs
# aggregate numbers only. With --timing it is the timing run of check F5 (CPG, round 1,
# fold 1, one task), which writes $LBW_WORKSPACE/derived/train_timing/run-<job id>/run.json
# only. The timing run (job 12618228, 2026-11-25) measured a three-view step at 13.8 s in
# strict mode: about 1.04 h of training for 3 epochs and 1.4 h for 4, about 1.5 h with the
# loads and the scoring; the limit, 2.5 h, is about 1.5 times that (B0 runs take about a
# third). The 4 CPUs are those
# of the cache jobs (encoder_cache.cpu_threads), so that the numeric settings match the
# caches' record; 48G holds the scoring model built on the CPU. After the settings freeze,
# submit from a checkout at the newest v0.1.x tag (git checkout v0.1.x on OMNI); the job
# refuses any other.
#
# Submit from an OMNI login node, at most two arrays at a time (10 GPU tasks):
#   scripts/slurm/submit.sh --job-name=lbw-train-<arm> train configs/pilot/<arm>.yaml <round>
# and the timing run, held until the CPU and GPU tests pass:
#   scripts/slurm/submit.sh --job-name=lbw-train-cpg --array=0 --time=01:00:00 \
#     --dependency=afterok:<CPU-test job>:<GPU-test job> train configs/pilot/cpg.yaml 1 --timing
#
# Array task t trains fold t + 1 of the given round (5 fold-runs). CPG's round 1, fold 1
# runs on its own (--array=0): first before the freeze, and again only as the redo or the
# 8-segment rerun that the pilot protocol names. After the freeze, CPG's round 1 is
# submitted with --array=1-4, since fold 1 has trained.
#
#SBATCH --job-name=lbw-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --array=0-4
set -euo pipefail
source scripts/slurm/common.sh

config="${1:?usage: train.sbatch <config> <round> [--timing]}"
round="${2:?usage: train.sbatch <config> <round> [--timing]}"
shift 2
task="${SLURM_ARRAY_TASK_ID:?submit this job as an array}"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
python scripts/train.py --config "$config" --round "$round" --fold $((task + 1)) "$@"
