#!/usr/bin/env bash
# Slurm job: analyse the pilot's test scores (CPU): fold AUROCs, Δ, the paired bootstrap's
# intervals, checks F1 to F5 and the decision rule (docs/pilot_analysis.md). It writes
# $LBW_WORKSPACE/derived/evaluate/run-<job id>/ (trial-<job id>/ with --trial) and prints
# aggregate numbers only. Submit from a checkout at the newest v0.1.x tag with its environment
# built there (git checkout v0.1.x, then bash scripts/setup_omni_env.sh), once the kept runs
# are pinned in configs/pilot/evaluate.yaml and squeue shows no lbw-train job; the job
# refuses any other checkout. Not yet timed on OMNI: a few minutes at most is the estimate
# (10,000 resamples of 10 fold-runs, about 38 participants each), so the limit is 30 minutes.
#
# Submit from an OMNI login node, the trial first, then the analysis once:
#   scripts/slurm/submit.sh evaluate --trial
#   scripts/slurm/submit.sh evaluate
#
#SBATCH --job-name=lbw-evaluate
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
source scripts/slurm/common.sh

python scripts/evaluate.py "$@"
