#!/usr/bin/env bash
# Slurm job: build the original, muffled and reversed clips (x0, x2, rev) of every
# segment (CPU). First checks x0 against the listening check of the pinned segments.
# Writes $LBW_WORKSPACE/views/<view>/run-<job id>/ for each view and prints aggregate
# numbers only. Runs session after session in one process: about 2.4 minutes on a Mac
# core with synthetic data. Not yet timed on OMNI (17,649 files to write and read back),
# so it asks for the short partition's default hour.
#
# Submit from an OMNI login node:
#   scripts/slurm/submit.sh build_views
#
#SBATCH --job-name=lbw-build-views
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
source scripts/slurm/common.sh

python scripts/build_views.py
