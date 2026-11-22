#!/usr/bin/env bash
# Slurm job: check the re-voiced clip x1 before it is built (CPU). A dry run of Kokoro's
# text front end on every part, then a test render of 20 segments, the first 10 of which
# the researcher hears. Writes $LBW_WORKSPACE/derived/revoice_check/run-<job id>/ and
# prints aggregate numbers only. Kokoro has run on OMNI only on a GPU node's CPU so far,
# so the hour asked for is a guess; the job prints its own timing for the build.
#
# Submit from an OMNI login node:
#   scripts/slurm/submit.sh check_revoice
#
#SBATCH --job-name=lbw-check-revoice
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
source scripts/slurm/common.sh

python scripts/check_revoice.py
