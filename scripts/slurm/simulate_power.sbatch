#!/usr/bin/env bash
# Slurm job: the main study's power, simulated again with the pilot's numbers (CPU): two
# checks first, then r fitted at each ρ and the simulated main studies at every true gain and
# α (docs/pilot_diagnostics.md, section "The main study's power"). It reads only the two
# published tables reports/pilot/summary.csv and reports/pilot/fold_values.csv of the checkout,
# writes $LBW_WORKSPACE/derived/power/run-<job id>/ and prints aggregate numbers only. Submit
# from a checkout at the newest v0.1.x tag with its environment built there (git checkout
# v0.1.x, then bash scripts/setup_omni_env.sh); the job refuses any other checkout. One worker
# per CPU, started by spawn with one BLAS thread each. Not yet timed on OMNI. On a laptop
# (Apple M2, synthetic inputs, the study design's middle scenario) with all 8 cores busy, a
# simulated study at its 11 true gains took 15 ms at 2,000 resamples and 88 ms at 10,000, and
# a pilot's bootstrap standard error 8 and 47 ms. With the resample check's fallback to
# 10,000 resamples, which it took on those synthetic inputs, the whole job (23 scenarios of
# 8,000 studies, the fits and the checks) takes about 5.7 process-hours: about 21 minutes on
# 16 workers at the laptop's speed, and about 85 minutes if OMNI's cores are 4 times slower.
# The limit is 1.5 times that, rounded up to 2.5 hours, within short's 6 hours; each worker
# needs about 200 MB.
#
# Submit from an OMNI login node, once:
#   scripts/slurm/submit.sh simulate_power
#
#SBATCH --job-name=lbw-simulate-power
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=02:30:00
set -euo pipefail
source scripts/slurm/common.sh

python scripts/simulate_power.py "$@"
