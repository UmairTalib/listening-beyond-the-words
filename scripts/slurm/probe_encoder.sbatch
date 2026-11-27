#!/usr/bin/env bash
# Slurm job: the pooled-encoder probe (CPU): does Voxtral's frozen audio encoder pass on the
# voice? Ridge and logistic probes on the pooled caches of x0, x2 and rev, with their controls
# and intervals (docs/pilot_diagnostics.md, section "Pooled-encoder probe"). It writes
# $LBW_WORKSPACE/derived/probe/run-<job id>/ and prints aggregate numbers only. Submit from a
# checkout at the newest v0.1.x tag with its environment built there (git checkout v0.1.x,
# then bash scripts/setup_omni_env.sh, which installs opensmile); the job refuses any other
# checkout. One process on 2 CPUs. Not yet timed on OMNI: opensmile on every x0 clip is the
# largest part (about 1.75 h at the word check's rate), the three caches about 13 minutes,
# and the fits with their controls and 10,000 resamples took 4.7 minutes on a laptop with 2
# threads at the real size (synthetic data), so about 2.5 h in all and a limit of 4 hours.
#
# Submit from an OMNI login node, once:
#   scripts/slurm/submit.sh probe_encoder
#
#SBATCH --job-name=lbw-probe-encoder
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
set -euo pipefail
source scripts/slurm/common.sh

python scripts/probe_encoder.py "$@"
