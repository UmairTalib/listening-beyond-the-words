# shellcheck shell=bash
# common.sh: shared setup for every Slurm job of this project.
#
# Source it near the top of a job file, after `set -euo pipefail`:
#   source scripts/slurm/common.sh
# Submit jobs with scripts/slurm/submit.sh, which runs sbatch from the repository root.

umask 077
: "${LBW_WORKSPACE:?LBW_WORKSPACE is not set; see docs/engineering_conventions.md}"
cd "${SLURM_SUBMIT_DIR:?submit this job with scripts/slurm/submit.sh}" || exit 1

# The job must run the code that was checked out when it was submitted.
if [ -n "${LBW_COMMIT:-}" ] && [ "$(git rev-parse HEAD)" != "$LBW_COMMIT" ]; then
  echo "the checkout moved since submission (expected $LBW_COMMIT); resubmit the job" >&2
  exit 1
fi

# Start from a clean module environment: the PyTorch wheels bring their own CUDA libraries.
if command -v module > /dev/null 2>&1; then
  set +u
  module purge
  set -u
fi

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$LBW_WORKSPACE/env/venv}"
# shellcheck source=/dev/null
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

export HF_HOME="$LBW_WORKSPACE/hf"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "job ${SLURM_JOB_ID:-?} (${SLURM_JOB_NAME:-?}) on $(hostname)," \
  "commit $(git rev-parse --short HEAD), started $(date '+%F %T')"
