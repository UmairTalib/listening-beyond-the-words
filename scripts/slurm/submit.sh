#!/usr/bin/env bash
# submit.sh: submit a job file from scripts/slurm/ with the project's log location.
#
# Usage:   scripts/slurm/submit.sh [--sbatch-option=value ...] <job> [job arguments]
# Example: scripts/slurm/submit.sh --job-name=lbw-train-cpg train configs/pilot/cpg.yaml 1
#
# <job> names the file scripts/slurm/<job>.sbatch. Give sbatch options in the
# --option=value form, before the job name. Run it on an OMNI login node.
# Logs go to $LBW_WORKSPACE/logs/slurm/<job name>-<job id>.log. Every job is submitted
# with --no-requeue: Slurm on OMNI would otherwise requeue a job after a node failure
# under the same job ID, and the requeued job would meet its own half-written output.
set -euo pipefail
umask 077

: "${LBW_WORKSPACE:?LBW_WORKSPACE is not set; see docs/engineering_conventions.md}"
usage="usage: submit.sh [--sbatch-option=value ...] <job> [job arguments]"
opts=()
while [ $# -gt 0 ] && [[ "$1" == --* ]]; do
  opts+=("$1")
  shift
done
job="${1:?$usage}"
shift

cd "$(git rev-parse --show-toplevel)"
file="scripts/slurm/$job.sbatch"
[ -f "$file" ] || { echo "no job file $file" >&2; exit 1; }
if [ -n "$(git status --porcelain)" ]; then
  echo "the checkout has uncommitted changes; commit them on the Mac and pull first" >&2
  exit 1
fi

logs="$LBW_WORKSPACE/logs/slurm"
mkdir -p "$logs"
LBW_COMMIT="$(git rev-parse HEAD)"
export LBW_COMMIT
sbatch --output="$logs/%x-%j.log" ${opts[@]+"${opts[@]}"} --no-requeue "$file" "$@"
