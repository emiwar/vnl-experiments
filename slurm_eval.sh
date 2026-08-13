#!/bin/bash
# Produce artifacts for a list of runs, on the cluster, next to the checkpoints.
#
#   sbatch slurm_eval.sh <run-list> [kind] [--set key=json ...]
#
# <run-list> is what `artifacts plan` wrote on the laptop:
#
#   python -m vnl_experiments.artifacts plan --kind eval \
#       --runs analysis/<question>/runs.csv --out todo.txt
#   scp todo.txt cluster:$SCRATCH/vnl-experiments/
#   ssh cluster 'cd $SCRATCH/vnl-experiments && sbatch slurm_eval.sh todo.txt eval'
#
# Results land in $VNL_ARTIFACTS (set below) with their .meta.json sidecars, so
# provenance travels with the bytes. Bring them home with:
#
#   python -m vnl_experiments.artifacts pull --kind eval --runs analysis/<question>/runs.csv
#
# Nothing here needs the repo's git state to be clean, and no manifest is written on
# this side -- `pull` reindexes from the sidecars on arrival.

#SBATCH -J vnlEval
#SBATCH -p gpu,gpu_h200 -t 0-08:00 --mem=48000 -c 4 -o slurm_logs/%j.out -e slurm_logs/%j.err --gres=gpu

set -euo pipefail

RUN_LIST="${1:?usage: sbatch slurm_eval.sh <run-list> [kind] [--set key=json ...]}"
KIND="${2:-eval}"
shift 2 || shift 1

source /n/holylfs06/LABS/olveczky_lab/Users/ewarnberg/python_venvs/jax_etc/bin/activate
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# The store root. Keep this on holylfs06 next to the checkpoints, and point
# $VNL_CLUSTER_ARTIFACTS at it from the laptop so `pull` knows where to look.
export VNL_ARTIFACTS="${VNL_ARTIFACTS:-/n/holylfs06/LABS/olveczky_lab/Users/ewarnberg/artifacts}"

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "kind=$KIND runs=$RUN_LIST store=$VNL_ARTIFACTS"

srun python -m vnl_experiments.artifacts ensure \
    --kind "$KIND" --runs "$RUN_LIST" "$@"
