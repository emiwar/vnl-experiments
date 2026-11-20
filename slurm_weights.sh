#!/bin/bash
# Export network weights for a list of runs, on the cluster, next to the checkpoints.
#
#   sbatch slurm_weights.sh <run-list> [--steps all|last|every:N|<int>,...]
#
# The CPU-only counterpart of slurm_eval.sh for the `weights` kind: one flat .npz per
# checkpoint step, with its .meta.json sidecar (see vnl_experiments/artifacts/weights.py).
# Steps already exported are skipped, unless their checkpoint was rewritten since (a
# requeue rewind), in which case they are re-exported -- so re-submitting this for runs
# still training is cheap and picks up only the new checkpoints.
#
# Bring the results home with:
#
#   python -m vnl_experiments.artifacts pull --kind weights --runs <run-list> [--steps ...]
#
# <run-list> is anything `--runs` accepts; `artifacts plan --kind weights --out todo.txt`
# writes one.

#SBATCH -J vnlWeights
#SBATCH -p shared,sapphire,olveczky,olveczky_sapphire -t 0-04:00 --mem=16000 -c 4 -o slurm_logs/%j.out -e slurm_logs/%j.err

set -euo pipefail

RUN_LIST="${1:?usage: sbatch slurm_weights.sh <run-list> [--steps SEL]}"
shift
# Every checkpoint unless told otherwise: exporting is cheap and the laptop chooses
# what to pull.
if [[ " $* " != *" --steps "* ]]; then
    set -- --steps all "$@"
fi

source /n/holylfs06/LABS/olveczky_lab/Users/ewarnberg/python_venvs/new_mujoco/bin/activate
# Exporting reads orbax arrays and writes numpy; no device is needed, and asking JAX
# for none keeps it from probing for a GPU that this partition does not have.
export JAX_PLATFORMS=cpu

# Same store as slurm_eval.sh; point $VNL_CLUSTER_ARTIFACTS at it from the laptop.
export VNL_ARTIFACTS="${VNL_ARTIFACTS:-/n/holylfs06/LABS/olveczky_lab/Users/ewarnberg/artifacts}"

echo "host=$(hostname) runs=$RUN_LIST store=$VNL_ARTIFACTS args=$*"

srun python -m vnl_experiments.artifacts ensure --kind weights --runs "$RUN_LIST" "$@"
