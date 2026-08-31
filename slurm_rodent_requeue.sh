#!/bin/bash
#SBATCH -J rodentRequeue
#SBATCH -p gpu_requeue -t 0-12:00 --mem=48000 -c 4 --gres=gpu
#SBATCH --requeue --open-mode=append
#SBATCH -o slurm_logs/%j.out -e slurm_logs/%j.err
#SBATCH --signal=USR1@120
#
# Preemption-safe training on the requeue partition.
#
# gpu_requeue runs on idle dedicated nodes, so this job will be killed and
# requeued whenever a node's owner wants it back. Slurm then re-runs this script
# from the top with the *same* job id, which is what
# train_rodent_requeue.py keys its run directory and WandB run on -- so every
# attempt continues the same run rather than starting a new one.
#
# What the directives above are for:
#   --requeue           let Slurm requeue this job when it is preempted
#   --open-mode=append  every attempt appends to one log (%j is stable too)
#   --signal=USR1@120   warn 120 s before the *time limit*. Note there is no
#                       "B:" prefix: with it the signal goes to the batch shell
#                       only, and the python process -- the one that has to write
#                       the checkpoint -- would never hear about it. Preemption
#                       itself arrives as SIGTERM; the script handles both by
#                       checkpointing and exiting 42.
#
# Usage:
#   sbatch slurm_rodent_requeue.sh --delay 5
#   sbatch slurm_rodent_requeue.sh --network RodentEncDecRecurrent --net-config rnn_cell=gru
#
# All arguments are passed through to train_rodent_requeue.py.

source /n/holylfs06/LABS/olveczky_lab/Users/ewarnberg/python_venvs/jax_etc/bin/activate
export MUJOCO_GL=egl

# Compiling ppo_step, the eval rollout and the render scan costs minutes, and
# every attempt would otherwise pay it again. The cache is keyed on the HLO, so
# it is shared across the runs of a sweep with the same shapes.
export JAX_COMPILATION_CACHE_DIR=/n/holylfs06/LABS/olveczky_lab/Users/ewarnberg/jax_cache

srun python -m vnl_experiments.delays.train_rodent_requeue "$@"
code=$?

# 42 = "saved my state, there is work left". Requeue explicitly rather than
# relying on the scheduler having flagged the job: this covers the time-limit
# path too, and is harmless when Slurm has already requeued us. Every other
# non-zero code is a real failure and is left to fail the job, so a crash loop
# is impossible.
if [ $code -eq 42 ]; then
    echo "=== interrupted and saved; requeueing job $SLURM_JOB_ID ==="
    scontrol requeue "$SLURM_JOB_ID" || true
    exit 0
fi

exit $code
