#!/bin/bash
#SBATCH -J dmcRequeue
#SBATCH -p gpu_requeue -t 0-12:00 --mem=16000 -c 4 --gres=gpu
#SBATCH --requeue --open-mode=append
#SBATCH -o slurm_logs/%j.out -e slurm_logs/%j.err
#SBATCH --signal=USR1@120
#
# Preemption-safe dm_control_suite training on the requeue partition.
#
# The dm_control counterpart of slurm_rodent_requeue.sh, and sized like
# slurm_dmc_delays.sh (16 GB rather than the rodent's 48): asking for more only
# costs queue priority on a sweep this wide. Worth using when a run will not fit
# a dedicated allocation comfortably -- HumanoidWalk at 1 G steps is ~8.8 h on an
# A100 and ~6.3 h on an H200, against a 12 h limit -- or simply because
# gpu_requeue is nearly free.
#
# gpu_requeue runs on idle dedicated nodes, so this job will be killed and
# requeued whenever a node's owner wants it back. Slurm then re-runs this script
# from the top with the *same* job id, which is what the trainer keys its run
# directory and WandB run on -- so every attempt continues the same run rather
# than starting a new one.
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
# On resume, a light checkpoint's env states are redrawn rather than restored:
# each env gets a fresh episode with its phase drawn over the whole episode
# length, and the network carry (delay / efference queues, any RNN state) starts
# empty. Both are bounded transients -- see analysis/dm_control_suite/README.md.
# Pass requeue.full_checkpoints=true to keep the env states instead, at the cost
# of a much larger and slower save.
#
# Usage (arguments are Hydra overrides, passed straight through):
#   sbatch slurm_dmc_requeue.sh env=dmc/humanoid_walk net=flat_forward_model \
#       train=dmc delay=10 train.ppo.total_steps=1000000000
#   sbatch slurm_dmc_requeue.sh env=dmc/walker_walk net=flat_recurrent train=dmc \
#       delay=15 net.rnn_cell=gru
#
# For a whole delay sweep:
#   python -m vnl_experiments.sweep --script slurm_dmc_requeue.sh \
#       env=dmc/humanoid_walk net=delayed_mlp train=dmc delay=0,5,7,10,15,20

source /n/holylfs06/LABS/olveczky_lab/Users/ewarnberg/python_venvs/jax_etc/bin/activate
export MUJOCO_GL=egl

# Compiling ppo_step, the eval rollout and the render scan costs minutes, and
# every attempt would otherwise pay it again. The cache is keyed on the HLO, so
# it is shared across the runs of a sweep with the same shapes.
export JAX_COMPILATION_CACHE_DIR=/n/holylfs06/LABS/olveczky_lab/Users/ewarnberg/jax_cache

srun python -m vnl_experiments.train requeue.enabled=true "$@"
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
