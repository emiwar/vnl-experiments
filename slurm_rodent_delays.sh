#!/bin/bash
#SBATCH -J rodentDelays
#SBATCH -p gpu,gpu_h200 -t 0-12:00 --mem=48000 -c 4 -o slurm_logs/%j.out -e slurm_logs/%j.err --gres=gpu
#
# One rodent delays run on a dedicated partition, where it will not be preempted.
# For gpu_requeue, use slurm_rodent_requeue.sh instead.
#
# Usage (arguments are Hydra overrides, passed straight through):
#   sbatch slurm_rodent_delays.sh delay=100
#   sbatch slurm_rodent_delays.sh net=forward_model delay=5 net.fm_loss_weight=0.0

source /n/holylfs06/LABS/olveczky_lab/Users/ewarnberg/python_venvs/jax_etc/bin/activate
export MUJOCO_GL=egl
srun python -m vnl_experiments.train "$@"
