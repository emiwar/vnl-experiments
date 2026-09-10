#!/bin/bash
#SBATCH -J dmcDelays
#SBATCH -p gpu,gpu_h200 -t 0-12:00 --mem=16000 -c 4 --gres=gpu
#SBATCH -o slurm_logs/%j.out -e slurm_logs/%j.err
#
# One dm_control_suite delay run on a dedicated partition.
#
# Sized for the control suite rather than the rodent: these tasks are a fraction of the
# rodent's memory footprint, and asking for 48 GB / 12 h like slurm_rodent_delays.sh does
# only costs queue priority on a sweep this wide. Raise --mem for the humanoids if they
# turn out to need it.
#
# The env group pins wandb.project=nnx-ppo-delays, so a run lands in the control-suite
# project without having to remember the flag.
#
# Usage (arguments are Hydra overrides, passed straight through):
#   sbatch slurm_dmc_delays.sh env=dmc/cartpole_swingup net=delayed_mlp train=dmc delay=5
#   sbatch slurm_dmc_delays.sh env=dmc/walker_walk net=flat_forward_model train=dmc delay=10
#   sbatch slurm_dmc_delays.sh env=dmc/walker_walk net=flat_recurrent train=dmc delay=10 \
#       net.rnn_cell=gru
#
# For a whole delay sweep:
#   python -m vnl_experiments.sweep --script slurm_dmc_delays.sh \
#       env=dmc/cartpole_swingup net=delayed_mlp train=dmc delay=0,1,2,5,10,20
#
# There is no requeue variant: at ~6 h these fit inside a dedicated allocation, and the
# preemption machinery is only worth its complexity for the 12 h rodent runs. If you do
# want gpu_requeue, slurm_rodent_requeue.sh takes the same overrides -- it is a generic
# pass-through -- just add train=dmc and the env/net groups.

source /n/holylfs06/LABS/olveczky_lab/Users/ewarnberg/python_venvs/jax_etc/bin/activate
export MUJOCO_GL=egl
srun python -m vnl_experiments.train "$@"
