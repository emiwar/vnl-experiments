#!/bin/bash
#SBATCH -J rodentDelays
#SBATCH -p gpu,gpu_h200 -t 0-12:00 --mem=48000 -c 4 -o slurm_logs/%j.out -e slurm_logs/%j.err --gres=gpu

source /n/holylfs06/LABS/olveczky_lab/Users/ewarnberg/python_venvs/jax_etc/bin/activate
export MUJOCO_GL=egl
srun python vnl_experiments/delays/train_rodent_forward_model.py --delay 100 --no-detach-prediction --fm-loss-weight 0.0
