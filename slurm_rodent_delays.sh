#!/bin/bash
#SBATCH -J rodentDelays
#SBATCH -p gpu -t 0-12:00 --mem=32000 -c 4 -o slurm_logs/%j.out -e slurm_logs/%j.err --gres=gpu

source /n/holylfs06/LABS/olveczky_lab/Users/ewarnberg/python_venvs/jax_etc/bin/activate
export MUJOCO_GL=egl
srun python vnl_experiments/delays/train_rodent_delays.py --delay 50
