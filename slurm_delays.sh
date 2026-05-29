#!/bin/bash
#SBATCH -J rlDelays
#SBATCH -p gpu,olveczkygpu -t 0-2:00 --mem=32000 -c 4 -o slurm_logs/%j.out -e slurm_logs/%j.err --gres=gpu

source /n/holylfs06/LABS/olveczky_lab/Users/ewarnberg/python_venvs/jax_etc/bin/activate
srun python vnl_experiments/delays/train_delays.py --env CartpoleBalance --delay 0

