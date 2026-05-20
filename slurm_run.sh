#!/bin/bash
#SBATCH -J rodentImitation
#SBATCH -p gpu -t 0-16:00 --mem=64000 -c 4 -o slurm_logs/%j.out -e slurm_logs/%j.err --gres=gpu

source /n/holylfs06/LABS/olveczky_lab/Users/ewarnberg/python_venvs/jax_etc/bin/activate
srun python vnl_experiments/modular/nervenet_v4_test.py
