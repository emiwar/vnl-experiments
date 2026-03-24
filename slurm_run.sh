#!/bin/bash
#SBATCH -J rodentImitation
#SBATCH -p gpu -t 1-12:00 --mem=128000 -c 4 -o slurm_logs/%j.out -e slurm_logs/%j.err --gres=gpu

source /n/holylfs06/LABS/olveczky_lab/Users/ewarnberg/python_venvs/jax_etc/bin/activate
srun python vnl_experiments/modular/nervenet_v3_test.py
