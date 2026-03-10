#!/bin/bash
#SBATCH -J rodentImitation
#SBATCH -p gpu,gpu_h200 -t 1-12:00 --mem=128000 -c 4 -o slurm_logs/%j.out -e slurm_logs/%j.err --gres=gpu

source /n/holylfs06/LABS/olveczky_lab/Users/ewarnberg/python_venvs/jax_etc/bin/activate
srun python vnl_experiments/modular/dense_mlp.py
