#!/bin/bash
#SBATCH -J rodentDistillation
#SBATCH -p olveczkygpu -t 2-00:00 --mem=16000 -c 4 -o slurm_logs/%j.out -e slurm_logs/%j.err --gres=gpu

source /n/holylfs06/LABS/olveczky_lab/Users/ewarnberg/python_venvs/jax_etc/bin/activate
srun python vnl_experiments/distillation/train.py
