#!/bin/bash
#SBATCH --job-name=bm_scvelo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=80G
#SBATCH --partition=cpu
#SBATCH --output=output-scvelo.out

script=B03_scvelo_benchmarking.py
environment=velvet_env1

python_path=/camp/home/maizelr/.conda2/my_envs/$environment/bin
$python_path/python $script